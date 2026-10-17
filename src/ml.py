"""
Clasificación de imágenes con CLIP (zero-shot).

CLIP compara la imagen contra frases de texto y devuelve qué frase describe
mejor la imagen. No hay entrenamiento: las categorías se definen acá abajo.

Mejoras:
- El modelo se carga perezosamente y una sola vez (antes se cargaba al importar
  el módulo, y como pipeline.py lo importaba, en Windows se cargaba en CADA
  proceso del pool).
- Las imágenes se convierten de BGR (OpenCV) a RGB antes de pasarlas a CLIP
  (antes le llegaban los colores invertidos).
- Se devuelve también la confianza de la predicción.
- Resultados cacheados por hash de imagen: solo se pasan por el modelo las
  imágenes que no se clasificaron antes. Se procesan en lotes de BATCH_SIZE.
- Inferencia con torch.no_grad() (menos memoria, más rápido).
"""
from .cache import array_hash, get_cache, stats
from .io_utils import cv2_to_pil

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16

# Categoría visible -> frase que se le pasa a CLIP
CATEGORIES = {
    "Persona": "a photo of a person",
    "Animal": "a photo of an animal",
    "Objeto": "a photo of an object",
    "Edificio": "a photo of a building",
    "Paisaje": "a photo of a landscape",
    "Dibujo o arte": "an abstract drawing or illustration",
}
LABELS = list(CATEGORIES.keys())
PROMPTS = list(CATEGORIES.values())

_model = None
_processor = None
_device = "cpu"


def load_model():
    """Carga CLIP una sola vez por proceso (descarga ~600 MB la primera vez)."""
    global _model, _processor, _device
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def _predict(images):
    import torch

    model, processor = load_model()
    inputs = processor(
        text=PROMPTS,
        images=[cv2_to_pil(img) for img in images],
        return_tensors="pt",
        padding=True,
    ).to(_device)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1).cpu()
    results = []
    for row in probs:
        best = int(row.argmax())
        results.append({"label": LABELS[best], "confidence": float(row[best])})
    return results


def classify_batch(images, use_cache=True):
    """
    Clasifica una lista de imágenes BGR.

    Returns:
        list[dict]: [{"label": "Persona", "confidence": 0.93}, ...]
                    (None en la posición de imágenes inválidas).
    """
    cache = get_cache()
    results = [None] * len(images)
    pending = []
    for i, img in enumerate(images):
        if img is None:
            continue
        key = ("clip", MODEL_NAME, tuple(PROMPTS), array_hash(img))
        cached = cache.get(key) if use_cache else None
        if cached is not None:
            stats["hits"] += 1
            results[i] = cached
        else:
            pending.append((i, key))

    for start in range(0, len(pending), BATCH_SIZE):
        chunk = pending[start : start + BATCH_SIZE]
        predictions = _predict([images[i] for i, _ in chunk])
        for (i, key), pred in zip(chunk, predictions):
            stats["misses"] += 1
            results[i] = pred
            if use_cache:
                cache.set(key, pred)
    return results
