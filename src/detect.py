"""
Detección de rostros (Haar Cascade de OpenCV) y colores dominantes (K-Means).

Cambios importantes:
- `detect_faces` ya NO modifica la imagen que recibe ni devuelve una imagen:
  solo devuelve las cajas. Antes dibujaba sobre la imagen cacheada.
- El pipeline detecta sobre la imagen ORIGINAL y dibuja las cajas sobre la
  filtrada. Antes detectaba después de Sobel/Canny y casi nunca encontraba caras.
- El clasificador se carga perezosamente (una vez por proceso).
"""
import cv2
import numpy as np

from .cache import memoize

_cascades = {}
MAX_DETECT_SIDE = 1024  # se achica para detectar más rápido
MIN_FACE_RATIO = 0.05   # una cara tiene que medir al menos 5% del lado menor de la foto


def _cascade(name):
    """Carga un clasificador Haar una sola vez por proceso."""
    if name not in _cascades:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} no trae CascadeClassifier (se quitó en OpenCV 5). "
                'Instalá la versión 4: pip install "opencv-python-headless>=4.10,<5"'
            )
        _cascades[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _cascades[name]


def _overlaps(a, b, min_iou=0.3):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return union > 0 and inter / union >= min_iou


def _has_eyes(gray_full, x, y, w, h, crop_width=160):
    """Busca al menos un ojo en la mitad superior del candidato (a resolución completa)."""
    top = gray_full[y : y + h // 2, x : x + w]
    if top.size == 0:
        return False
    top = cv2.resize(top, (crop_width, max(1, int(top.shape[0] * crop_width / top.shape[1]))))
    eyes = _cascade("haarcascade_eye_tree_eyeglasses.xml").detectMultiScale(
        top, scaleFactor=1.1, minNeighbors=3, minSize=(14, 14)
    )
    return len(eyes) > 0


@memoize
def detect_faces(image):
    """
    Detecta rostros frontales.

    Haar Cascade es un método clásico (2001) y rápido, pero da falsos
    positivos en texturas (pelo de animales, pinceladas, follaje). Para
    reducirlos, un candidato del detector principal se acepta solo si lo
    confirma una segunda opinión: otro clasificador entrenado distinto
    (alt2) encuentra una cara en el mismo lugar, o se detecta un ojo en su
    mitad superior. Además se exige un tamaño mínimo proporcional a la foto.

    Returns:
        list[tuple[int, int, int, int]]: cajas (x, y, w, h) en coordenadas
        de la imagen original.
    """
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray_full
    scale = min(1.0, MAX_DETECT_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    side = max(30, int(min(gray.shape) * MIN_FACE_RATIO))

    candidates = _cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(side, side)
    )
    second = _cascade("haarcascade_frontalface_alt2.xml").detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(side, side)
    )

    faces = []
    for cand in candidates:
        cand = tuple(int(v) for v in cand)
        box = tuple(int(v / scale) for v in cand)
        if any(_overlaps(cand, tuple(o)) for o in second) or _has_eyes(gray_full, *box):
            faces.append(box)
    return faces


def draw_faces(image, faces, color=(43, 134, 200), thickness=None):
    """Devuelve una copia de la imagen con las cajas dibujadas."""
    out = image.copy()
    thickness = thickness or max(2, round(max(image.shape[:2]) / 300))
    for x, y, w, h in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


@memoize
def get_dominant_colors(image, k=5, sample=20000):
    """
    Colores dominantes (BGR) ordenados de más a menos frecuente.
    Se usa una muestra de píxeles para que K-Means no tarde una eternidad
    en imágenes grandes.
    """
    from sklearn.cluster import KMeans  # import perezoso: los workers arrancan más livianos

    pixels = image.reshape(-1, 3).astype(np.float32)
    if len(pixels) > sample:
        idx = np.random.default_rng(0).choice(len(pixels), sample, replace=False)
        pixels = pixels[idx]
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
    counts = np.bincount(kmeans.labels_, minlength=k)
    order = np.argsort(counts)[::-1]
    return np.clip(kmeans.cluster_centers_[order].round(), 0, 255).astype(int)