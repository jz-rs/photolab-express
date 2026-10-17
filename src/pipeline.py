"""
Orquestación del procesamiento por lotes.

- `process_image` procesa UNA imagen: decodifica, detecta rostros sobre la
  original, aplica los filtros en orden y dibuja las cajas al final.
- Caché por foto completa (hash de los bytes + filtros elegidos). Además se
  memoiza la detección de rostros, cuyo resultado es chico (unas cajas).
  Antes también se memoizaba cada filtro, pero llenar la caché tardaba casi
  5 veces más que no usarla: guardar imágenes intermedias de varios MB
  costaba más que recalcularlas.
- `run_pipeline` es un GENERADOR: va devolviendo cada resultado apenas
  termina (as_completed), lo que permite actualizar la barra de progreso en
  la UI en vez de esperar a que termine todo el lote.
- En Windows los procesos se crean con "spawn" (cada worker arranca un
  intérprete nuevo e importa este módulo). Por eso este módulo NO importa
  nada pesado (ni CLIP, ni Streamlit) y el pool se reutiliza entre corridas.
"""
import hashlib
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

from . import cache, detect, filters, io_utils

# Por debajo de esta cantidad no vale la pena pagar el costo de procesos
MIN_IMAGES_FOR_PARALLEL = 3


def default_workers():
    """Cantidad de procesos. Se puede fijar con la variable PHOTOLAB_WORKERS."""
    env = os.environ.get("PHOTOLAB_WORKERS")
    if env:
        return max(1, int(env))
    return max(2, min(8, os.cpu_count() or 2))


def _init_worker():
    """
    Se ejecuta una vez al arrancar cada proceso del pool.
    OpenCV usa varios hilos internamente; con 8 procesos haciendo lo mismo
    se pisan entre ellos (oversubscription). Cada worker usa 1 hilo y el
    paralelismo lo maneja el pool.
    """
    cv2.setNumThreads(1)


def create_executor(max_workers=None):
    # "spawn" en todos los sistemas: es lo que usa Windows sí o sí, y en Linux/Mac
    # evita problemas de hacer fork de un servidor con hilos como Streamlit.
    return ProcessPoolExecutor(
        max_workers=max_workers or default_workers(),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )


def process_image(name, data, selected_filters, face_detection, use_cache=True):
    """
    Procesa una imagen a partir de sus bytes.

    Returns:
        dict con: name, ok, error, size, thumb_original, thumb_processed,
        processed_png, faces, seconds, cache_hits, cache_misses.
    """
    cache.set_enabled(use_cache)
    hits0, misses0 = cache.stats["hits"], cache.stats["misses"]
    start = time.perf_counter()

    # Nivel 1 de caché: el resultado completo de esta foto con estos ajustes.
    # La clave es el hash de los bytes del archivo (barato: no hace falta ni
    # decodificar la imagen). Si pega, nos salteamos decode, filtros y encode.
    deterministic = "Tono aleatorio" not in selected_filters
    key = ("process", hashlib.sha256(data).hexdigest(), tuple(selected_filters), bool(face_detection))
    if use_cache and deterministic:
        cached = cache.get_cache().get(key)
        if cached is not None:
            cache.stats["hits"] += 1
            return dict(cached, name=name, seconds=time.perf_counter() - start, cache_hits=1, cache_misses=0)

    result = {"name": name, "ok": False, "error": None, "faces": []}
    image = io_utils.decode_image(data)
    if image is None:
        result["error"] = "No es una imagen válida o el formato no está soportado"
        return result

    faces = detect.detect_faces(image) if face_detection else []

    processed = image
    for filter_name in selected_filters:
        func, _ = filters.FILTERS[filter_name]
        processed = func(processed)

    if faces:
        processed = detect.draw_faces(processed, faces)

    # Al proceso principal se devuelven miniaturas + PNG comprimido, no los
    # arrays completos: así se pickean muchos menos bytes entre procesos.
    result.update(
        ok=True,
        size=(image.shape[1], image.shape[0]),
        thumb_original=io_utils.thumbnail(image),
        thumb_processed=io_utils.thumbnail(processed),
        processed_png=io_utils.encode_png(processed),
        faces=faces,
        seconds=time.perf_counter() - start,
        cache_hits=cache.stats["hits"] - hits0,
        cache_misses=cache.stats["misses"] - misses0,
    )
    if use_cache and deterministic:
        cache.get_cache().set(key, result)
    return result


def _task(args):
    index, name, data, selected_filters, face_detection, use_cache = args
    return index, process_image(name, data, selected_filters, face_detection, use_cache)


def run_pipeline(items, selected_filters, face_detection, executor=None, use_cache=True):
    """
    Procesa un lote. `items` es una lista de (nombre, bytes).

    Yields:
        (indice, resultado) a medida que cada imagen termina.
    """
    tasks = [
        (i, name, data, list(selected_filters), face_detection, use_cache)
        for i, (name, data) in enumerate(items)
    ]
    if executor is None or len(tasks) < MIN_IMAGES_FOR_PARALLEL:
        for task in tasks:
            yield _task(task)
        return

    futures = [executor.submit(_task, task) for task in tasks]
    for future in as_completed(futures):
        yield future.result()
