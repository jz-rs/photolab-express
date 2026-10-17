"""
Caché en disco para resultados costosos (filtros, detección, clasificación).

Mejoras respecto a la versión original:
- La clave NO es la imagen entera pickleada: se usa el SHA-256 de los bytes
  (+ shape y dtype). Calcular el hash es mucho más barato que serializar
  megas de píxeles en cada consulta.
- La caché tiene tope de tamaño (política LRU de diskcache), así no crece
  infinito en disco.
- Se lleva un contador de hits/misses por proceso para mostrar métricas en la UI.
- diskcache usa SQLite, así que es seguro usarlo desde varios procesos a la vez
  (importante porque el pipeline corre con ProcessPoolExecutor).
"""
import functools
import hashlib
import os
from pathlib import Path

import numpy as np
from diskcache import Cache

# Se puede cambiar con la variable de entorno PHOTOLAB_CACHE_DIR (la heredan
# también los procesos del pool, cosa que no pasa con una variable de Python).
CACHE_DIR = Path(os.environ.get("PHOTOLAB_CACHE_DIR") or Path(__file__).resolve().parent.parent / ".cache")
SIZE_LIMIT = 1024 * 1024 * 1024  # 1 GB

_cache = None
_enabled = True
stats = {"hits": 0, "misses": 0}


def get_cache():
    """Abre la caché de forma perezosa (una sola vez por proceso)."""
    global _cache
    if _cache is None:
        _cache = Cache(str(CACHE_DIR), size_limit=SIZE_LIMIT, eviction_policy="least-recently-used")
    return _cache


def set_enabled(value: bool):
    """Permite apagar la caché (útil para comparar tiempos con y sin caché)."""
    global _enabled
    _enabled = bool(value)


def clear():
    """Borra todo el contenido de la caché."""
    get_cache().clear()


def array_hash(arr: np.ndarray) -> str:
    """Hash SHA-256 de un array de numpy, incluyendo forma y tipo."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _to_key_part(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", array_hash(value))
    if isinstance(value, (list, tuple)):
        return tuple(_to_key_part(v) for v in value)
    return value


def make_key(name, args, kwargs):
    return (
        name,
        tuple(_to_key_part(a) for a in args),
        tuple(sorted((k, _to_key_part(v)) for k, v in kwargs.items())),
    )


def memoize(func):
    """Decorador de memoización en disco con clave por hash."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _enabled:
            return func(*args, **kwargs)
        key = make_key(f"{func.__module__}.{func.__qualname__}", args, kwargs)
        cache = get_cache()
        result = cache.get(key, default=None)
        if result is not None:
            stats["hits"] += 1
            return result
        stats["misses"] += 1
        result = func(*args, **kwargs)
        cache.set(key, result)
        return result

    return wrapper
