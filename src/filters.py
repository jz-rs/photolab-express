"""
Filtros de procesamiento de imágenes (OpenCV + NumPy).

- Los filtros NO se memoizan uno por uno: el benchmark mostró que guardar en
  disco cada imagen intermedia (varios MB cada una, desde 8 procesos a la vez)
  costaba mucho más que recalcular el filtro, que tarda milisegundos. La caché
  se aplica a nivel de foto completa en pipeline.py.
- Blur y Sharpen procesan por teselas (generador `padded_tiles`) cuando la
  imagen es grande, para no duplicar en memoria imágenes gigantes. El
  resultado es idéntico al de procesar la imagen entera (hay test que lo valida).
- "Pixelado" usa el generador `tiles` directamente.
"""
import cv2
import numpy as np

from .gen import padded_tiles, tiles

# A partir de cuántos píxeles conviene procesar por teselas
TILE_THRESHOLD = 2000 * 2000
TILE_SIZE = 512
TILE_PAD = 8  # mayor al radio de cualquier kernel que usamos

SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def apply_tiled(func, image, tile_size=TILE_SIZE, pad=TILE_PAD):
    """Aplica `func` por teselas con margen y arma la imagen de salida."""
    out = np.empty_like(image)
    for y, x, h, w, tile, oy, ox in padded_tiles(image, tile_size, pad):
        filtered = func(tile)
        out[y : y + h, x : x + w] = filtered[oy : oy + h, ox : ox + w]
    return out


def _maybe_tiled(func, image):
    if image.shape[0] * image.shape[1] > TILE_THRESHOLD:
        return apply_tiled(func, image)
    return func(image)


def apply_sobel(image):
    """Detección de bordes con Sobel (magnitud del gradiente)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.hypot(sobelx, sobely)
    max_value = magnitude.max()
    if max_value == 0:  # imagen lisa: evita división por cero
        sobel = np.zeros_like(gray)
    else:
        sobel = np.uint8(magnitude / max_value * 255)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)


def apply_canny(image):
    """Detección de bordes con Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Desenfoque gaussiano (por teselas si la imagen es grande)."""
    return _maybe_tiled(lambda img: cv2.GaussianBlur(img, kernel_size, 0), image)


def apply_sharpen(image):
    """Aumento de nitidez (por teselas si la imagen es grande)."""
    return _maybe_tiled(lambda img: cv2.filter2D(img, -1, SHARPEN_KERNEL), image)


def apply_pixelate(image, block=16):
    """Pixelado: cada bloque de la imagen se reemplaza por su color promedio."""
    out = image.copy()
    for tile in tiles(out, block, block):  # las teselas son vistas: se escribe in-place
        tile[:] = tile.reshape(-1, tile.shape[-1]).mean(axis=0).astype(image.dtype)
    return out


def apply_random_hue_shift(image, seed=None):
    """
    Cambio de tono aleatorio. No se memoiza porque no es determinista
    (salvo que se pase `seed`, que se usa en los tests).
    """
    rng = np.random.default_rng(seed)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = int(rng.integers(1, 180))
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int32) + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Registro de filtros: nombre visible -> (función, descripción corta)
FILTERS = {
    "Sobel": (apply_sobel, "Resalta bordes por intensidad de gradiente"),
    "Canny": (apply_canny, "Bordes finos en blanco y negro"),
    "Desenfoque gaussiano": (apply_gaussian_blur, "Suaviza ruido y detalle"),
    "Nitidez": (apply_sharpen, "Realza contornos y textura"),
    "Pixelado": (apply_pixelate, "Bloques de color promedio"),
    "Tono aleatorio": (apply_random_hue_shift, "Rota los colores al azar"),
}
