"""
Utilidades de entrada/salida y conversión de imágenes.
"""
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """Carga una imagen desde disco (BGR). Funciona con rutas con tildes en Windows."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    return decode_image(data.tobytes())


_REDUCED_FLAGS = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                  4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def decode_image(data: bytes, reduce: int = 1):
    """
    Decodifica bytes (png/jpg/webp...) a una imagen BGR. Devuelve None si no es imagen.
    Con reduce=2/4/8 decodifica directamente a 1/2, 1/4 u 1/8 del tamaño: en JPEG
    es mucho más rápido que decodificar entero y después achicar.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, _REDUCED_FLAGS[reduce])


def encode_png(image, compression=1) -> bytes:
    """
    Codifica a PNG. Compresión 1 (de 0 a 9): el profiling mostró que con la
    compresión por defecto (3) el imencode se comía un tercio del tiempo total.
    """
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise ValueError("No se pudo codificar la imagen como PNG")
    return buf.tobytes()


def save_image(image, save_path):
    """Guarda una imagen (soporta rutas con tildes en Windows)."""
    save_path = Path(save_path)
    ext = save_path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"No se pudo guardar {save_path}")
    buf.tofile(str(save_path))


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cv2_to_pil(cv2_image):
    """OpenCV (BGR) -> PIL (RGB)."""
    return Image.fromarray(bgr_to_rgb(cv2_image))


def pil_to_cv2(pil_image):
    """PIL (RGB) -> OpenCV (BGR)."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def thumbnail(image, max_side=480):
    """Achica una imagen para mostrarla en la UI sin mandar megas al navegador."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


ALREADY_COMPRESSED = (".png", ".jpg", ".jpeg", ".webp")


def build_zip(files):
    """
    Arma un .zip en memoria a partir de pares (ruta_dentro_del_zip, bytes).
    Las imágenes ya vienen comprimidas: se guardan sin volver a comprimir
    (ZIP_STORED). Comprimirlas otra vez tardaba ~20 veces más y no achicaba nada.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files:
            method = zipfile.ZIP_STORED if name.lower().endswith(ALREADY_COMPRESSED) else zipfile.ZIP_DEFLATED
            zf.writestr(name, data, compress_type=method)
    return buffer.getvalue()