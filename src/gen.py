"""
Generadores para procesar imágenes por bloques (teselas).

`tiles` produce teselas simples. `padded_tiles` produce teselas con un margen
(padding) alrededor, que es lo que se necesita para aplicar filtros de
vecindad (blur, sharpen) por bloques SIN que aparezcan costuras: cada tesela
se filtra con su margen y después se recorta solo la parte interior.

Al ser generadores, nunca hay más de una tesela "extra" en memoria a la vez.
"""


def tiles(image, tile_width, tile_height):
    """Genera teselas consecutivas de la imagen (sin solapamiento)."""
    for y in range(0, image.shape[0], tile_height):
        for x in range(0, image.shape[1], tile_width):
            yield image[y : y + tile_height, x : x + tile_width]


def padded_tiles(image, tile_size, pad):
    """
    Genera (y, x, h, w, tesela_con_margen, offset_y, offset_x).

    - (y, x, h, w): región de la imagen final que cubre la tesela.
    - tesela_con_margen: recorte de la imagen con `pad` píxeles extra por lado
      (menos en los bordes de la imagen, donde no hay más píxeles).
    - (offset_y, offset_x): dónde empieza la región útil dentro de la tesela.
    """
    height, width = image.shape[:2]
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            h = min(tile_size, height - y)
            w = min(tile_size, width - x)
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(height, y + h + pad), min(width, x + w + pad)
            yield y, x, h, w, image[y0:y1, x0:x1], y - y0, x - x0
