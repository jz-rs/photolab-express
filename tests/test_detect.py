from pathlib import Path

import numpy as np

from src import detect, io_utils

DATA = Path(__file__).resolve().parent.parent / "data"


def test_no_faces_in_simple_image():
    image = io_utils.load_image(DATA / "test_image.png")
    assert detect.detect_faces(image) == []


def test_detect_does_not_modify_image():
    image = io_utils.load_image(DATA / "test_image.png")
    copy = image.copy()
    detect.detect_faces(image)
    assert np.array_equal(image, copy)


def test_draw_faces_returns_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = detect.draw_faces(image, [(10, 10, 30, 30)])
    assert image.max() == 0 and out.max() > 0


def test_dominant_colors_black_and_white():
    image = io_utils.load_image(DATA / "test_image.png")
    colors = detect.get_dominant_colors(image, k=2)
    found = {tuple(c) for c in colors.tolist()}
    # comparación por fila completa (antes `[0,0,0] in array` daba True con cualquier 0)
    assert found == {(0, 0, 0), (255, 255, 255)}
