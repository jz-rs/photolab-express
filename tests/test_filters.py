"""
Tests de filtros con "golden images".

Las golden images están versionadas en data/golden_images. Si falta alguna,
el test FALLA (antes se generaba sola y el test pasaba siempre).
Para regenerarlas a propósito: set UPDATE_GOLDEN=1 && pytest   (Windows)
                               UPDATE_GOLDEN=1 pytest          (Linux/Mac)
"""
import os
from pathlib import Path

import numpy as np
import pytest

from src import filters, io_utils

DATA = Path(__file__).resolve().parent.parent / "data"
GOLDEN = DATA / "golden_images"


@pytest.fixture
def image():
    return io_utils.load_image(DATA / "test_image.png")


@pytest.fixture
def color_image():
    return io_utils.load_image(DATA / "color_test_image.png")


def random_image(h=300, w=420, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "name, func",
    [
        ("sobel", filters.apply_sobel),
        ("canny", filters.apply_canny),
        ("gaussian_blur", filters.apply_gaussian_blur),
        ("sharpen", filters.apply_sharpen),
    ],
)
def test_golden(name, func, image):
    result = func(image)
    golden_path = GOLDEN / f"{name}.png"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        io_utils.save_image(result, golden_path)
    assert golden_path.exists(), f"Falta la golden image {golden_path.name}"
    golden = io_utils.load_image(golden_path)
    assert result.shape == golden.shape
    assert np.array_equal(result, golden)


def test_filters_keep_shape_and_type():
    img = random_image()
    for name, (func, _) in filters.FILTERS.items():
        out = func(img)
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


def test_filters_do_not_modify_input():
    img = random_image()
    copy = img.copy()
    for func, _ in filters.FILTERS.values():
        func(img)
    assert np.array_equal(img, copy)


def test_sobel_on_flat_image_does_not_crash():
    flat = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert filters.apply_sobel(flat).max() == 0


@pytest.mark.parametrize("func", [
    lambda img: __import__("cv2").GaussianBlur(img, (5, 5), 0),
    lambda img: __import__("cv2").filter2D(img, -1, filters.SHARPEN_KERNEL),
])
def test_tiled_equals_full(func):
    """Procesar por teselas tiene que dar EXACTAMENTE lo mismo que la imagen entera."""
    img = random_image(517, 389)  # tamaños que no son múltiplo de la tesela
    full = func(img)
    tiled = filters.apply_tiled(func, img, tile_size=64, pad=8)
    assert np.array_equal(full, tiled)


def test_pixelate_blocks_are_uniform():
    out = filters.apply_pixelate(random_image(64, 64), block=16)
    block = out[:16, :16].reshape(-1, 3)
    assert (block == block[0]).all()


def test_random_hue_shift(color_image):
    a = filters.apply_random_hue_shift(color_image, seed=42)
    b = filters.apply_random_hue_shift(color_image, seed=42)
    assert a.shape == color_image.shape
    assert not np.array_equal(a, color_image)
    assert np.array_equal(a, b)  # con semilla es reproducible
