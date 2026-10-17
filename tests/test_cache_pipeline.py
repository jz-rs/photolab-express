from pathlib import Path

import numpy as np

from src import cache, io_utils, pipeline

DATA = Path(__file__).resolve().parent.parent / "data"


def test_hash_depends_on_content_and_shape():
    a = np.zeros((10, 20, 3), np.uint8)
    assert cache.array_hash(a) == cache.array_hash(a.copy())
    assert cache.array_hash(a) != cache.array_hash(a.reshape(20, 10, 3))
    b = a.copy(); b[0, 0, 0] = 1
    assert cache.array_hash(a) != cache.array_hash(b)


def test_memoize_hits(tmp_path, monkeypatch):
    from diskcache import Cache

    monkeypatch.setattr(cache, "_cache", Cache(str(tmp_path)))
    cache.set_enabled(True)
    calls = []

    @cache.memoize
    def slow_sum(image):
        calls.append(1)
        return int(image.sum())

    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    h0 = cache.stats["hits"]
    assert slow_sum(img) == slow_sum(img.copy())
    assert len(calls) == 1  # la segunda vez salió de caché
    assert cache.stats["hits"] == h0 + 1


def test_process_image_uses_full_result_cache(tmp_path, monkeypatch):
    from diskcache import Cache

    monkeypatch.setattr(cache, "_cache", Cache(str(tmp_path)))
    data = (DATA / "test_image.png").read_bytes()
    first = pipeline.process_image("a.png", data, ["Canny"], True, use_cache=True)
    second = pipeline.process_image("b.png", data, ["Canny"], True, use_cache=True)
    assert first["cache_hits"] == 0 and second["cache_hits"] == 1
    assert second["name"] == "b.png"
    assert first["processed_png"] == second["processed_png"]


def test_process_image_and_invalid_file():
    data = (DATA / "test_image.png").read_bytes()
    ok = pipeline.process_image("x.png", data, ["Sobel", "Nitidez"], True, use_cache=False)
    assert ok["ok"] and ok["faces"] == [] and ok["processed_png"][:4] == b"\x89PNG"
    bad = pipeline.process_image("x.txt", b"no soy una imagen", [], True, use_cache=False)
    assert not bad["ok"] and bad["error"]


def test_run_pipeline_parallel_keeps_order():
    data = (DATA / "test_image.png").read_bytes()
    items = [(f"img{i}.png", data) for i in range(5)]
    with pipeline.create_executor(2) as ex:
        out = dict(pipeline.run_pipeline(items, ["Canny"], False, ex, use_cache=False))
    assert sorted(out) == list(range(5))
    assert all(out[i]["name"] == f"img{i}.png" for i in range(5))
