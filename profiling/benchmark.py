"""
Benchmark y profiling del pipeline, sin interfaz.

Escenarios sobre el mismo lote:
  1. En serie, OpenCV con 1 hilo   -> un solo núcleo de verdad (línea base)
  2. En serie, OpenCV multihilo    -> lo que hace OpenCV "de fábrica"
  3. En paralelo, sin caché
  4. En paralelo, llenando caché   -> primera corrida con caché vacía
  5. En paralelo, caché caliente   -> segunda corrida

Uso (desde la raíz del repositorio):
    python profiling/benchmark.py                     # 24 imágenes sintéticas de 1920x1080
    python profiling/benchmark.py --folder C:\\fotos   # tus propias fotos
    python profiling/benchmark.py --profile           # además genera profile.out (cProfile)
    snakeviz profile.out                              # ver el profiling en el navegador
"""
import argparse
import cProfile
import os
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Caché aislada y vacía para el benchmark. Tiene que definirse ANTES de importar
# src.cache, y por variable de entorno para que la hereden los procesos del pool.
# Ojo: en Windows cada worker reimporta este archivo; como heredan la variable,
# reutilizan la carpeta del proceso principal en vez de crear una nueva.
BENCH_CACHE = os.environ.get("PHOTOLAB_CACHE_DIR") or tempfile.mkdtemp(prefix="photolab_bench_")
os.environ["PHOTOLAB_CACHE_DIR"] = BENCH_CACHE

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils, pipeline  # noqa: E402

FILTERS = ["Desenfoque gaussiano", "Nitidez", "Sobel"]


def synthetic_items(n, w, h):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
        noise = rng.normal(0, 25, (h, w, 3))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        items.append((f"sintetica_{i:02d}.png", io_utils.encode_png(img)))
    return items


def folder_items(folder):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [(p.name, p.read_bytes()) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]


def timed(label, items, executor, use_cache):
    start = time.perf_counter()
    results = list(pipeline.run_pipeline(items, FILTERS, True, executor, use_cache))
    elapsed = time.perf_counter() - start
    print(f"  {label:<36} {elapsed:7.2f} s   ({len(results)} imágenes)")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("-n", type=int, default=24)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    items = folder_items(args.folder) if args.folder else synthetic_items(args.n, 1920, 1080)
    if not items:
        sys.exit("No encontré imágenes en esa carpeta.")
    workers = pipeline.default_workers()
    print(f"Lote: {len(items)} imágenes. Filtros: {', '.join(FILTERS)} + detección de rostros")
    print(f"CPU: {os.cpu_count()} núcleos lógicos. OpenCV {cv2.__version__}.\n")

    try:
        default_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        single = timed("En serie, OpenCV 1 hilo", items, None, False)
        cv2.setNumThreads(default_threads)
        serial = timed(f"En serie, OpenCV {default_threads} hilos", items, None, False)

        with pipeline.create_executor(workers) as executor:
            # Calentar: que los procesos ya estén creados y con todo importado
            list(pipeline.run_pipeline(items[:workers], FILTERS, True, executor, False))
            parallel = timed(f"Paralelo ({workers} procesos), sin caché", items, executor, False)
            filling = timed("Paralelo, llenando caché", items, executor, True)
            cached = timed("Paralelo, caché caliente", items, executor, True)

        print(f"\nSpeedup paralelo vs 1 núcleo:        {single / parallel:.1f}x")
        print(f"Speedup paralelo vs OpenCV multihilo: {serial / parallel:.1f}x")
        print(f"Costo de llenar la caché:             {(filling / parallel - 1) * 100:+.0f} %")
        print(f"Speedup por caché caliente:           {parallel / cached:.1f}x")

        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            list(pipeline.run_pipeline(items, FILTERS, True, None, False))
            profiler.disable()
            out = Path(__file__).resolve().parent.parent / "profile.out"
            profiler.dump_stats(out)
            print(f"\nProfiling (corrida en serie) guardado en {out}. Top 10 por tiempo acumulado:\n")
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(10)
    finally:
        shutil.rmtree(BENCH_CACHE, ignore_errors=True)


if __name__ == "__main__":  # obligatorio en Windows para usar procesos
    main()
