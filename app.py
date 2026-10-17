"""
PhotoLab Express: interfaz en Streamlit.

Correr desde la raíz del repositorio:
    streamlit run app.py

IMPORTANTE (Windows): toda la interfaz está dentro de main() y se llama solo
si __name__ == "__main__". Los procesos del pool se crean con "spawn" y
reimportan este archivo como "__mp_main__"; sin esa guarda, cada proceso
volvería a ejecutar la app entera e intentaría crear sus propios procesos.
"""
import time
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd
import streamlit as st

from src import cache, filters, io_utils, pipeline

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
NO_CATEGORY = "Sin clasificar"
PREVIEW_LIMIT = 32
CATEGORY_COLORS = {
    "Persona": "orange",
    "Animal": "green",
    "Objeto": "blue",
    "Edificio": "violet",
    "Paisaje": "green",
    "Dibujo o arte": "red",
    NO_CATEGORY: "gray",
}
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&display=swap');
h1, h2, h3 { font-family: 'Archivo', 'Segoe UI', sans-serif !important; letter-spacing: -0.01em; }
h1 { font-weight: 800 !important; }
.frame-number { font-family: 'Archivo', 'Segoe UI', sans-serif; font-weight: 700;
                color: #24527A; font-size: 0.85rem; }
.lead { font-size: 1.05rem; color: #3E4A44; max-width: 62ch; margin-top: -0.6rem; }
</style>
"""


# --------------------------------------------------------------------------
# Recursos compartidos entre reruns
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_executor():
    """Pool de procesos creado una sola vez: en Windows crear procesos es caro."""
    return pipeline.create_executor()


@st.cache_resource(show_spinner=False)
def get_ml():
    """Importa y carga CLIP una sola vez por servidor."""
    from src import ml

    ml.load_model()
    return ml


@st.cache_data(show_spinner=False, max_entries=256)
def preview_thumb(data: bytes):
    """Miniatura para la vista previa (cacheada: no se recalcula en cada rerun)."""
    image = io_utils.decode_image(data)
    return None if image is None else io_utils.bgr_to_rgb(io_utils.thumbnail(image, 240))


# --------------------------------------------------------------------------
# Secciones de la interfaz
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("Ajustes")
        settings = {
            "filters": st.multiselect(
                "Filtros",
                options=list(filters.FILTERS.keys()),
                placeholder="Sin filtros",
                help="Se aplican en el orden en que los elegís.\n\n"
                + "\n".join(f"- **{n}**: {d}" for n, (_, d) in filters.FILTERS.items()),
            ),
            "faces": st.toggle("Detectar rostros", value=True),
            "classify": st.toggle(
                "Clasificar con IA",
                value=True,
                help="Usa CLIP para decir si la foto es una persona, animal, objeto, "
                "edificio, paisaje o dibujo. La primera vez descarga el modelo (unos 600 MB).",
            ),
        }
        st.divider()
        st.subheader("Rendimiento")
        settings["parallel"] = st.toggle(
            "Procesar en paralelo",
            value=True,
            help=f"Reparte las fotos entre {pipeline.default_workers()} procesos. "
            f"Con menos de {pipeline.MIN_IMAGES_FOR_PARALLEL} fotos se procesa en serie.",
        )
        settings["cache"] = st.toggle(
            "Usar caché",
            value=True,
            help="Guarda en disco los resultados. Si volvés a procesar las mismas fotos "
            "con los mismos filtros, sale casi instantáneo.",
        )
        if st.button("Vaciar caché", width="stretch"):
            cache.clear()
            st.toast("Caché vaciada")
    return settings


def render_upload(settings):
    """Muestra el uploader y la vista previa. Devuelve (archivos, se_apretó_procesar)."""
    uploaded = st.file_uploader(
        "Fotos", type=ACCEPTED_TYPES, accept_multiple_files=True, label_visibility="collapsed"
    )
    if not uploaded:
        if "results" not in st.session_state:
            st.info("Arrastrá una o más fotos acá arriba para empezar. "
                    "Los filtros se eligen en el panel de la izquierda.")
        return [], False

    with st.expander(f"Vista previa ({len(uploaded)} fotos)", expanded=len(uploaded) <= 8):
        cols = st.columns(8)
        for i, f in enumerate(uploaded[:PREVIEW_LIMIT]):
            thumb = preview_thumb(f.getvalue())
            if thumb is None:
                cols[i % 8].caption(f"{f.name}: no se puede leer")
            else:
                cols[i % 8].image(thumb, caption=f.name, width="stretch")
        if len(uploaded) > PREVIEW_LIMIT:
            st.caption(f"Y {len(uploaded) - PREVIEW_LIMIT} más.")

    chosen = ", ".join(settings["filters"]) if settings["filters"] else None
    st.caption(f"Se van a procesar {len(uploaded)} fotos " + (f"con {chosen}." if chosen else "sin filtros."))
    run = st.button(f"Procesar {len(uploaded)} fotos", type="primary", icon=":material/play_arrow:")
    return uploaded, run


def unique_name(name, used):
    stem = Path(name).stem
    candidate, n = stem, 2
    while candidate in used:
        candidate, n = f"{stem}_{n}", n + 1
    used.add(candidate)
    return candidate + ".png"


def build_table(results):
    rows = []
    for i, r in enumerate(results, start=1):
        if not r["ok"]:
            continue
        rows.append(
            {
                "N°": i,
                "Archivo": r["name"],
                "Categoría": r["category"],
                "Confianza": r["confidence"],
                "Rostros": len(r["faces"]),
                "Ancho": r["size"][0],
                "Alto": r["size"][1],
                "Tiempo (ms)": round(r["seconds"] * 1000, 1),
                "Desde caché": r["cache_hits"] > 0 and r["cache_misses"] == 0,
            }
        )
    return pd.DataFrame(rows)


def process_batch(files, settings):
    items = [(f.name, f.getvalue()) for f in files]
    total = len(items)
    results = [None] * total
    progress = st.progress(0.0, text="Preparando…")
    start = time.perf_counter()

    use_pool = settings["parallel"] and total >= pipeline.MIN_IMAGES_FOR_PARALLEL
    executor = get_executor() if use_pool else None
    mode = f"en paralelo con {pipeline.default_workers()} procesos" if executor else "en serie"
    args = (settings["filters"], settings["faces"])

    try:
        gen = pipeline.run_pipeline(items, *args, executor, settings["cache"])
        for done, (index, result) in enumerate(gen, start=1):
            results[index] = result
            progress.progress(done / total, text=f"Procesando {done} de {total}: {result['name']}")
    except BrokenProcessPool:
        # Si el pool se rompe, se descarta y se termina en serie lo que faltaba
        get_executor.clear()
        mode = "en serie (el pool de procesos falló)"
        pending = [i for i, r in enumerate(results) if r is None]
        gen = pipeline.run_pipeline([items[i] for i in pending], *args, None, settings["cache"])
        for j, (_, result) in enumerate(gen):
            results[pending[j]] = result
            progress.progress((total - len(pending) + j + 1) / total, text=f"Procesando {result['name']}")

    filters_seconds = time.perf_counter() - start

    for r in results:
        r.setdefault("category", NO_CATEGORY)
        r.setdefault("confidence", None)

    classify_seconds = 0.0
    if settings["classify"]:
        progress.progress(1.0, text="Clasificando con IA…")
        t0 = time.perf_counter()
        try:
            with st.spinner("Cargando el modelo de IA (la primera vez tarda un poco)…"):
                ml = get_ml()
            images = [
                    io_utils.thumbnail(io_utils.decode_image(data, reduce=2), 448) if r["ok"] else None
                for (_, data), r in zip(items, results)
            ]
            for r, pred in zip(results, ml.classify_batch(images, use_cache=settings["cache"])):
                if pred:
                    r["category"], r["confidence"] = pred["label"], pred["confidence"]
        except Exception as exc:  # sin torch/transformers instalados o sin internet
            st.warning(f"No se pudo clasificar con IA ({exc}). Las fotos quedan como “{NO_CATEGORY}”.")
        classify_seconds = time.perf_counter() - t0

    # Guardar en disco: output/<categoría>/<archivo>.png  +  armar el zip
    used, zip_files = set(), []
    for r in results:
        if not r["ok"]:
            continue
        r["out_name"] = unique_name(r["name"], used)
        folder = OUTPUT_DIR / r["category"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / r["out_name"]).write_bytes(r["processed_png"])
        zip_files.append((f"{r['category']}/{r['out_name']}", r["processed_png"]))
    progress.empty()

    table = build_table(results)
    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig: Excel muestra bien las tildes
    zip_files.append(("metricas.csv", csv_bytes))

    st.session_state.update(
        results=results,
        table=table,
        csv=csv_bytes,
        zip=io_utils.build_zip(zip_files),
        meta={
            "mode": mode,
            "filters": list(settings["filters"]),
            "filters_seconds": filters_seconds,
            "classify_seconds": classify_seconds,
            "total_seconds": time.perf_counter() - start,
        },
    )


def render_results():
    results = st.session_state.get("results")
    if not results:
        return
    meta = st.session_state["meta"]
    ok_results = [r for r in results if r["ok"]]

    st.divider()
    st.header("Resultados")
    used_filters = ", ".join(meta["filters"]) if meta["filters"] else "sin filtros"
    st.caption(f"Última corrida: {used_filters}, procesada {meta['mode']}.")

    hits = sum(r["cache_hits"] for r in ok_results)
    lookups = hits + sum(r["cache_misses"] for r in ok_results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fotos procesadas", len(ok_results))
    c2.metric("Con rostros", sum(1 for r in ok_results if r["faces"]))
    c3.metric(
        "Tiempo total",
        f"{meta['total_seconds']:.2f} s",
        help=f"Filtros y detección: {meta['filters_seconds']:.2f} s. IA: {meta['classify_seconds']:.2f} s.",
    )
    c4.metric("Aciertos de caché", f"{(hits / lookups * 100) if lookups else 0:.0f} %")

    for r in results:
        if not r["ok"]:
            st.warning(f"{r['name']}: {r['error']}")

    tab_sheet, tab_table, tab_download = st.tabs(["Hoja de contactos", "Tabla", "Descargas"])
    with tab_sheet:
        render_contact_sheet(results)
    with tab_table:
        render_table(ok_results)
    with tab_download:
        render_downloads()


def render_contact_sheet(results):
    categories = sorted({r["category"] for r in results if r["ok"]})
    f1, f2, f3 = st.columns([3, 1.4, 1.4], vertical_alignment="bottom")
    chosen = f1.pills("Categorías", categories, selection_mode="multi", default=categories) or []
    only_faces = f2.toggle("Solo con rostros")
    view = f3.segmented_control("Mostrar", ["Procesada", "Original", "Ambas"], default="Procesada") or "Procesada"

    shown = [
        (i, r)
        for i, r in enumerate(results, start=1)
        if r["ok"] and r["category"] in chosen and (r["faces"] or not only_faces)
    ]
    if not shown:
        st.info("Ninguna foto coincide con esta vista. Activá más categorías o apagá “Solo con rostros”.")
        return

    per_row = 2 if view == "Ambas" else 4
    for start in range(0, len(shown), per_row):
        cols = st.columns(per_row)
        for col, (i, r) in zip(cols, shown[start : start + per_row]):
            with col.container(border=True):
                st.markdown(f'<span class="frame-number">{i:02d}</span>&nbsp; {r["name"]}', unsafe_allow_html=True)
                if view == "Ambas":
                    a, b = st.columns(2)
                    a.image(io_utils.bgr_to_rgb(r["thumb_original"]), caption="Original", width="stretch")
                    b.image(io_utils.bgr_to_rgb(r["thumb_processed"]), caption="Procesada", width="stretch")
                else:
                    key = "thumb_original" if view == "Original" else "thumb_processed"
                    st.image(io_utils.bgr_to_rgb(r[key]), width="stretch")

                color = CATEGORY_COLORS.get(r["category"], "gray")
                label = r["category"] + (f" {r['confidence'] * 100:.0f}%" if r["confidence"] is not None else "")
                tags = [f":{color}-background[{label}]"]
                if r["faces"]:
                    n = len(r["faces"])
                    tags.append(f":orange-background[{n} {'rostro' if n == 1 else 'rostros'}]")
                st.markdown(" ".join(tags))


def render_table(ok_results):
    st.dataframe(
        st.session_state["table"],
        hide_index=True,
        column_config={
            "Confianza": st.column_config.ProgressColumn("Confianza", min_value=0.0, max_value=1.0, format="percent"),
            "Desde caché": st.column_config.CheckboxColumn("Desde caché"),
        },
    )
    counts = Counter(r["category"] for r in ok_results)
    if len(counts) > 1:
        st.subheader("Fotos por categoría")
        st.bar_chart(pd.Series(counts, name="Fotos"), horizontal=True, color="#24527A")


def render_downloads():
    d1, d2 = st.columns(2)
    d1.download_button(
        "Descargar fotos procesadas (.zip)",
        data=st.session_state["zip"],
        file_name="photolab_resultados.zip",
        mime="application/zip",
        type="primary",
        icon=":material/folder_zip:",
        width="stretch",
        on_click="ignore",
    )
    d2.download_button(
        "Descargar métricas (.csv)",
        data=st.session_state["csv"],
        file_name="photolab_metricas.csv",
        mime="text/csv",
        icon=":material/table:",
        width="stretch",
        on_click="ignore",
    )
    st.caption(f"El zip viene ordenado en carpetas por categoría. También quedó una copia en {OUTPUT_DIR}.")


def main():
    st.set_page_config(page_title="PhotoLab Express", page_icon="🎞️", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    settings = render_sidebar()
    st.title("PhotoLab Express")
    st.markdown(
        '<p class="lead">Aplicá filtros a un lote de fotos, detectá rostros y dejá que la IA '
        "las ordene por categoría.</p>",
        unsafe_allow_html=True,
    )
    files, run = render_upload(settings)
    if run:
        process_batch(files, settings)
    render_results()


if __name__ == "__main__":
    main()
