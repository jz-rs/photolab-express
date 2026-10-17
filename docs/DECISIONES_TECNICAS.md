# Decisiones técnicas: PhotoLab Express

## Qué hace

Recibe un lote de fotos y, para cada una:

1. Detecta rostros sobre la foto original (Haar Cascade de OpenCV).
2. Aplica los filtros elegidos en orden (Sobel, Canny, desenfoque, nitidez, pixelado, tono aleatorio).
3. Dibuja las cajas de los rostros sobre el resultado.
4. Clasifica la foto original con CLIP (zero-shot) en: Persona, Animal, Objeto, Edificio, Paisaje o Dibujo o arte.
5. Guarda el resultado en `output/<categoría>/` y arma un `.zip` y un `.csv` descargables.

No hay entrenamiento ni dataset propio: CLIP compara cada imagen contra frases de texto
("a photo of a person", "a photo of an object", …) y elige la que mejor la describe.

## Estructura

- `app.py`: interfaz Streamlit. Toda la UI está en `main()` (necesario en Windows, ver Paralelismo).
- `src/pipeline.py`: procesamiento de una foto y del lote completo (generador).
- `src/filters.py`: filtros y registro `FILTERS`.
- `src/detect.py`: rostros y colores dominantes.
- `src/ml.py`: clasificación con CLIP.
- `src/cache.py`: caché en disco.
- `src/gen.py`: generadores de teselas.
- `src/io_utils.py`: lectura, escritura, conversiones, miniaturas, zip.
- `profiling/benchmark.py`: comparación serie / paralelo / caché + cProfile.
- `tests/`: suite de pytest (golden images, teselas, caché y pipeline en paralelo).

## Decisiones y mediciones

### Paralelismo
`ProcessPoolExecutor` (procesos, no hilos, porque el trabajo es de CPU y el GIL
no deja paralelizar con hilos). En Windows los procesos arrancan con `spawn`:
cada worker importa de nuevo los módulos. Por eso:
- `pipeline.py` no importa CLIP ni Streamlit (antes importaba `ml` y cada worker cargaba el modelo entero).
- `app.py` tiene la UI dentro de `main()` con la guarda `if __name__ == "__main__"`; si no, cada worker volvía a ejecutar la app.
- El pool se crea una sola vez (`st.cache_resource`) y se reutiliza.
- Con menos de 3 fotos se procesa en serie: crear procesos cuesta más que lo que se gana.
- Si el pool falla, la app termina el lote en serie en lugar de romperse.
- Los workers devuelven miniaturas y un PNG comprimido, no los arrays completos, para pasar menos datos entre procesos.

### Generadores
- `gen.tiles` y `gen.padded_tiles`: las imágenes grandes (> 4 MP) se filtran por teselas con margen.
  El margen evita costuras: un test verifica que el resultado es idéntico al de la imagen entera.
- `pipeline.run_pipeline` es un generador que devuelve cada foto apenas termina, y así se actualiza la barra de progreso.

### Caché
`diskcache` (SQLite, seguro entre procesos) con tope de 1 GB y política LRU.
- Resultado completo por foto + filtros, con clave = SHA-256 de los bytes del archivo. Si pega, no se decodifica ni se filtra nada.
- `@memoize` en la detección de rostros y los colores dominantes (resultados chicos).
- CLIP cachea por imagen y solo pasa por el modelo las que no vio antes.

Decisión tomada midiendo: al principio también se memoizaba cada filtro. El benchmark
mostró que con 8 procesos llenar la caché tardaba casi 5 veces más que no usarla
(20 s contra 4 s): cada filtro tarda milisegundos, pero guardar cada imagen intermedia
(varios MB) en SQLite desde 8 procesos a la vez cuesta mucho más. Cachear solo conviene
cuando guardar y leer es más barato que recalcular.

### Paralelismo: oversubscription
OpenCV usa varios hilos internamente. Con 8 procesos cada uno lanzando sus propios hilos,
se pisaban entre ellos. Cada worker se inicializa con `cv2.setNumThreads(1)`
(`initializer` del pool) y el tiempo en paralelo bajó de 7,7 s a 4,2 s.

### Profiling
El profiling mostró que la codificación PNG (`cv2.imencode`) se llevaba cerca de un tercio
del tiempo. Se bajó la compresión a 1 (los archivos quedan algo más grandes, pero se generan mucho más rápido).

### Problemas de la versión original corregidos en esta versión
- CLIP recibía las imágenes en BGR (colores invertidos).
- La detección de rostros se hacía después de Sobel/Canny, donde ya no hay caras reconocibles.
- `detect_faces` dibujaba sobre la imagen recibida (y cacheada).
- La caché usaba la imagen entera como clave y no tenía límite de tamaño.
- Los resultados desaparecían al tocar el botón de descarga (no se guardaban en `session_state`).
- Las golden images se autogeneraban si faltaban, así el test pasaba siempre.
- El test de colores dominantes usaba `in` sobre un array de numpy, que daba verdadero con cualquier cero.
- División por cero en Sobel con imágenes lisas.
- Se escribían los uploads en una carpeta temporal y los nombres repetidos se pisaban.
