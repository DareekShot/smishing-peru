# Detección de smishing en SMS peruanos con machine learning

Código y datos del trabajo de investigación *Modelo de detección de smishing en el contexto peruano mediante análisis de patrones lingüísticos utilizando técnicas de machine learning* (Hiroshi Alberto Takeda Chumpitazi, Universidad de Lima, 2026).

El notebook arma un corpus de 4,972 SMS en español peruano a partir de tres datasets públicos en inglés y de SMS generados con Gemini, entrena y compara cinco clasificadores y dos ensambles, y produce todas las tablas y figuras del artículo. El modelo final es una SVM lineal con TF-IDF y 7 variables estilísticas.

## Estructura

```
notebooks/smishing_peru.ipynb   el notebook: los pasos, en el orden del artículo
src/smishing_peru/              las funciones del notebook, un módulo por sección
data/checkpoints/               las respuestas guardadas de Gemini (corrida del 25 de octubre de 2026)
data/corpus/                    el corpus final, con los enlaces reales desactivados
data/referencia/                aquí van los 3 CSV de referencia (no se suben; ver su README)
requirements.txt                versiones de la corrida en Colab
```

Módulos de `src/smishing_peru/`:

| Módulo | Sección del notebook | Qué hace |
|---|---|---|
| `datos_referencia.py` | 1 | Carga los 3 datasets de referencia |
| `catalogo.py` | 2.2 | Entidades peruanas, escenarios de fraude y legítimos, detección de enlaces |
| `gemini.py` | 2.3 | Llamadas a Gemini, idioma y casi-duplicados |
| `adaptacion.py` | 2.4 | Adaptación semántica de los SMS en inglés al contexto peruano |
| `generacion.py` | 2.5 | Generación controlada de SMS nuevos |
| `corpus.py` | 3 | Control de calidad, enlaces del original y versión para publicar |
| `preprocesamiento.py` | 4 | Desofuscación, normalización y stopwords |
| `caracteristicas.py` | 4 | Las 7 variables estilísticas, con lógica difusa |
| `modelos.py` | 5 | Pipelines, validación cruzada por grupos, Grid Search y ensambles |
| `estadistica.py` | 5.3.1 | Bootstrap, McNemar y Holm |
| `ataques.py` | 5.5 | Ataques adversarios |
| `diagnostico.py` | 6 | Diagnóstico interpretable de un SMS |
| `graficos.py` | 7 | Matriz de confusión y curva ROC |
| `carpeta_trabajo.py` | 0 | Prepara la carpeta `trabajo/` con los datos |

## Cómo correrlo en Colab

1. Abre el notebook en Colab: [notebooks/smishing_peru.ipynb en Colab](https://colab.research.google.com/github/DareekShot/smishing-peru/blob/main/notebooks/smishing_peru.ipynb)
2. Ejecuta todas las celdas. La primera descarga este repositorio y prepara la carpeta `trabajo/`. La segunda instala las versiones de la corrida del artículo; si muestra un aviso, reinicia la sesión (Entorno de ejecución > Reiniciar sesión) y vuelve a ejecutar todo.
3. Cuando la sección 1 lo pida, sube los 3 CSV de referencia (de dónde bajarlos: [data/referencia/README.md](data/referencia/README.md)).

No hace falta una API key de Gemini: el corpus se arma con las respuestas guardadas en `data/checkpoints/` y sale igual que en la corrida del artículo. La key solo se usa si pones `REGENERAR_CON_GEMINI = True` en la sección 2.1, y entonces el corpus y todos los números cambian.

Todas las salidas (evidencias, Excel de resultados, experimentos de la sección 9 y el modelo final) se guardan en `trabajo/`.

## Cómo correrlo en tu computadora

Con Python 3.12 o 3.13:

```
pip install -r requirements.txt
pip install notebook
jupyter notebook notebooks/smishing_peru.ipynb
```

Antes, pon los 3 CSV de referencia en `data/referencia/`.

## Datos

| Dataset | Uso | Licencia | Dónde está |
|---|---|---|---|
| SMS Spam Collection (Almeida et al., 2011) | Referencia para la adaptación | CC BY 4.0 | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| SMS Phishing Dataset, Dataset_5971 (Mishra & Soni, 2022) | Referencia para la adaptación | CC BY 4.0 | [Mendeley Data](https://doi.org/10.17632/f45bkkt8pr.1) |
| Smishtank, `analysisdataset.csv` (Timko & Rahman, 2024) | Referencia para la adaptación | La de sus autores | [smishtank.com](https://smishtank.com) |
| IMC25 (Agarwal et al., 2025) | Prueba externa (sección 5.6 b) | CC BY 4.0 | El notebook lo descarga de [GitHub](https://github.com/reportsmishing/Smishing-Dataset-IMC25) |

**El corpus** (`data/corpus/corpus_smishing_peru_publico.csv`) tiene los 4,972 SMS del dataset final, con su clase, su gatillo psicológico, su origen (adaptado o generado), su fuente y el grupo que usa la partición. Es el archivo que el notebook guarda como `evidencias/04b_dataset_publico.csv`, con un cambio: ahí solo se desactivan los enlaces de los SMS adaptados, y aquí también los que Gemini inventó en los SMS generados.

**Los enlaces están desactivados.** Los SMS adaptados llevan el enlace de su mensaje original, y varios son enlaces maliciosos reales. En el corpus y en los checkpoints, `http` aparece como `hxxp` y los puntos del dominio como `[.]`, para que nadie los abra por accidente. El notebook los vuelve a activar solo dentro de `trabajo/`, porque necesita el texto exacto para reproducir el corpus. El modelo no lee el dominio: todos los enlaces pasan a `urltoken` antes del TF-IDF.

## Reproducibilidad

- scikit-learn está fijado en 1.6.1: con otra versión cambia la partición train/test y cambian los resultados.
- La corrida del artículo se hizo en Colab con Python 3.13, scikit-learn 1.6.1, xgboost 3.4.1, numpy 2.1.3, pandas 2.2.3 y scipy 1.16.3.
- Al ordenar, numpy desempata distinto en los procesadores con AVX-512, y eso cambia qué 3,000 términos entran al TF-IDF. La corrida del artículo se hizo en un procesador sin AVX-512, así que la primera celda del notebook le pide a numpy que no use AVX-512 (`NPY_DISABLE_CPU_FEATURES`). Si corres el código fuera del notebook, define esa variable de entorno antes de importar numpy.
- Las latencias dependen de la máquina. El tamaño de los modelos puede variar en 0.1 KB respecto del artículo, porque el modelo guardado ahora registra que su código está en `smishing_peru`.

## Cómo citar

Takeda Chumpitazi, H. A. (2026). *Modelo de detección de smishing en el contexto peruano mediante análisis de patrones lingüísticos utilizando técnicas de machine learning* [Trabajo de investigación]. Universidad de Lima.
