# Respuestas guardadas de Gemini

Gemini responde distinto en cada llamada. Para que el corpus salga igual que en el artículo, el notebook usa las respuestas de la corrida del 25 de octubre de 2026 y no vuelve a llamar a Gemini.

| Archivo | Sección | Filas | Qué tiene |
|---|---|---|---|
| `dataset_traducido_adaptado_enlaces_desactivados.csv` | 2.4 | 2,192 | Cada SMS en inglés (`text_original`) y su adaptación peruana (`text`), con el gatillo, las entidades y si se descartó |
| `corpus_gemini_peru_v2_enlaces_desactivados.csv` | 2.5 | 4,190 | Los SMS generados, con el plan de cada llamada (gatillo, escenario, canal, estilo y entidades sugeridas) |

En los dos archivos, los enlaces están desactivados (`hxxp`, `[.]`). La primera celda del notebook los copia a `trabajo/` con los enlaces activados y con los nombres que leen las secciones 2.4 y 2.5 (`dataset_traducido_adaptado.csv` y `corpus_gemini_peru_v2.csv`). El texto que resulta es idéntico, byte a byte, al de la corrida original.

Los textos de `text_original` vienen de los 3 datasets de referencia (ver `data/referencia/README.md`).
