# Datasets de referencia

El notebook necesita estos 3 archivos, con estos nombres exactos. No se suben al repositorio: cada uno se descarga de su fuente original.

| Archivo | Dataset | Dónde bajarlo | Qué archivo es |
|---|---|---|---|
| `SMSSpamCollection.csv` | SMS Spam Collection (Almeida et al., 2011) | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | El archivo `SMSSpamCollection` del zip, renombrado a `.csv` (etiqueta, tabulador y texto, sin encabezado) |
| `Dataset_5971.csv` | SMS Phishing Dataset (Mishra & Soni, 2022) | [Mendeley Data](https://doi.org/10.17632/f45bkkt8pr.1) | `Dataset_5971.csv`, dentro del zip |
| `analysisdataset.csv` | Smishtank (Timko & Rahman, 2024) | [smishtank.com](https://smishtank.com) | El dataset de análisis de Smishtank, con las columnas `MainText`, `Phishing` y `Malicious` |

Puedes ponerlos en esta carpeta (`data/referencia/`) o subirlos a Colab cuando el notebook los pida.

## Referencias

- Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. En *Proceedings of the 11th ACM Symposium on Document Engineering* (pp. 259–262). ACM. https://doi.org/10.1145/2034691.2034742
- Mishra, S., & Soni, D. (2022). *SMS phishing dataset for machine learning and pattern recognition* (Versión 1) [Conjunto de datos]. Mendeley Data. https://doi.org/10.17632/f45bkkt8pr.1
- Timko, D., & Rahman, M. L. (2024). Smishing dataset I: Phishing SMS dataset from Smishtank.com. En *Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy (CODASPY '24)* (pp. 289–294). ACM. https://doi.org/10.1145/3626232.3653282
