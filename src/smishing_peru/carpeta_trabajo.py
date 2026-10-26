"""Prepara la carpeta trabajo/, donde el notebook lee los datos y guarda sus salidas.

Copia ahí los 3 CSV de referencia (desde data/referencia/ o desde otra carpeta) y los checkpoints
de Gemini de data/checkpoints/. Los checkpoints se publican con los enlaces desactivados ("hxxp",
"[.]"), así que al copiarlos se vuelven a activar: el notebook necesita el texto exacto para que el
corpus salga igual que en la corrida del 25 de octubre.
"""
import csv
import re
import shutil
import sys
from pathlib import Path

ARCHIVOS_REFERENCIA = ["SMSSpamCollection.csv", "Dataset_5971.csv", "analysisdataset.csv"]
# nombre en trabajo/ -> nombre en data/checkpoints/
CHECKPOINTS = {
    "dataset_traducido_adaptado.csv": "dataset_traducido_adaptado_enlaces_desactivados.csv",
    "corpus_gemini_peru_v2.csv": "corpus_gemini_peru_v2_enlaces_desactivados.csv",
}


def _hxxp(texto):
    """http -> hxxp en todo el texto, conservando las mayúsculas (Http -> Hxxp, HTTP -> HXXP)."""
    return re.sub(r"([hH])([tT])([tT])([pP])",
                  lambda m: m[1] + ("X" if m[2] == "T" else "x") + ("X" if m[3] == "T" else "x") + m[4], texto)


def desactivar_enlace_reversible(enlace):
    """Como desactivar_enlace (3.4), pero conserva las mayúsculas del esquema."""
    if enlace[:4].lower() == "http":
        enlace = _hxxp(enlace[:4]) + enlace[4:]
    esquema, resto = re.match(r"((?:hxxps?:/*|//)?)(.*)", enlace, flags=re.IGNORECASE | re.DOTALL).groups()
    dominio, barra, ruta = resto.partition("/")
    return esquema + dominio.replace(".", "[.]") + barra + ruta


def desactivar_enlaces_reversible(texto):
    """Desactiva los enlaces del texto y cualquier "http" que quede (por ejemplo, dentro de otro enlace)."""
    from .corpus import buscar_enlaces  # aquí y no arriba: preparar la carpeta no debe cargar numpy

    for m in reversed(buscar_enlaces(texto)):
        texto = texto[:m.start()] + desactivar_enlace_reversible(m.group(0)) + texto[m.end():]
    return _hxxp(texto)


def activar_enlaces(texto):
    """Deshace desactivar_enlaces_reversible. Sirve porque los textos originales no traen "hxxp" ni "[.]"."""
    texto = re.sub(r"([hH])([xX])([xX])([pP])",
                   lambda m: m[1] + ("T" if m[2] == "X" else "t") + ("T" if m[3] == "X" else "t") + m[4], texto)
    return texto.replace("[.]", ".")


def _transformar_csv(origen, destino, funcion):
    csv.field_size_limit(sys.maxsize)
    with open(origen, encoding="utf-8", newline="") as f:
        filas = [[funcion(x) for x in fila] for fila in csv.reader(f)]
    with open(destino, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(filas)


def preparar_carpeta_trabajo(raiz, buscar_tambien_en=()):
    """Crea raiz/trabajo con los datos que necesita el notebook y devuelve su ruta."""
    raiz = Path(raiz)
    trabajo = raiz / "trabajo"
    trabajo.mkdir(exist_ok=True)
    for nombre in ARCHIVOS_REFERENCIA:
        if (trabajo / nombre).exists():
            continue
        for carpeta in [raiz / "data" / "referencia", *map(Path, buscar_tambien_en)]:
            if (carpeta / nombre).exists():
                shutil.copy2(carpeta / nombre, trabajo / nombre)
                break
        else:
            print(f"Falta {nombre}: la sección 1 te pedirá subirlo (de dónde bajarlo: data/referencia/README.md).")
    for nombre, publicado in CHECKPOINTS.items():
        # si ya está (por ejemplo, porque se regeneró con Gemini), no se toca
        if (trabajo / nombre).exists():
            continue
        _transformar_csv(raiz / "data" / "checkpoints" / publicado, trabajo / nombre, activar_enlaces)
    return trabajo
