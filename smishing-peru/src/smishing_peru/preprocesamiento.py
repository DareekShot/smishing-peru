"""Preprocesamiento del texto: desofuscación, normalización y stopwords (sección 4).

Código tomado del notebook v9, sin cambios.
"""
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin

from .catalogo import REGEX_ENLACE_AMPLIO, normalizar


# Preprocesamiento NLP y extractor de características estilísticas/estructurales
# v5: se agregan desofuscar() (defensa adversaria, opcional) y PreprocesadorTexto (para el pipeline).

# --- Stopwords adaptadas al español peruano (ficha, Fase 2) ---
# Base: lista estándar de stopwords del español (la de NLTK, sin tildes), que incluye
# artículos, preposiciones, conjunciones, pronombres y formas de ser/estar/haber/tener.
STOPWORDS_BASE_ES = set(normalizar(" ".join("""
de la que el en y a los del se las por un para con no una su al lo como mas pero sus le ya o este si porque
esta entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos uno les ni contra
otros ese eso ante ellos e esto mi antes algunos que unos yo otro otras otra el tanto esa estos mucho quienes
nada muchos cual poco ella estar estas algunas algo nosotros mis tu te ti tus ellas nosotras vosotros vosotras
os mio mia mios mias tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro
vuestra vuestros vuestras esos esas estoy estas estamos estais estan este estes estemos esten estare estaras
estara estaremos estareis estaran estaria estarias estariamos estariais estarian estaba estabas estabamos
estabais estaban estuve estuviste estuvo estuvimos estuvisteis estuvieron estuviera estuvieras estuvieramos
estuvierais estuvieran estando estado estada estados estadas estad he has ha hemos habeis han haya hayas
hayamos hayais hayan habre habras habra habremos habreis habran habria habrias habriamos habriais habrian
habia habias habiamos habiais habian hube hubiste hubo hubimos hubisteis hubieron hubiera hubieras
hubieramos hubierais hubieran habiendo habido habida habidos habidas soy eres es somos sois son sea seas
seamos seais sean sere seras sera seremos sereis seran seria serias seriamos seriais serian era eras eramos
erais eran fui fuiste fue fuimos fuisteis fueron fuera fueras fueramos fuerais fueran siendo sido tengo
tienes tiene tenemos teneis tienen tenga tengas tengamos tengais tengan tendre tendras tendra tendremos
tendreis tendran tendria tendrias tendriamos tendriais tendrian tenia tenias teniamos teniais tenian tuve
tuviste tuvo tuvimos tuvisteis tuvieron tuviera tuvieras tuvieramos tuvierais tuvieran teniendo tenido
tenida tenidos tenidas tened usted ustedes
""".split())).split())

# "Adaptadas al español peruano" significa dos cosas:
# 1) Se PROTEGEN palabras que la lista estándar elimina pero que en el smishing peruano
#    llevan señal:
#    - el registro: el estafador imita el trato corporativo formal ("Estimado cliente,
#      su cuenta...") y la conversación real tutea ("tu", "te");
#    - la negación ("no comparta su clave" frente a "no pierda su premio");
#    - "ya", partícula muy frecuente del habla peruana ("ya pe", "ya te yapeé").
PALABRAS_PROTEGIDAS = {"no", "si", "su", "sus", "tu", "tus", "te", "usted", "ustedes", "ya", "mi", "mis"}

# 2) Se AGREGAN las abreviaturas de chat peruanas que son solo palabras funcionales
#    escritas en corto (q = que, xq/pq = porque, d = de, tmb = también, x = por).
STOPWORDS_CHAT_PERU = {"q", "k", "xq", "pq", "porq", "d", "tmb", "tb", "x"}

STOPWORDS_PERU = (STOPWORDS_BASE_ES - PALABRAS_PROTEGIDAS) | STOPWORDS_CHAT_PERU

REGEX_TOKEN = re.compile(r"urltoken|celulartoken|telefonotoken|s/|\d+(?:[.,]\d+)?|[a-z]+")

REGEX_CELULAR = re.compile(r"(?:\+?51[\s-]?(?=9)|(?<!\d))9\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")

REGEX_FIJO = re.compile(r"\(?\b0?1\)?[\s-]\d{3}[\s-]?\d{4}\b|\b0?1-\d{7}\b")

# --- Desofuscación (defensa ante ataques adversarios, sección 5.5) ---
# Para evadir los filtros, los estafadores alteran las palabras clave: homoglifos (letras
# cirílicas o griegas idénticas a las latinas, p. ej. "ВСР" en vez de "BCP"), "leet"
# (bl0queada, v3rifique, Y4pe), letras separadas ("b l o q u e a d a", "B.C.P") y
# caracteres invisibles. desofuscar() deshace esas alteraciones ANTES del preprocesado.
# Solo se usa en la variante robusta (robusto=True); el modelo principal no la usa.
HOMOGLIFOS = str.maketrans({
    # cirílico -> latino
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I",
    # griego -> latino
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})

REGEX_INVISIBLES = re.compile("[­​-‏⁠﻿]")

# 3 o más letras sueltas separadas por espacio, punto, guion, guion bajo o asterisco.
REGEX_LETRAS_SEPARADAS = re.compile(r"(?<![^\W\d_])(?:[^\W\d_][ .\-_*]){2,}[^\W\d_](?![^\W\d_])")

REGEX_SEPARADORES = re.compile(r"[ .\-_*]")

# Palabras que empiezan con letra y mezclan letras con 0 3 4 5 7 1 @ $ (leet).
REGEX_PALABRA_LEET = re.compile(r"(?<![\w@$])[^\W\d_][\w@$]*[013457@$][\w@$]*")

LEET = {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}


def _quitar_leet(match):
    palabra = match.group(0)
    letras = sum(c.isalpha() for c in palabra)
    sustituibles = [i for i, c in enumerate(palabra) if c in LEET or c == "1"]
    # Se exige que sea una palabra (3+ letras), que no tenga otros dígitos (2, 6, 8, 9: es un
    # código o un año, p. ej. "SOAT2024") y que cada carácter sustituible toque una letra.
    es_correo = "@" in palabra and match.string[match.end():match.end() + 1] == "."
    if (letras < 3 or len(sustituibles) > letras or es_correo or "_" in palabra
            or any(c.isdigit() and c not in LEET and c != "1" for c in palabra)):
        return palabra
    for i in sustituibles:
        vecinos = palabra[max(i - 1, 0):i] + palabra[i + 1:i + 2]
        if not any(v.isalpha() for v in vecinos):
            return palabra
    salida = []
    for i, c in enumerate(palabra):
        if c == "1":  # "1" es "l" antes de vocal (b1oqueada, c1ave) y si no, "i" (s1n, pr1mer)
            siguiente = palabra[i + 1:i + 2].lower()
            salida.append("l" if siguiente and siguiente in "aeiouáéíóú" else "i")
        else:
            salida.append(LEET.get(c, c))
    return "".join(salida)


def desofuscar(texto):
    """Revierte homoglifos, leet, letras separadas y caracteres invisibles."""
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFKC", texto)   # letras "de ancho completo" o matemáticas -> normales
    t = REGEX_INVISIBLES.sub("", t)
    t = t.translate(HOMOGLIFOS)
    t = REGEX_LETRAS_SEPARADAS.sub(lambda m: REGEX_SEPARADORES.sub("", m.group(0)), t)
    return REGEX_PALABRA_LEET.sub(_quitar_leet, t)


def preprocesar_texto(texto, quitar_stopwords=True, robusto=False):
    """
    1. Normaliza: minúsculas y sin tildes (unifica "nación" y "nacion").
    2. Enmascara URLs ('urltoken') y teléfonos ('celulartoken', 'telefonotoken'): se
       conserva la señal "pide entrar a un enlace / llamar a un número" sin memorizar
       dominios ni números concretos.
    3. Tokeniza conservando montos, códigos numéricos y el símbolo de soles 's/', y
       además marca cada número por su forma: 'montotoken' (cifra tras 's/') y
       'codigotoken' (código de 4 a 8 dígitos). El número literal se conserva.
    4. Quita las stopwords adaptadas al español peruano (opcional, para la ablación).
    Con robusto=True, antes de todo se aplica desofuscar() (sección 5.5).
    """
    if not isinstance(texto, str):
        return ""
    if robusto:
        texto = desofuscar(texto)
    t = REGEX_ENLACE_AMPLIO.sub(" urltoken ", texto)
    t = REGEX_CELULAR.sub(" celulartoken ", t)
    t = REGEX_FIJO.sub(" telefonotoken ", t)
    t = normalizar(t)
    tokens = []
    for tok in REGEX_TOKEN.findall(t):
        if quitar_stopwords and tok in STOPWORDS_PERU:
            continue
        tokens.append(tok)
        if tok[0].isdigit():
            if len(tokens) >= 2 and tokens[-2] == "s/":
                tokens.append("montotoken")
            elif tok.isdigit() and 4 <= len(tok) <= 8:
                tokens.append("codigotoken")
    return " ".join(tokens)


class PreprocesadorTexto(BaseEstimator, TransformerMixin):
    """Paso de preprocesado dentro del pipeline de sklearn (sección 5): recibe los SMS en
    texto plano y devuelve el texto normalizado y tokenizado que usa el TF-IDF. No aprende
    nada de los datos (son reglas fijas)."""

    def __init__(self, quitar_stopwords=True, robusto=False):
        self.quitar_stopwords = quitar_stopwords
        self.robusto = robusto

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocesar_texto(t, self.quitar_stopwords, self.robusto) for t in X]
