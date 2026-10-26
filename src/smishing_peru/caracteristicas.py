"""Variables estilísticas y estructurales, con lógica difusa (sección 4).

Código tomado del notebook v9, sin cambios.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .catalogo import CATALOGO_ENTIDADES, REGEX_ENLACE_AMPLIO, normalizar
from .preprocesamiento import desofuscar


# Indicadores de urgencia / bloqueo (gatillo URGENCIA_MIEDO), ya sin tildes.
PALABRAS_URGENCIA = [
    "urgente", "bloquead", "suspend", "expira", "vence", "inmediat", "ahora mismo", "hoy mismo",
    "24 horas", "24h", "ultimo aviso", "verifique", "verificar ahora", "sera cancelad",
    "sera desactivad", "desactivad", "embargo", "multa", "alerta", "no reconocid",
]

# Un solo patrón con TODOS los alias del catálogo: da el mismo resultado que
# bool(detectar_entidades(texto)), pero ~30 veces más rápido (importa porque en la v5 el
# extractor se recalcula dentro de cada pliegue de la validación cruzada).
_ALIAS_CATALOGO = sorted({a for ents in CATALOGO_ENTIDADES.values() for alias in ents.values() for a in alias},
                         key=len, reverse=True)

_PATRON_CUALQUIER_ENTIDAD = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in _ALIAS_CATALOGO) + r")(?![a-z0-9])")


def menciona_contexto_peruano(texto):
    """Entidad peruana del catálogo, o montos en soles."""
    t = normalizar(texto)
    return bool(_PATRON_CUALQUIER_ENTIDAD.search(t)) or "s/" in t or " soles" in f" {t}"


def _membresia_triangular(x, a, b, c):
    '''Función de membresía triangular estándar de lógica difusa: sube linealmente
    de 0 a 1 entre a y b, y baja linealmente de 1 a 0 entre b y c.'''
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _fuzzificar(x, bajo, medio, alto):
    '''Convierte un valor crudo en 3 grados de membresía difusa (bajo/medio/alto).
    v5: "bajo" y "alto" son hombros (trapecios abiertos): todo valor por debajo de "bajo"
    tiene membresía 1 en "bajo" y todo valor por encima de "alto", membresía 1 en "alto".
    En la v4 eran triángulos cerrados, así que los valores extremos (p. ej. un SMS todo en
    MAYÚSCULAS o de más de 210 caracteres) quedaban con membresía 0 en los tres conjuntos.'''
    g_bajo = 1.0 if x <= bajo else _membresia_triangular(x, bajo - (medio - bajo), bajo, medio)
    g_medio = _membresia_triangular(x, bajo, medio, alto)
    g_alto = 1.0 if x >= alto else _membresia_triangular(x, medio, alto, alto + (alto - medio))
    return g_bajo, g_medio, g_alto


class StylisticFeatureExtractor(BaseEstimator, TransformerMixin):
    '''
    Extrae 7 señales estructurales de un SMS: longitud, proporción de dígitos,
    presencia de URL, mención de contexto peruano (entidad del catálogo o soles),
    score de urgencia, proporción de mayúsculas y densidad de puntuación de énfasis.

    usar_logica_difusa=True: las 5 variables continuas se convierten en 3 grados de
    membresía difusa cada una (17 columnas). False: las 7 variables crudas (ablación).
    robusto=True: las señales de contenido se buscan en el texto desofuscado (sección 5.5).
    '''

    def __init__(self, usar_logica_difusa=True, robusto=False):
        self.usar_logica_difusa = usar_logica_difusa
        self.robusto = robusto

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        filas = []
        for texto in X:
            texto_raw = texto if isinstance(texto, str) else ""
            # Las señales de contenido (URL, entidad, urgencia) se buscan en el texto
            # desofuscado si robusto=True; las de forma (longitud, dígitos, mayúsculas,
            # puntuación) siempre se miden sobre el SMS tal como llegó.
            texto_contenido = desofuscar(texto_raw) if self.robusto else texto_raw
            texto_norm = normalizar(texto_contenido)
            longitud = len(texto_raw)
            n_digitos = sum(c.isdigit() for c in texto_raw)
            prop_digitos = n_digitos / longitud if longitud > 0 else 0.0
            tiene_url = 1.0 if REGEX_ENLACE_AMPLIO.search(texto_contenido) else 0.0
            tiene_kw_peru = 1.0 if menciona_contexto_peruano(texto_contenido) else 0.0
            n_urgencia = sum(texto_norm.count(p) for p in PALABRAS_URGENCIA)
            score_urgencia = min(n_urgencia / 3.0, 1.0)

            letras = [c for c in texto_raw if c.isalpha()]
            prop_mayusculas = (sum(c.isupper() for c in letras) / len(letras)) if letras else 0.0
            n_puntuacion = texto_raw.count("!") + texto_raw.count("?")
            score_puntuacion = min(n_puntuacion / 3.0, 1.0)

            if self.usar_logica_difusa:
                l_bajo, l_medio, l_alto = _fuzzificar(longitud, 40, 90, 150)
                d_bajo, d_medio, d_alto = _fuzzificar(prop_digitos, 0.02, 0.10, 0.25)
                u_bajo, u_medio, u_alto = _fuzzificar(score_urgencia, 0.0, 0.4, 0.8)
                m_bajo, m_medio, m_alto = _fuzzificar(prop_mayusculas, 0.05, 0.20, 0.40)
                p_bajo, p_medio, p_alto = _fuzzificar(score_puntuacion, 0.0, 0.4, 0.8)
                filas.append([
                    l_bajo, l_medio, l_alto,
                    d_bajo, d_medio, d_alto,
                    tiene_url, tiene_kw_peru,
                    u_bajo, u_medio, u_alto,
                    m_bajo, m_medio, m_alto,
                    p_bajo, p_medio, p_alto,
                ])
            else:
                filas.append([
                    longitud, prop_digitos, tiene_url, tiene_kw_peru, score_urgencia,
                    prop_mayusculas, score_puntuacion,
                ])
        return np.array(filas, dtype=float)

    def get_feature_names_out(self, input_features=None):
        if self.usar_logica_difusa:
            return np.array([
                "longitud_bajo", "longitud_medio", "longitud_alto",
                "digitos_bajo", "digitos_medio", "digitos_alto",
                "tiene_url", "tiene_contexto_peru",
                "urgencia_bajo", "urgencia_medio", "urgencia_alto",
                "mayusculas_bajo", "mayusculas_medio", "mayusculas_alto",
                "puntuacion_bajo", "puntuacion_medio", "puntuacion_alto",
            ])
        return np.array([
            "longitud_msg", "proporcion_digitos", "tiene_url",
            "tiene_contexto_peru", "score_urgencia",
            "proporcion_mayusculas", "score_puntuacion",
        ])
