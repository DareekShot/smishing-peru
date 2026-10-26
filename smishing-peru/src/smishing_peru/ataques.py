"""Ataques adversarios sobre los SMS para medir la robustez (sección 5.5).

Código tomado del notebook v9, sin cambios.
"""
import re

import numpy as np

from .catalogo import REGEX_ENLACE_AMPLIO


# --- 5.5 Robustez adversaria: ¿el modelo resiste SMS alterados para evadirlo? ---
# Salinas Bolaños y Ticona (2025) evaluaron su modelo con fraudes alterados por ataques
# adversarios, y Bajaj y Vishwakarma (2023) y Salman et al. (2024) muestran que los cambios
# de caracteres (sobre todo el espaciado entre letras) evaden a muchos clasificadores. Aquí
# se alteran los SMISHING del test con 6 ataques aplicados a las 3 palabras del SMS que más
# empujan al modelo hacia SMISHING (el peor caso: el atacante sabe qué palabras delatan el
# fraude) y se mide cuántos se siguen detectando (recall). Después se repite con la
# variante robusta del mismo modelo (desofuscar() antes del preprocesado, sección 4),
# entrenada con el mismo train. El HAM no se altera: al estafador solo le interesa que su
# SMISHING pase como HAM.

CIRILICO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
            "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
            "X": "Х"}

A_LEET = {"o": "0", "e": "3", "a": "4", "i": "1", "s": "5", "t": "7", "l": "1"}


def _espaciado(palabra, rng):
    return " ".join(palabra)


def _homoglifos(palabra, rng):
    return "".join(CIRILICO.get(c, c) for c in palabra)


def _leet(palabra, rng):
    posiciones = [i for i, c in enumerate(palabra) if c.lower() in A_LEET]
    elegidas = set(rng.sample(posiciones, min(2, len(posiciones))))
    return "".join(A_LEET[c.lower()] if i in elegidas else c for i, c in enumerate(palabra))


def _intercambio(palabra, rng):
    if len(palabra) < 4:
        return palabra
    i = rng.randint(1, len(palabra) - 3)  # se dejan fijas la primera y la última letra
    return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]


def _borrado(palabra, rng):
    return ""


def _mixto(palabra, rng):
    return rng.choice([_espaciado, _homoglifos, _leet, _intercambio])(palabra, rng)


ATAQUES = {
    "Espaciado entre letras": _espaciado,
    "Homoglifos (letras cirílicas)": _homoglifos,
    "Leet (números por letras)": _leet,
    "Intercambio de caracteres": _intercambio,
    "Borrado de palabras (EDA)": _borrado,
    "Mixto (uno al azar por palabra)": _mixto,
}

REGEX_PALABRA = re.compile(r"[^\W\d_]{3,}")  # palabras de 3+ letras (sin números ni signos)


def _puntaje(modelo, textos):
    """Qué tan SMISHING ve el modelo cada texto (función de decisión o probabilidad)."""
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(textos)
    return modelo.predict_proba(textos)[:, 1]


def palabras_clave_del_fraude(modelo, texto, n=3):
    """Las n palabras cuyo borrado más baja el puntaje de SMISHING (sin tocar URLs ni números)."""
    zonas_url = [m.span() for m in REGEX_ENLACE_AMPLIO.finditer(texto)]
    candidatas = [m for m in REGEX_PALABRA.finditer(texto)
                  if not any(a <= m.start() < b for a, b in zonas_url)]
    if not candidatas:
        return []
    variantes = [texto[:m.start()] + texto[m.end():] for m in candidatas]
    puntajes = _puntaje(modelo, [texto] + variantes)
    caidas = puntajes[0] - puntajes[1:]
    orden = np.argsort(-caidas)[:n]
    return [candidatas[i].span() for i in orden if caidas[i] > 0]


def atacar(texto, spans, funcion, rng):
    for inicio, fin in sorted(spans, reverse=True):  # de derecha a izquierda: no se corren los índices
        texto = texto[:inicio] + funcion(texto[inicio:fin], rng) + texto[fin:]
    return re.sub(r" {2,}", " ", texto).strip()
