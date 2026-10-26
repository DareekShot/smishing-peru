"""Dataset final: control de calidad, enlaces del original, trazabilidad y versión para publicar
(secciones 3 a 3.4).

Código tomado del notebook v9, sin cambios.
"""
import re

import numpy as np
import pandas as pd

from .catalogo import REGEX_ENLACE_AMPLIO
from .gemini import (
    _es_rechazo, idioma_aproximado, normalizar_para_duplicados, quitar_casi_duplicados,
)


# Construcción del dataset final (en español peruano) con control de calidad.


def control_calidad(df, largo_min=15, largo_max=320, umbral_casi_duplicado=0.90):
    informe = {"inicial": len(df)}
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip().str.strip('"')
    df = df[(df["text"] != "") & (df["text"].str.lower() != "nan")]
    df = df[~df["text"].map(_es_rechazo)]
    informe["tras quitar vacíos/rechazos del modelo"] = len(df)

    largo = df["text"].str.len()
    df = df[(largo >= largo_min) & (largo <= largo_max)]
    informe[f"tras filtro de longitud ({largo_min}-{largo_max} car.)"] = len(df)

    # Red de seguridad: ningún SMS crudo en inglés entra al dataset (solo sus versiones
    # adaptadas), pero si Gemini devolviera alguno sin traducir, se detecta y se elimina.
    df["idioma"] = df["text"].map(idioma_aproximado)
    informe["detectados en inglés por la heurística (eliminados)"] = int((df["idioma"] == "en").sum())
    informe["cortos sin palabras funcionales reconocibles (se conservan)"] = int((df["idioma"] == "indeterminado").sum())
    df = df[df["idioma"] != "en"]

    df["_clave"] = df["text"].map(normalizar_para_duplicados)
    df = df.drop_duplicates(subset="_clave").drop(columns="_clave").reset_index(drop=True)
    informe["tras quitar duplicados exactos (normalizando URLs y números)"] = len(df)

    df, n_casi = quitar_casi_duplicados(df, umbral=umbral_casi_duplicado)
    informe[f"casi-duplicados eliminados (coseno >= {umbral_casi_duplicado})"] = n_casi
    informe["final"] = len(df)
    return df, informe


def muestrear_estratificado(df, n, columna, seed=42):
    """Toma n filas repartidas lo más parejo posible entre los valores de 'columna'."""
    if len(df) <= n:
        return df
    grupos = {k: g.sample(frac=1, random_state=seed) for k, g in df.groupby(columna, dropna=False)}
    cupo = {k: 0 for k in grupos}
    restantes = n
    while restantes > 0:
        activos = [k for k in grupos if cupo[k] < len(grupos[k])]
        if not activos:
            break
        por_grupo = max(1, restantes // len(activos))
        for k in activos:
            extra = min(por_grupo, len(grupos[k]) - cupo[k], restantes)
            cupo[k] += extra
            restantes -= extra
            if restantes == 0:
                break
    return pd.concat([grupos[k].iloc[:c] for k, c in cupo.items()], ignore_index=True)


def construir_dataset_final(df_adaptado, df_generado, n_adaptado_por_clase=1000,
                            n_generado_por_clase=1500, seed=42):
    columnas = ["text", "label", "trigger", "source", "origen", "grupo", "entidades", "categoria",
                "escenario", "canal", "estilo", "source_original", "text_original"]
    adaptado = df_adaptado.assign(grupo="A_" + df_adaptado["id_ref"].astype(str))
    # Los 10 SMS de una misma llamada a Gemini comparten plantilla, entidad y escenario:
    # forman un "grupo" que en el entrenamiento nunca se parte entre train y test.
    generado = df_generado.assign(grupo="G_" + df_generado["plan_id"].astype(str))
    pool = pd.concat([adaptado, generado], ignore_index=True)
    for c in columnas:
        if c not in pool.columns:
            pool[c] = np.nan
    pool = pool[columnas]
    # Se barajan juntos para que la deduplicación no favorezca a un origen por su orden.
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    pool, informe = control_calidad(pool)

    partes = []
    for (origen, label), g in pool.groupby(["origen", "label"]):
        objetivo = n_adaptado_por_clase if origen == "adaptado_internacional" else n_generado_por_clase
        col_estrato = "escenario" if origen == "generado_llm" else "trigger"
        if origen == "generado_llm":
            # El refuerzo (familiar que cambió de número / avisos con teléfono) entra completo;
            # el muestreo solo recorta el resto de los generados.
            es_refuerzo = g["grupo"].astype(str).str.startswith("G_R_")
            partes.append(g[es_refuerzo])
            objetivo -= int(es_refuerzo.sum())
            g = g[~es_refuerzo]
        partes.append(muestrear_estratificado(g, objetivo, col_estrato, seed=seed))
    df_final = pd.concat(partes, ignore_index=True)

    # Si a una clase le faltan SMS adaptados (se agotaron los candidatos reales), se
    # completa con generados de la MISMA clase: se mantiene el balance 2,500 / 2,500.
    objetivo_clase = n_adaptado_por_clase + n_generado_por_clase
    for label in ["SMISHING", "HAM"]:
        faltan = objetivo_clase - int((df_final["label"] == label).sum())
        if faltan > 0:
            usados = set(df_final["text"])
            reserva = pool[(pool["label"] == label) & (pool["origen"] == "generado_llm") & ~pool["text"].isin(usados)]
            extra = reserva.sample(min(faltan, len(reserva)), random_state=seed)
            df_final = pd.concat([df_final, extra], ignore_index=True)
            print(f"Aviso: {label} tenía {faltan} SMS menos que el objetivo; se completaron {len(extra)} "
                  f"con SMS generados de la misma clase.")
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_final["label"] = df_final["label"].str.upper()
    return df_final, informe


# --- 3.0 Enlaces: en los SMS adaptados se usa el enlace del mensaje original ---
# Las instrucciones de adaptación (2.4) le pedían a Gemini inventar un enlace peruano cuando
# el original traía uno, y Gemini solía poner el nombre de la entidad en el dominio (por
# ejemplo, bcp-seguro.pe). En los originales en inglés eso no es una regla: solo cerca de 1 de
# cada 4 SMISHING con enlace lleva en el dominio una palabra del mensaje. Para no inventar
# enlaces en los SMS que vienen de mensajes reales, a cada SMS adaptado se le pone el enlace
# de su original:
#  - si los dos tienen enlace, los del adaptado se reemplazan por los del original, en orden;
#  - si el original tenía enlace y el adaptado no, se deja como está (Gemini cambió el canal,
#    por ejemplo a un teléfono);
#  - si el original NO tenía enlace y Gemini agregó uno, el SMS se excluye: quitar el enlace
#    dejaría la frase incompleta ("ingrese aquí para ...").
# Los SMS generados no se tocan: son sintéticos y solo se usan para entrenar (sección 5.0).
# El modelo no lee el dominio, porque los enlaces pasan a "urltoken" antes del TF-IDF.


def buscar_enlaces(texto):
    """Enlaces del texto (también los ofuscados), sin contar dominios de correo (usuario@gmail.com)."""
    texto = str(texto)
    return [m for m in REGEX_ENLACE_AMPLIO.finditer(texto) if m.start() == 0 or texto[m.start() - 1] != "@"]


def _sin_puntuacion_final(enlace):
    return enlace.rstrip(".,;:!?)")


def usar_enlace_original(texto, texto_original):
    """Devuelve el SMS adaptado con los enlaces de su original y qué se hizo con él."""
    del_original = [_sin_puntuacion_final(m.group(0)) for m in buscar_enlaces(texto_original)]
    en_adaptado = buscar_enlaces(texto)
    if not en_adaptado:
        return texto, "Sin enlace en el adaptado (se deja igual)"
    if not del_original:
        return texto, "Gemini agregó un enlace que el original no tenía (se excluye)"
    for k, m in reversed(list(enumerate(en_adaptado))):
        cola = m.group(0)[len(_sin_puntuacion_final(m.group(0))):]
        texto = texto[:m.start()] + (del_original[min(k, len(del_original) - 1)] + cola) + texto[m.end():]
    return texto, "Se puso el enlace del original"


# --- 3.3 Trazabilidad: el recorrido de los datos, desde los 3 CSV originales hasta el dataset final ---
# Cada etapa muestra cuántos SMS hay de cada fuente y clase, y todo se guarda en la
# carpeta "evidencias/" para poder revisarlo fuera del notebook.

NOMBRE_FUENTE = {
    "uci_sms_spam": "UCI SMS Spam Collection",
    "dataset_5971": "Dataset_5971 (Mendeley)",
    "analysis_dataset_real": "analysisdataset (smishing real)",
    "gemini_llm_peru": "Generado con Gemini",
    "gemini_refuerzo": "Generado con Gemini (refuerzo)",
}


def fuente_detallada(d):
    """Fuente de cada SMS del dataset final: el CSV original (si es adaptado) o Gemini."""
    fuente = d["source_original"].where(d["origen"] == "adaptado_internacional", "gemini_llm_peru")
    es_refuerzo = d["categoria"].astype(str).str.startswith("refuerzo")
    return fuente.mask(es_refuerzo, "gemini_refuerzo")


def _conteo_por_fuente(d, etapa, columna_fuente):
    t = d.groupby([columna_fuente, "label"]).size().unstack(fill_value=0)
    t = t.reindex(columns=["HAM", "SMISHING"], fill_value=0)
    t["Total"] = t.sum(axis=1)
    t = t.reset_index().rename(columns={columna_fuente: "Fuente"})
    t["Fuente"] = t["Fuente"].map(lambda f: NOMBRE_FUENTE.get(f, f))
    t.insert(0, "Etapa", etapa)
    return t


# --- 3.4 Versión del corpus para publicar: enlaces reales desactivados ---
# Desde la v7, los SMS adaptados llevan el enlace real de su mensaje original (3.0), y varios
# son enlaces maliciosos reales (por ejemplo, los de Smishtank). Para publicar el corpus se
# desactivan con la convención habitual en seguridad: "http" pasa a "hxxp" y los puntos del
# dominio a "[.]", así nadie los abre por accidente. No se inventa ningún enlace, y el dataset
# de modelado no cambia.


def desactivar_enlace(enlace):
    enlace = re.sub(r"^http", "hxxp", enlace, flags=re.IGNORECASE)
    esquema, resto = re.match(r"((?:hxxps?:/*|//)?)(.*)", enlace, flags=re.IGNORECASE | re.DOTALL).groups()
    dominio, barra, ruta = resto.partition("/")
    return esquema + dominio.replace(".", "[.]") + barra + ruta


def desactivar_enlaces(texto):
    for m in reversed(buscar_enlaces(texto)):
        texto = texto[:m.start()] + desactivar_enlace(m.group(0)) + texto[m.end():]
    return texto
