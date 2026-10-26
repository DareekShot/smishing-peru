"""Llamadas a Gemini con salida JSON y reintentos; idioma y casi-duplicados (sección 2.3).

Código tomado del notebook v9, sin cambios.
"""
import re
import time

import numpy as np
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer

from .catalogo import REGEX_URL, normalizar


# Utilidades comunes: (1) llamadas a Gemini con salida JSON con esquema, reintentos
# con espera exponencial ante límites de cuota (429) y detección de rechazos;
# (2) control de calidad: idioma, duplicados y casi-duplicados.

CONTEXTO_INVESTIGACION_TXT = (
    "Contexto: esto es investigación académica de ciberseguridad defensiva (tesis universitaria) "
    "para ENTRENAR UN CLASIFICADOR que DETECTE y BLOQUEE smishing real, igual que los datasets "
    "públicos de phishing ya existentes (UCI SMS Spam Collection, entre otros). Los mensajes son "
    "ejemplos sintéticos y genéricos, no dirigidos a ninguna persona real. Todos los enlaces, "
    "teléfonos y nombres de personas deben ser ficticios."
)

PATRONES_RECHAZO = [
    "cannot fulfill", "cannot assist", "can't assist", "cannot help", "can't help",
    "i cannot generate", "i can't generate", "i'm sorry", "i am sorry",
    "as an ai", "as a language model", "i'm not able to", "i am not able to",
    "no puedo generar", "no puedo ayudar", "no puedo crear", "no puedo cumplir",
    "lo siento, no puedo", "no está permitido", "no puedo proporcionar",
]


def _es_rechazo(texto):
    """True si el 'mensaje' devuelto es en realidad un rechazo del modelo, no un SMS."""
    t = str(texto).strip().lower()
    return any(p in t for p in PATRONES_RECHAZO)


def _diagnosticar_bloqueo(response):
    """Explica por qué la API no devolvió nada utilizable (filtro de seguridad, corte, etc.)."""
    try:
        retro_prompt = getattr(response, "prompt_feedback", None)
        if retro_prompt is not None and getattr(retro_prompt, "block_reason", None):
            return f"el prompt fue bloqueado por el filtro de seguridad de Gemini ({retro_prompt.block_reason})"
        candidatos = getattr(response, "candidates", None) or []
        if candidatos:
            razon = getattr(candidatos[0], "finish_reason", None)
            if razon is not None and str(razon).split(".")[-1] not in ("STOP", "None"):
                return f"la respuesta se cortó por finish_reason={razon} (probable filtro de seguridad o límite de tokens)"
    except Exception:
        pass
    return "respuesta vacía sin más detalle (probable filtro de seguridad de Gemini para este tema)"


SAFETY_SETTINGS = [
    # Umbrales relajados a "solo bloquear riesgo alto": los filtros por defecto bloquean
    # con frecuencia la suplantación de entidades aunque el propósito (ejemplos para un
    # clasificador defensivo) sea legítimo.
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
]


def llamar_gemini(client, modelo, prompt, esquema, temperatura=0.9, max_reintentos=5):
    """Llama a Gemini pidiendo JSON con el esquema pydantic dado.
    Devuelve el objeto ya validado, o None si se agotan los reintentos."""
    espera = 2
    for intento in range(max_reintentos):
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperatura,
                    response_mime_type="application/json",
                    response_schema=esquema,
                    safety_settings=SAFETY_SETTINGS,
                ),
            )
            if response.parsed is None:
                raise ValueError(_diagnosticar_bloqueo(response))
            return response.parsed
        except errors.APIError as e:
            if e.code == 429 or (e.code is not None and e.code >= 500):
                print(f"  [reintento {intento+1}/{max_reintentos}] error {e.code}, esperando {espera}s...")
                time.sleep(espera)
                espera *= 2
                continue
            raise
        except Exception as e:
            print(f"  [reintento {intento+1}/{max_reintentos}] {e}")
            time.sleep(espera)
            espera *= 2
    print("  -> Llamada descartada tras agotar reintentos.")
    return None


# --- Control de calidad ---
_MARCADORES_ES = set("""de la que el en y los se del las un por con una su para es al lo como más pero sus le ya
este sí porque esta entre cuando muy sin sobre también hasta hay donde desde todo nos todos les ni ese eso esto
qué unos yo otro él tanto esa estos mucho nada cual ella estar estas algo mi mis tú te ti tu tus usted ustedes
pe oe causa hola gracias tienes tiene cuenta pago hoy mañana aquí ahora bien voy vas está estás soy eres
señor estimado cliente favor ingresa ingrese verifica llama mas q xq tmb listo chevere chévere bacan bacán
jaja jajaja vemos bueno dale vale porfa sale chau besos abrazo amor ma pa hermano hermana pata tío tía tio tia
buenas días dias tardes noches tranqui hijo hija ya""".split())

_MARCADORES_EN = set("""the you your to and is are for of in it my we have will be this that with on at call
please get now just can what not but if do from ur im dont send txt text reply free won claim yes yeah
how why when where who much good well done day days since then there their they she his her was were
been had got going gonna wanna love luv all am its it's i'm i've i'll don't can't won't
already while ago keep know think tell told see watch home night tomorrow today later sorry thanks thank""".split())


def idioma_aproximado(texto):
    """Heurística ligera: compara palabras funcionales del español y del inglés.
    'indeterminado' = no se reconoce ninguna (típico de SMS muy cortos)."""
    palabras = re.findall(r"[a-záéíóúñü']+", str(texto).lower())
    es = sum(p in _MARCADORES_ES for p in palabras)
    en = sum(p in _MARCADORES_EN for p in palabras)
    if en > es:
        return "en"
    if es > 0 or re.search(r"[áéíóúñ¿¡]", str(texto).lower()):
        return "es"
    return "indeterminado"


def normalizar_para_duplicados(texto):
    t = REGEX_URL.sub(" url ", str(texto))
    t = normalizar(t)
    t = re.sub(r"\d+", "0", t)
    return re.sub(r"\s+", " ", t).strip()


def quitar_casi_duplicados(df, columna="text", umbral=0.90, bloque=500):
    """Elimina mensajes casi idénticos (similitud coseno de n-gramas de caracteres >= umbral),
    conservando el primero. Devuelve (df_filtrado, n_eliminados)."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(
        df[columna].map(normalizar_para_duplicados))
    conservar = np.ones(X.shape[0], dtype=bool)
    for ini in range(0, X.shape[0], bloque):
        sim = (X[ini:ini + bloque] @ X.T).toarray()
        for fila in range(sim.shape[0]):
            i = ini + fila
            if not conservar[i]:
                continue
            similares = np.where(sim[fila, :i] >= umbral)[0]
            if len(similares) and conservar[similares].any():
                conservar[i] = False
    return df[conservar].reset_index(drop=True), int((~conservar).sum())
