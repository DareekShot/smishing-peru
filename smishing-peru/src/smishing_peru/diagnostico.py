"""Diagnóstico interpretable de un SMS (sección 6).

Código tomado del notebook v9. Único cambio: evaluar_sms_peruano recibe modelo_gatillo como
argumento, en lugar de leerlo del notebook.
"""
import re

from .caracteristicas import PALABRAS_URGENCIA
from .catalogo import REGEX_ENLACE_AMPLIO, detectar_entidades, normalizar


# Inferencia en vivo y diagnóstico interactivo

# Palabras gatillo por categoría psicológica (sin tildes). En la v4 el gatillo se detectaba
# con estas listas; en la v5 solo se usan para compararlas con el clasificador de abajo.
GATILLOS_DIAGNOSTICO = {
    "URGENCIA_MIEDO": ["urgente", "bloquead", "suspend", "expira", "vence", "inmediat", "24h", "24 horas",
                       "cancelad", "desactivad", "alerta", "no reconocid", "ultimo aviso", "hoy mismo",
                       "acceso desde otro dispositivo"],
    "AUTORIDAD": ["sunat", "reniec", "sat lima", "sutran", "migraciones", "poder judicial", "indecopi",
                  "pnp", "essalud", "onp", "embargo", "multa", "papeleta", "deuda", "cobranza coactiva",
                  "citacion", "denuncia", "ruc", "notificacion", "sbs"],
    "RECOMPENSA": ["felicidades", "ganaste", "ganador", "premio", "sorteo", "bono", "regalo", "seleccionad",
                   "gratis", "devolucion", "reembolso", "puntos", "millas", "preaprobado", "sin requisitos",
                   "trabajo desde casa", "pago diario", "beca"],
    "CONFIANZA_ENCOMIENDA": ["paquete", "encomienda", "envio", "aduana", "arancel", "courier", "serpost",
                             "olva", "shalom", "pedido", "entrega", "cambie de numero", "nuevo numero",
                             "codigo que te llego", "codigo de verificacion"],
}

# Pedido de contacto fuera del canal oficial (fraudes sin enlace)
REGEX_CONTACTO = re.compile(r"whatsapp|wa\.me|\+51|\b9\d{2}[ -]?\d{3}[ -]?\d{3}\b|llame al|llamar al|escribenos|escribe al")


def _gatillo_por_palabras(texto_norm):
    mejor, mejor_score = None, 0
    for gatillo, palabras in GATILLOS_DIAGNOSTICO.items():
        score = sum(p in texto_norm for p in palabras)
        if score > mejor_score:
            mejor, mejor_score = gatillo, score
    return mejor if mejor_score > 0 else None


def evaluar_sms_peruano(texto_sms, modelo, label_encoder, modelo_gatillo, verbose=True):
    '''
    Recibe un SMS en texto plano, lo pasa por el pipeline completo (preprocesamiento
    -> TF-IDF + features estilísticas -> modelo entrenado) y devuelve un diagnóstico
    interpretable. La clase sale de modelo.predict(), igual que en las métricas del test;
    el "riesgo %" es la probabilidad estimada de SMISHING (solo informativa).
    modelo_gatillo es el clasificador del gatillo psicológico entrenado en el notebook.
    '''
    pred = label_encoder.inverse_transform(modelo.predict([texto_sms]))[0]
    proba_smishing = modelo.predict_proba([texto_sms])[0][list(label_encoder.classes_).index("SMISHING")]
    es_smishing = pred == "SMISHING"
    clasificacion = "🚨 SMISHING" if es_smishing else "✅ HAM"

    texto_norm = normalizar(texto_sms)
    gatillo = modelo_gatillo.predict([texto_sms])[0] if es_smishing else None

    anomalias = []
    if REGEX_ENLACE_AMPLIO.search(texto_sms):
        anomalias.append("Contiene enlace/URL")
    entidades = detectar_entidades(texto_sms)
    if entidades:
        anomalias.append(f"Menciona entidad que opera en el Perú ({', '.join(entidades)})")
    if REGEX_CONTACTO.search(texto_norm):
        anomalias.append("Pide contactar por WhatsApp o por teléfono")
    if any(p in texto_norm for p in PALABRAS_URGENCIA):
        anomalias.append("Lenguaje de urgencia o amenaza de bloqueo")
    if sum(c.isdigit() for c in texto_sms) / max(len(texto_sms), 1) > 0.15:
        anomalias.append("Alta proporción de dígitos (montos/códigos)")

    resultado = {
        "texto": texto_sms,
        "clasificacion": clasificacion,
        "riesgo_pct": round(proba_smishing * 100, 2),
        "gatillo_psicologico": gatillo,
        "anomalias_detectadas": anomalias,
    }

    if verbose:
        print(f"SMS: {texto_sms}")
        print(f"  -> {clasificacion}  |  Riesgo: {resultado['riesgo_pct']}%")
        if gatillo:
            print(f"  -> Gatillo psicologico probable: {gatillo}")
        if anomalias:
            print(f"  -> Señales: {', '.join(anomalias)}")
        print()

    return resultado
