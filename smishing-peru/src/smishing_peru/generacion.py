"""Generación ampliada y controlada de SMS peruanos con Gemini (sección 2.5).

Código tomado del notebook v9, sin cambios.
"""
import math
import os
import random
import time
from itertools import cycle

import pandas as pd
from pydantic import BaseModel

from .catalogo import (
    CANALES_SMISHING, CATALOGO_ENTIDADES, ESCENARIOS_HAM, ESCENARIOS_SMISHING, ESTILOS,
    ESTILOS_HAM, LUGARES_PERU, detectar_entidades,
)
from .gemini import CONTEXTO_INVESTIGACION_TXT, _es_rechazo, llamar_gemini


# Generación ampliada con Gemini: en vez de pedir "180 mensajes de urgencia" y confiar
# en que el modelo varíe solo, cada llamada recibe una combinación distinta de
# gatillo x escenario x entidad x canal x estilo. La variedad queda garantizada por
# el plan (y es medible), no por la suerte.


class TandaMensajes(BaseModel):
    mensajes: list[str]


INSTRUCCIONES_TRIGGER = {
    "URGENCIA_MIEDO": "generar miedo o urgencia inmediata (plazos cortos, bloqueos, cargos no reconocidos)",
    "AUTORIDAD": "suplantar a una entidad con autoridad (Estado, regulador, cobranza) para intimidar",
    "RECOMPENSA": "ofrecer un beneficio inesperado (bono, premio, devolución, préstamo, trabajo) para generar codicia",
    "CONFIANZA_ENCOMIENDA": "explotar la confianza: una entrega esperada, un familiar o un soporte técnico conocido",
}

REGLAS_FORMATO_TXT = (
    "Reglas obligatorias:\n"
    "- Cada mensaje debe sonar como un SMS real enviado en el Perú: máximo ~180 caracteres, español peruano.\n"
    "- Montos en soles escritos explícitamente (ej. 'S/ 350.00') y códigos numéricos reales, sin placeholders.\n"
    "- Nombres de personas, enlaces y teléfonos siempre ficticios.\n"
    "- Mensajes distintos entre sí: no empieces todos igual ni repitas la misma plantilla cambiando solo el monto.\n"
    "- Sin explicaciones, numeración ni comillas: solo la lista de mensajes en el campo pedido."
)

# --- Refuerzo: dos casos muy comunes en el Perú que el modelo fallaba en las pruebas a mano ---
# 1) Fraude del "familiar o amigo que cambió de número": el estafador suplanta a una PERSONA
#    (no a una empresa) y pide el dinero por Yape o Plin como lo haría un conocido.
# 2) Avisos LEGÍTIMOS que terminan con el número de la central ("si no reconoces esta
#    operación, llama al..."): sin ellos, el modelo aprende que "llama al..." siempre es fraude.
# Estos SMS entran completos al dataset final (el muestreo de la sección 3 no los recorta).
N_REFUERZO_POR_CASO = 140

ESCENARIOS_REFUERZO = {
    "refuerzo_familiar": [
        "hijo, hija, sobrino o nieto que avisa que cambió de número porque se le malogró o le robaron "
        "el celular, y pide que le yapeen o plineen un monto por una emergencia",
        "'ma' o 'pa, este es mi nuevo número, guárdalo', y enseguida un pedido de dinero urgente",
        "amigo o compañero de trabajo que pide un préstamo rápido y promete devolverlo mañana o a fin de mes",
        "familiar que dice estar en un apuro (choque, clínica, comisaría, se quedó sin batería) y pide "
        "dinero sin poder llamar",
        "conocido que pide que le reenvíes el código de 6 dígitos que 'te llegó por error' a tu celular",
    ],
    "refuerzo_aviso_telefono": [
        "aviso de consumo con tarjeta que termina indicando que, si no lo reconoces, llames a la central",
        "aviso de transferencia o yapeo enviado desde tu cuenta, con el número de la central para reportarlo",
        "aviso de retiro en cajero o agente, con el número de atención al cliente",
        "aviso de cambio de clave, correo o celular registrado en la app, con el número de la central",
        "recordatorio de pago de cuota o recibo con el número de consultas, sin amenazas",
    ],
}

SECTORES_AVISO_TELEFONO = ["banca", "cajas_financieras", "billeteras_fintech", "telecom", "servicios"]


def _repartidor(lista, rng):
    """Iterador cíclico sobre una copia barajada: reparte los elementos de forma pareja."""
    copia = list(lista)
    rng.shuffle(copia)
    return cycle(copia)


def planificar_generacion(n_smishing, n_ham, tam_lote=10, seed=42, n_refuerzo=N_REFUERZO_POR_CASO):
    """Arma el plan de llamadas a Gemini. Cada fila es una llamada con su combinación."""
    rng = random.Random(seed)
    por_sector = {s: _repartidor(list(e), rng) for s, e in CATALOGO_ENTIDADES.items()}
    por_escenario = {}  # repartidor propio para escenarios con entidades concretas
    canales = _repartidor(["url", "url", "telefono", "whatsapp", "responder"], rng)
    estilos = _repartidor(list(ESTILOS), rng)
    estilos_ham = _repartidor(["natural", "natural", "breve", "detallado"], rng)
    plan = []

    def elegir_entidades(escenario, sectores, concretas, k):
        if concretas:
            if escenario not in por_escenario:
                por_escenario[escenario] = _repartidor(concretas, rng)
            fuente = [por_escenario[escenario]]
        elif sectores:
            fuente = [por_sector[s] for s in sectores]
        else:
            return []
        elegidas = []
        for i in range(k * 2):
            ent = next(fuente[i % len(fuente)])
            if ent not in elegidas:
                elegidas.append(ent)
            if len(elegidas) == k:
                break
        return elegidas

    llamadas_por_gatillo = math.ceil(n_smishing / tam_lote / len(ESCENARIOS_SMISHING))
    for gatillo, escenarios in ESCENARIOS_SMISHING.items():
        ciclo_esc = cycle(escenarios)
        for k in range(llamadas_por_gatillo):
            escenario, sectores, concretas = next(ciclo_esc)
            plan.append({
                # id estable: si luego se amplía el plan, las llamadas ya hechas conservan su id
                "plan_id": f"S_{gatillo}_{k:03d}", "label": "SMISHING", "trigger": gatillo,
                "categoria": gatillo, "escenario": escenario,
                "entidades_sugeridas": "|".join(elegir_entidades(escenario, sectores, concretas, 3)),
                "canal": next(canales), "estilo": next(estilos),
                "lugar": rng.choice(LUGARES_PERU), "n": tam_lote,
            })

    # La ficha define el HAM como mensajes transaccionales o promocionales legítimos:
    # en lo generado predominan esos (80%); la conversación personal ya llega desde UCI.
    pesos_ham = {"notificacion_legitima": 0.60, "promocional_legitimo": 0.20, "conversacion": 0.20}
    llamadas_ham = math.ceil(n_ham / tam_lote)
    enlace_oficial = cycle([False, False, True])
    for categoria, peso in pesos_ham.items():
        ciclo_esc = cycle(ESCENARIOS_HAM[categoria])
        for k in range(max(1, round(llamadas_ham * peso))):
            escenario, sectores, concretas = next(ciclo_esc)
            plan.append({
                "plan_id": f"H_{categoria}_{k:03d}", "label": "HAM", "trigger": None,
                "categoria": categoria, "escenario": escenario,
                "entidades_sugeridas": "|".join(elegir_entidades(escenario, sectores, concretas, 2)),
                "canal": "enlace_oficial" if (categoria != "conversacion" and next(enlace_oficial)) else "sin_enlace",
                "estilo": next(estilos_ham), "lugar": rng.choice(LUGARES_PERU), "n": tam_lote,
            })

    # Refuerzo: se planifica después del resto para que los plan_id anteriores no cambien.
    llamadas_refuerzo = math.ceil(n_refuerzo / tam_lote)
    ciclo_familiar = cycle(ESCENARIOS_REFUERZO["refuerzo_familiar"])
    billeteras = _repartidor(["Yape", "Plin", "Yape", "Plin", "Tunki"], rng)
    estilos_familiar = _repartidor(["descuidado", "informal_peruano", "informal_peruano", "breve"], rng)
    for k in range(llamadas_refuerzo):
        plan.append({
            "plan_id": f"R_familiar_{k:03d}", "label": "SMISHING", "trigger": "CONFIANZA_ENCOMIENDA",
            "categoria": "refuerzo_familiar", "escenario": next(ciclo_familiar),
            "entidades_sugeridas": next(billeteras), "canal": "pago_entre_personas",
            "estilo": next(estilos_familiar), "lugar": rng.choice(LUGARES_PERU), "n": tam_lote,
        })
    ciclo_aviso = cycle(ESCENARIOS_REFUERZO["refuerzo_aviso_telefono"])
    for k in range(llamadas_refuerzo):
        # Se rota el orden de los sectores para que cada llamada use entidades de otro sector.
        sectores = SECTORES_AVISO_TELEFONO[k % 5:] + SECTORES_AVISO_TELEFONO[:k % 5]
        plan.append({
            "plan_id": f"R_aviso_telefono_{k:03d}", "label": "HAM", "trigger": None,
            "categoria": "refuerzo_aviso_telefono", "escenario": next(ciclo_aviso),
            "entidades_sugeridas": "|".join(elegir_entidades(None, sectores[:2], None, 2)),
            "canal": "telefono_central", "estilo": next(estilos_ham),
            "lugar": rng.choice(LUGARES_PERU), "n": tam_lote,
        })
    return pd.DataFrame(plan)


def _prompt_generar(p):
    entidades = p["entidades_sugeridas"].replace("|", ", ") if p["entidades_sugeridas"] else ""
    if p["categoria"] == "refuerzo_familiar":
        return (
            f"{CONTEXTO_INVESTIGACION_TXT}\n\n"
            f"Genera {p['n']} SMS de estafa distintos entre sí, del tipo 'familiar o amigo que cambió de "
            "número', dirigidos a usuarios en el Perú.\n"
            f"- Escenario: {p['escenario']}.\n"
            "- El estafador se hace pasar por una PERSONA conocida (hijo, sobrino, primo, amigo, compañero), "
            "NO por un banco ni por una empresa: no lo escribas como notificación ni pongas una entidad como remitente.\n"
            f"- Pide el dinero por {entidades} como lo pediría una persona ('yapéame', 'plinéame', "
            f"'pásame por {entidades}'), con montos entre S/ 50 y S/ 800, o pide reenviar un código.\n"
            "- Sin enlaces. Puede incluir un número de celular ficticio (9XX XXX XXX).\n"
            f"- Estilo: {ESTILOS[p['estilo']]}. Trato familiar (ma, pa, tío, tía, primo, causa) y nombres peruanos.\n"
            f"- Si encaja, menciona {p['lugar']}.\n\n{REGLAS_FORMATO_TXT}"
        )
    if p["categoria"] == "refuerzo_aviso_telefono":
        return (
            f"Genera {p['n']} SMS LEGÍTIMOS (no fraudulentos) distintos entre sí, típicos del Perú.\n"
            f"- Tipo: {p['escenario']}, con el formato real que usaría la entidad ({entidades}).\n"
            "- Cada mensaje termina con el número de la central de la entidad en formato peruano "
            "(por ejemplo (01) 311-XXXX, 01-595-XXXX o 0800-XXXXX), como en los avisos reales.\n"
            "- Sin enlaces. No pidas claves, códigos ni datos personales, ni que respondan el SMS.\n"
            "- Sin urgencia, amenazas ni premios: es un aviso informativo.\n"
            f"- Longitud: {ESTILOS_HAM[p['estilo']]}.\n\n{REGLAS_FORMATO_TXT}"
        )
    if p["label"] == "SMISHING":
        return (
            f"{CONTEXTO_INVESTIGACION_TXT}\n\n"
            f"Genera {p['n']} SMS de estafa (smishing) distintos entre sí, dirigidos a usuarios en el Perú.\n"
            f"- Gatillo psicológico: {INSTRUCCIONES_TRIGGER[p['trigger']]}.\n"
            f"- Escenario: {p['escenario']}.\n"
            f"- Entidades a suplantar (reparte los mensajes entre ellas): {entidades}.\n"
            f"- Llamado a la acción: {CANALES_SMISHING[p['canal']]}.\n"
            f"- Estilo: {ESTILOS[p['estilo']]}.\n"
            f"- Si encaja, menciona {p['lugar']} u otro lugar del Perú.\n\n{REGLAS_FORMATO_TXT}"
        )
    if p["categoria"] == "conversacion":
        detalle = (f"conversación real entre peruanos: {p['escenario']}. Usa jerga peruana natural "
                   f"(sin exagerar), nombres peruanos y lugares como {p['lugar']}.")
        if entidades:
            detalle += f" Puede mencionar pagos entre personas por {entidades}."
    else:
        detalle = (f"{p['escenario']}, con el formato real que usaría la entidad"
                   + (f" ({entidades})" if entidades else "") + ".")
    enlace = ("Algunos pueden incluir SOLO el dominio oficial real de la entidad (p. ej. viabcp.com, "
              "interbank.pe, gob.pe), nunca un dominio inventado."
              if p["canal"] == "enlace_oficial" else "No incluyas enlaces.")
    return (
        f"Genera {p['n']} SMS LEGÍTIMOS (no fraudulentos) distintos entre sí, típicos del Perú.\n"
        f"- Tipo: {detalle}\n- {enlace}\n- Longitud: {ESTILOS_HAM[p['estilo']]}.\n"
        "- NO deben contener urgencia, amenazas, premios inesperados ni pedir datos, claves o códigos.\n\n"
        f"{REGLAS_FORMATO_TXT}"
    )


def generar_corpus_peru(client, modelo, plan, archivo_checkpoint="corpus_gemini_peru_v2.csv",
                        pausa_entre_llamadas=1.0):
    """Ejecuta el plan de generación con checkpoint por llamada (retomable)."""
    filas = []
    if os.path.exists(archivo_checkpoint):
        filas = pd.read_csv(archivo_checkpoint).to_dict("records")
        print(f"Checkpoint encontrado: {len(filas)} mensajes ya generados.")
    hechas = {f["plan_id"] for f in filas}
    vistos = {f["text"] for f in filas}

    pendientes = plan[~plan["plan_id"].isin(hechas)]
    print(f"Llamadas en el plan: {len(plan)}  |  pendientes: {len(pendientes)}")
    for k, (_, p) in enumerate(pendientes.iterrows(), start=1):
        tanda = llamar_gemini(client, modelo, _prompt_generar(p), TandaMensajes, temperatura=1.0)
        nuevos = 0
        for m in (tanda.mensajes if tanda else []):
            m = str(m).strip().strip('"')
            if m and not _es_rechazo(m) and m not in vistos:
                vistos.add(m)
                filas.append({
                    "text": m, "label": p["label"], "trigger": p["trigger"],
                    "source": "gemini_llm_peru", "categoria": p["categoria"],
                    "escenario": p["escenario"], "entidades_sugeridas": p["entidades_sugeridas"],
                    "canal": p["canal"], "estilo": p["estilo"], "plan_id": p["plan_id"],
                })
                nuevos += 1
        if k % 10 == 0 or k == len(pendientes):
            print(f"  llamada {k}/{len(pendientes)} ({p['plan_id']} {p['label']} / {p['categoria']}): +{nuevos}")
        pd.DataFrame(filas).to_csv(archivo_checkpoint, index=False)
        time.sleep(pausa_entre_llamadas)

    df = pd.DataFrame(filas)
    df["entidades"] = df["text"].map(lambda t: "|".join(detectar_entidades(t)))
    df["origen"] = "generado_llm"
    return df
