"""Traducción y adaptación del corpus internacional al contexto peruano (sección 2.4).

Código tomado del notebook v9, sin cambios.
"""
import hashlib
import os
import random
import re
import time
from collections import Counter

import pandas as pd
from pydantic import BaseModel

from .catalogo import (
    CATALOGO_ENTIDADES, GATILLOS, LUGARES_PERU, SECTOR_DE_ENTIDAD, detectar_entidades,
    normalizar,
)
from .gemini import CONTEXTO_INVESTIGACION_TXT, _es_rechazo, llamar_gemini, quitar_casi_duplicados


# Traducción y adaptación semántica del corpus internacional al contexto peruano.
# No es una traducción literal: Gemini conserva la intención y el mecanismo
# persuasivo de cada SMS real, y reemplaza marcas, montos y lugares extranjeros
# por equivalentes peruanos del catálogo (criterio de inclusión de la ficha).


class MensajeAdaptado(BaseModel):
    id: int
    inicio_original: str   # primeras palabras del original, copiadas: permite verificar el emparejamiento
    texto_adaptado: str
    descartar: bool
    gatillo: str


class TandaAdaptada(BaseModel):
    mensajes: list[MensajeAdaptado]


def _id_texto(texto):
    """Identificador estable de un SMS (sirve para retomar desde el checkpoint)."""
    return hashlib.md5(str(texto).encode("utf-8")).hexdigest()[:12]


def _catalogo_como_texto():
    return "\n".join(f"- {sector}: {', '.join(ents)}" for sector, ents in CATALOGO_ENTIDADES.items())


def seleccionar_candidatos(referencia, seed=42, largo_min=20, largo_max=400):
    """Ordena los SMS de referencia por prioridad de adaptación.
    SMISHING: primero fraude verificado (analysis_dataset_real y la etiqueta 'smishing'
    de Dataset_5971), luego spam de UCI/5971 (que Gemini puede descartar si es solo
    publicidad sin intención de fraude). HAM: solo UCI y Dataset_5971, porque el 'HAM'
    de analysis_dataset_real solo significa 'URL no detectada', no 'legítimo verificado'."""
    ref = referencia.copy()
    ref["largo"] = ref["text"].astype(str).str.len()
    ref = ref[(ref["largo"] >= largo_min) & (ref["largo"] <= largo_max)].reset_index(drop=True)
    # Dataset_5971 repite muchos SMS de UCI con pequeñas ediciones: se quitan los casi
    # duplicados ANTES de adaptar, para no adaptar dos veces el mismo mensaje.
    ref, n_casi = quitar_casi_duplicados(ref, umbral=0.90)
    print(f"Casi-duplicados eliminados del corpus de referencia antes de adaptar: {n_casi}")

    def prioridad_smishing(fila):
        if fila["source"] == "analysis_dataset_real":
            return 0
        if fila["source"] == "dataset_5971" and fila["label_original"] == "smishing":
            return 1
        return 2

    smi = ref[ref["label"] == "SMISHING"].copy()
    smi["prio"] = smi.apply(prioridad_smishing, axis=1)
    smi = smi.sample(frac=1, random_state=seed).sort_values("prio", kind="stable")

    ham = ref[(ref["label"] == "HAM") & (ref["source"].isin(["uci_sms_spam", "dataset_5971"]))]
    ham = ham.sample(frac=1, random_state=seed)
    return {"SMISHING": smi, "HAM": ham}


def _prompt_adaptar_smishing(lote, entidades_prioritarias):
    lineas = "\n".join(f"[{i}] {t}" for i, t in enumerate(lote))
    return (
        f"{CONTEXTO_INVESTIGACION_TXT}\n\n"
        "Tarea: TRADUCCIÓN Y ADAPTACIÓN SEMÁNTICA (no traducción literal) de SMS fraudulentos "
        "reales en inglés al contexto peruano. Para cada mensaje:\n"
        "1. Conserva la intención fraudulenta y el MISMO mecanismo persuasivo del original "
        "(si ofrece un premio, sigue ofreciendo un premio; si amenaza con un bloqueo, sigue amenazando).\n"
        "2. Reemplaza marcas, instituciones, monedas y lugares extranjeros por equivalentes peruanos "
        "reales (la entidad del catálogo más coherente con el mensaje). Montos en soles (S/) realistas.\n"
        "3. Redáctalo como lo escribiría un estafador que opera en el Perú: español peruano, "
        "peruanismos solo si encajan, que NO suene a traducción.\n"
        "4. Si el original trae un enlace, usa un enlace ficticio verosímil (dominio inventado); si trae "
        "un teléfono, uno peruano ficticio; si no trae ninguno, no lo agregues.\n"
        "5. Máximo ~200 caracteres.\n"
        "6. En inicio_original copia tal cual las 3 primeras palabras del mensaje original.\n"
        "7. Marca descartar=true SOLO si el original es publicidad genérica SIN intención de fraude "
        "financiero ni robo de datos (tonos de celular, chats de citas, contenido adulto, cadenas). "
        "Si descartas, deja texto_adaptado vacío.\n"
        "8. En gatillo pon el principal: URGENCIA_MIEDO, AUTORIDAD, RECOMPENSA o CONFIANZA_ENCOMIENDA.\n\n"
        f"Para variar, prioriza estas entidades (las menos usadas hasta ahora) cuando sean coherentes: "
        f"{', '.join(entidades_prioritarias)}.\n"
        f"Catálogo de entidades peruanas por sector:\n{_catalogo_como_texto()}\n\n"
        f"Mensajes originales:\n{lineas}\n\n"
        "Devuelve exactamente un elemento por cada id recibido."
    )


def _prompt_adaptar_ham(lote, lugares):
    lineas = "\n".join(f"[{i}] {t}" for i, t in enumerate(lote))
    return (
        "Tarea: TRADUCCIÓN Y ADAPTACIÓN SEMÁNTICA (no traducción literal) de SMS LEGÍTIMOS reales "
        "en inglés al español peruano. Para cada mensaje:\n"
        "1. Conserva el sentido y el tono: una conversación casual sigue siendo casual; una "
        "notificación sigue siendo una notificación legítima.\n"
        "2. Que suene escrito por un peruano de verdad: modismos donde encajen (pe, ya, causa, oe, "
        "chamba, jato, al toque, bacán), nombres de personas peruanos y lugares del Perú "
        f"(por ejemplo {', '.join(lugares)}) en lugar de los extranjeros.\n"
        "3. Si el original usa abreviaturas de chat (u, r, 2morrow), usa las peruanas equivalentes "
        "(q, xq, tmb, ntp, tqm).\n"
        "4. NO agregues urgencia, amenazas, premios, pedidos de datos ni enlaces que el original no tenga.\n"
        "5. Máximo ~200 caracteres.\n"
        "6. En inicio_original copia tal cual las 3 primeras palabras del mensaje original.\n"
        "7. descartar=true solo si el mensaje no tiene sentido al adaptarlo (p. ej. es solo un código "
        "o está incompleto). En gatillo pon siempre NINGUNO.\n\n"
        f"Mensajes originales:\n{lineas}\n\n"
        "Devuelve exactamente un elemento por cada id recibido."
    )


def _emparejamiento_valido(original, inicio_devuelto):
    """Comprueba que el id devuelto corresponde al SMS original correcto: la primera
    palabra que Gemini dice haber copiado debe estar entre las 5 primeras del original."""
    palabras_orig = re.findall(r"[a-z0-9]+", normalizar(original))[:5]
    palabras_dev = re.findall(r"[a-z0-9]+", normalizar(inicio_devuelto))
    return bool(palabras_dev) and palabras_dev[0] in palabras_orig


def adaptar_corpus_internacional(client, modelo, referencia, n_por_clase=1000, tam_lote=15,
                                 archivo_checkpoint="dataset_traducido_adaptado.csv",
                                 pausa_entre_llamadas=1.0, seed=42, solo_checkpoint=False):
    """Traduce y adapta hasta n_por_clase SMS de cada clase. Guarda TODO en el
    checkpoint (incluidos los descartados y el texto original en inglés), para que
    el proceso sea trazable y se pueda retomar si se corta."""
    rng = random.Random(seed)
    candidatos = seleccionar_candidatos(referencia, seed=seed)

    filas = []
    if os.path.exists(archivo_checkpoint):
        filas = pd.read_csv(archivo_checkpoint).to_dict("records")
        print(f"Checkpoint encontrado: {len(filas)} SMS ya procesados. Se continúa desde ahí.")
        ham_hechos = sum(1 for f in filas if f["label"] == "HAM" and not f["descartado"])
        if solo_checkpoint and ham_hechos >= n_por_clase:
            # Modo reproducción: el checkpoint ya está completo, así que se usa tal cual,
            # sin nuevas llamadas a Gemini, y el dataset sale idéntico al de la corrida original.
            # (Si la corrida se cortó a medias, el checkpoint está incompleto y se continúa).
            print("Modo reproducción: no se llama a Gemini; se usa solo el checkpoint.")
            return pd.DataFrame(filas)
    procesados = {f["id_ref"] for f in filas}

    uso_entidades = Counter({ent: 0 for ent in SECTOR_DE_ENTIDAD})
    for f in filas:
        if isinstance(f.get("entidades"), str) and f["entidades"]:
            uso_entidades.update(f["entidades"].split("|"))

    for clase in ["SMISHING", "HAM"]:
        pendientes = candidatos[clase][~candidatos[clase]["text"].map(_id_texto).isin(procesados)]
        hechos = sum(1 for f in filas if f["label"] == clase and not f["descartado"])
        print(f"\nAdaptando {clase}: objetivo {n_por_clase}, ya adaptados {hechos}, "
              f"candidatos restantes {len(pendientes)}")
        pos = 0
        while hechos < n_por_clase and pos < len(pendientes):
            lote_df = pendientes.iloc[pos:pos + tam_lote]
            pos += tam_lote
            lote = lote_df["text"].astype(str).tolist()

            if clase == "SMISHING":
                menos_usadas = [e for e, _ in sorted(uso_entidades.items(), key=lambda kv: (kv[1], rng.random()))][:12]
                prompt = _prompt_adaptar_smishing(lote, menos_usadas)
            else:
                prompt = _prompt_adaptar_ham(lote, rng.sample(LUGARES_PERU, 4))

            tanda = llamar_gemini(client, modelo, prompt, TandaAdaptada, temperatura=0.7)
            if tanda is None:
                continue
            por_id = {m.id: m for m in tanda.mensajes
                      if 0 <= m.id < len(lote) and _emparejamiento_valido(lote[m.id], m.inicio_original)}

            nuevos = 0
            for i, (_, fila) in enumerate(lote_df.iterrows()):
                m = por_id.get(i)
                if m is None:
                    continue  # id omitido o mal emparejado: se reintentará en otra corrida
                texto = (m.texto_adaptado or "").strip().strip('"')
                descartado = bool(m.descartar) or not texto or _es_rechazo(texto)
                gatillo = m.gatillo if (clase == "SMISHING" and m.gatillo in GATILLOS) else None
                entidades = detectar_entidades(texto) if not descartado else []
                filas.append({
                    "id_ref": _id_texto(fila["text"]),
                    "text_original": fila["text"],
                    "source_original": fila["source"],
                    "label": clase,
                    "text": "" if descartado else texto,
                    "trigger": gatillo,
                    "descartado": descartado,
                    "entidades": "|".join(entidades),
                })
                if not descartado:
                    hechos += 1
                    nuevos += 1
                    uso_entidades.update(entidades)
            print(f"  +{nuevos} adaptados ({hechos}/{n_por_clase})")
            pd.DataFrame(filas).to_csv(archivo_checkpoint, index=False)
            time.sleep(pausa_entre_llamadas)

        if hechos < n_por_clase:
            print(f"  -> Aviso: solo se alcanzaron {hechos} SMS {clase} adaptados "
                  f"(se agotaron los candidatos válidos).")

    df_proc = pd.DataFrame(filas)
    df_proc.to_csv(archivo_checkpoint, index=False)
    return df_proc
