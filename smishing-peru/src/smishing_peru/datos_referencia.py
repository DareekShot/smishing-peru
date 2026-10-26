"""Carga de los tres datasets de referencia en inglés (sección 1 del notebook).

Código tomado del notebook v9, sin cambios.
"""
import numpy as np
import pandas as pd


def cargar_uci_sms(path="SMSSpamCollection.csv"):
    """SMS Spam Collection (UCI). Formato: label + tab + texto."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["label_original"] = df["label"]
    df["label"] = df["label"].map({"ham": "HAM", "spam": "SMISHING"})
    df["source"] = "uci_sms_spam"
    return df[["text", "label", "label_original", "source"]]


def cargar_dataset_5971(path="Dataset_5971.csv"):
    """Dataset_5971: variante etiquetada de SMS spam/smishing en inglés."""
    df = pd.read_csv(path)
    df["label_norm"] = df["LABEL"].str.lower()
    df["label_original"] = df["label_norm"]
    df["label"] = df["label_norm"].map({"ham": "HAM", "spam": "SMISHING", "smishing": "SMISHING"})
    df = df.rename(columns={"TEXT": "text"})
    df["source"] = "dataset_5971"
    return df[["text", "label", "label_original", "source"]]


def cargar_analysis_dataset(path="analysisdataset.csv"):
    """
    Dataset real de smishing con metadatos de analisis de URL/dominio
    (score de motores de deteccion estilo VirusTotal: Phishing, Malicious, etc.)
    Se deriva la etiqueta binaria: si Phishing>0 o Malicious>0 => SMISHING.
    """
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["MainText"])
    riesgo = (df["Phishing"].fillna(0) > 0) | (df["Malicious"].fillna(0) > 0)
    df["label"] = np.where(riesgo, "SMISHING", "HAM")
    df["label_original"] = np.where(riesgo, "detectado_por_motores", "sin_deteccion")
    df = df.rename(columns={"MainText": "text"})
    df["source"] = "analysis_dataset_real"
    return df[["text", "label", "label_original", "source"]]
