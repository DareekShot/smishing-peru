"""Pipelines, validación cruzada por grupos, comparación de modelos, Grid Search y ensambles
(sección 5).

Código tomado del notebook v9, sin cambios.
"""
import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from .caracteristicas import StylisticFeatureExtractor
from .preprocesamiento import PreprocesadorTexto


# Suite de entrenamiento y comparación de modelos de clasificación
# v5: cada modelo es un PIPELINE completo (SMS en texto plano -> predicción). Así, en la
# validación cruzada el TF-IDF y el escalado se reajustan dentro de cada pliegue, sin ver
# los SMS de validación (en la v4 se ajustaban una vez con todo train y luego se hacía la
# validación cruzada sobre esas matrices: una fuga leve que inflaba un poco el F1 de CV).
# v7: los SMS generados con Gemini (data augmentation) se usan solo para entrenar: el test y
# los pliegues de validación tienen solo SMS adaptados de datasets reales.

RANDOM_STATE = 42


def crear_pipeline(modelo, usar_logica_difusa=False, quitar_stopwords=False, robusto=False):
    """SMS en texto plano -> [TF-IDF del texto preprocesado | 7 variables escaladas] -> modelo.
    Todo lo que aprende de los datos (vocabulario e IDF del TF-IDF, escala del MaxAbsScaler
    y el modelo) se ajusta solo con los SMS que recibe en fit()."""
    variables = FeatureUnion([
        ("tfidf", Pipeline([
            ("preprocesado", PreprocesadorTexto(quitar_stopwords=quitar_stopwords, robusto=robusto)),
            ("vectorizador", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=2,
                                             token_pattern=r"\S+")),  # el texto ya viene tokenizado
        ])),
        ("estilo", Pipeline([
            ("extractor", StylisticFeatureExtractor(usar_logica_difusa=usar_logica_difusa, robusto=robusto)),
            ("escalado", MaxAbsScaler()),
        ])),
    ])
    return Pipeline([("variables", variables), ("modelo", modelo)])


# Los grupos de los SMS generados con Gemini empiezan con "G_" y los de los adaptados de
# datasets reales con "A_" (sección 3, construir_dataset_final).
PREFIJO_SINTETICO = "G_"


def es_sintetico(grupos):
    return pd.Series(np.asarray(grupos)).astype(str).str.startswith(PREFIJO_SINTETICO).values


class ValidacionSoloReales:
    """Validación cruzada en la que los SMS generados (data augmentation) SOLO se usan para
    entrenar: los pliegues se arman con los SMS adaptados de datasets reales, por grupos y
    estratificados por clase, y todos los generados se suman siempre a la parte de
    entrenamiento. Así ningún SMS sintético se usa para validar ni para elegir el modelo."""

    def __init__(self, n_splits=5, seed=RANDOM_STATE):
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y=None, groups=None):
        sintetico = es_sintetico(groups)
        reales, generados = np.where(~sintetico)[0], np.where(sintetico)[0]
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for i_ent, i_val in sgkf.split(reales, np.asarray(y)[reales], np.asarray(groups)[reales]):
            yield np.sort(np.concatenate([reales[i_ent], generados])), reales[i_val]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class CVPorGrupos:
    """StratifiedGroupKFold que recupera el grupo de cada SMS a partir de su texto. Se usa en
    la calibración de probabilidades y en el Stacking (sección 5.2): esos modelos hacen su
    propia validación cruzada interna y no reciben 'groups', así que sin esto podrían separar
    a dos SMS "hermanos" (misma llamada a Gemini) entre sus pliegues internos. Esos pliegues
    son parte del ENTRENAMIENTO (ajustan la calibración y el meta-modelo), no de la evaluación,
    así que ahí los SMS generados sí pueden quedar en las dos partes."""

    def __init__(self, grupo_de_texto, n_splits=3, seed=RANDOM_STATE):
        self.grupo_de_texto = grupo_de_texto
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y=None, groups=None):
        # Un SMS que no está en el diccionario (p. ej. otro dataset) es su propio grupo.
        grupos = np.array([self.grupo_de_texto.get(t, f"sin_grupo_{i}") for i, t in enumerate(X)])
        return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                    random_state=self.seed).split(X, y, grupos)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _forzar_un_hilo(modelo, cambios=None):
    """Pone n_jobs=1 en el modelo y en todos sus sub-modelos ya entrenados (pasos del
    pipeline, ensambles, calibración). Devuelve la lista de cambios para restaurarlos."""
    cambios = [] if cambios is None else cambios
    if hasattr(modelo, "n_jobs") and hasattr(modelo, "set_params"):
        cambios.append((modelo, modelo.n_jobs))
        modelo.set_params(n_jobs=1)
    hijos = [paso for _, paso in getattr(modelo, "steps", [])]
    hijos += [t for _, t in getattr(modelo, "transformer_list", [])]
    if not hasattr(modelo, "tree_"):
        hijos += list(getattr(modelo, "estimators_", []) or [])
    hijos += [getattr(c, "estimator", None) for c in getattr(modelo, "calibrated_classifiers_", [])]
    hijos.append(getattr(modelo, "final_estimator_", None))
    for hijo in hijos:
        if hijo is not None and hijo is not modelo and not isinstance(hijo, str) and not hasattr(hijo, "tree_"):
            _forzar_un_hilo(hijo, cambios)
    return cambios


def medir_latencia_y_tamano(pipeline, textos, n_muestra=200, seed=RANDOM_STATE):
    """Latencia de inferencia EXTREMO A EXTREMO por SMS (ms): preprocesamiento + TF-IDF +
    extractor + escalado + predicción, SMS por SMS, sobre una muestra fija del test y en UN
    SOLO HILO (como correría en un celular). Tamaño: el pipeline completo serializado (KB,
    proxy de memoria en Edge AI)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(textos), size=min(n_muestra, len(textos)), replace=False)
    cambios = _forzar_un_hilo(pipeline)
    try:
        pipeline.predict([textos[idx[0]]])  # calentamiento (la primera llamada siempre es más lenta)
        t0 = time.perf_counter()
        for i in idx:
            pipeline.predict([textos[i]])
        latencia_ms = (time.perf_counter() - t0) / len(idx) * 1000
    finally:
        for obj, valor in cambios:
            obj.set_params(n_jobs=valor)
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return latencia_ms, buffer.tell() / 1024


def particion_por_grupos(df, n_folds_test=5, seed=RANDOM_STATE):
    """Train/test 80/20 hecha SOLO con los SMS adaptados de datasets reales, por grupos y
    estratificada por clase y dataset de origen. Los SMS generados con Gemini (data
    augmentation) van todos a train: nunca se usan para evaluar."""
    grupos = df["grupo"].values if "grupo" in df.columns else np.arange(len(df))
    origen = df["origen"].values if "origen" in df.columns else np.array(["n/d"] * len(df))
    fuente = df["source_original"].values if "source_original" in df.columns else np.array(["n/d"] * len(df))
    sintetico = es_sintetico(grupos)
    reales = np.where(~sintetico)[0]
    estratos = pd.Series(df["label"].values[reales]) + "|" + pd.Series(fuente[reales]).astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_folds_test, shuffle=True, random_state=seed)
    i_ent, i_pru = next(sgkf.split(reales, estratos, groups=grupos[reales]))
    idx_train = np.sort(np.concatenate([reales[i_ent], np.where(sintetico)[0]]))
    idx_test = reales[i_pru]
    return idx_train, idx_test, grupos, origen


def modelos_base(seed=RANDOM_STATE):
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                 class_weight="balanced", n_jobs=-1),
        # probability=True solo sirve para el "riesgo %" y la curva ROC: la predicción
        # (predict) de SVC usa la función de decisión, no esas probabilidades.
        "SVM (kernel lineal)": SVC(kernel="linear", probability=True, class_weight="balanced",
                                    random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.15,
                                  eval_metric="logloss", random_state=seed, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "Regresión Logística": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=seed),
    }


def entrenar_y_comparar(df, cv_folds=5, seed=RANDOM_STATE, usar_logica_difusa=True, quitar_stopwords=True):
    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # HAM=0, SMISHING=1 (orden alfabético)
    X_text = df["text"].values
    idx_train, idx_test, grupos, origen = particion_por_grupos(df, seed=seed)
    X_train_txt, X_test_txt = X_text[idx_train], X_text[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    grupos_train, origen_test = grupos[idx_train], origen[idx_test]

    resultados = []
    modelos_entrenados = {}
    # Validación cruzada solo sobre train, por grupos y validando solo con SMS reales.
    cv = ValidacionSoloReales(n_splits=cv_folds, seed=seed)

    for nombre, clasificador in modelos_base(seed).items():
        pipeline = crear_pipeline(clasificador, usar_logica_difusa, quitar_stopwords)
        # CV sin fuga: en cada pliegue se reajusta el pipeline COMPLETO (TF-IDF, escalado y modelo).
        cv_f1 = cross_val_score(pipeline, X_train_txt, y_train, groups=grupos_train, cv=cv,
                                scoring="f1", n_jobs=-1)

        t0 = time.perf_counter()
        pipeline.fit(X_train_txt, y_train)
        t_entrenamiento = time.perf_counter() - t0

        latencia_ms, tamano_kb = medir_latencia_y_tamano(pipeline, X_test_txt)
        y_pred = pipeline.predict(X_test_txt)
        y_proba = pipeline.predict_proba(X_test_txt)[:, 1]

        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "F1 CV (5-fold, media)": cv_f1.mean(),
            "F1 CV (desv. estandar)": cv_f1.std(),
            "Latencia (ms/SMS)": latencia_ms,
            "Tamaño modelo (KB)": tamano_kb,
            "Tiempo entrenamiento (s)": t_entrenamiento,
        })
        modelos_entrenados[nombre] = pipeline

    # Se ordena por el F1 de validación cruzada (solo train): el test NO se usa para elegir.
    df_resultados = pd.DataFrame(resultados).sort_values("F1 CV (5-fold, media)", ascending=False).reset_index(drop=True)

    contexto = {
        "label_encoder": le,
        "X_train_txt": X_train_txt, "y_train": y_train, "grupos_train": grupos_train, "cv": cv,
        "X_test_txt": X_test_txt, "y_test": y_test, "origen_test": origen_test,
        "idx_train": idx_train, "idx_test": idx_test,  # posiciones en df: permiten ver qué SMS hay en cada partición
        "grupo_de_texto": dict(zip(X_text, grupos)),
        "modelos_entrenados": modelos_entrenados,
        "usar_logica_difusa": usar_logica_difusa, "quitar_stopwords": quitar_stopwords,
    }
    return df_resultados, contexto


def fijar_parametro(modelo, nombre, valor):
    """Copia SIN entrenar del modelo con todos sus parámetros 'nombre' cambiados a 'valor'.
    Sirve también para ensambles, que contienen varios pipelines (p. ej. robusto=True)."""
    from sklearn.base import clone
    copia = clone(modelo)
    claves = [k for k in copia.get_params(deep=True) if k == nombre or k.endswith("__" + nombre)]
    copia.set_params(**{k: valor for k in claves})
    return copia


# Ajuste de hiperparámetros (Grid Search) de los 4 modelos de la ficha y ensambles (Voting + Stacking)
# v5: la búsqueda se hace sobre el pipeline completo (el TF-IDF se reajusta en cada pliegue), y
# la calibración y el Stacking usan pliegues internos por grupos (CVPorGrupos).


def afinar_hiperparametros_y_ensamblar(ctx, cv_folds_grid=3, seed=RANDOM_STATE):
    X_train, y_train, grupos = ctx["X_train_txt"], ctx["y_train"], ctx["grupos_train"]
    X_test, y_test = ctx["X_test_txt"], ctx["y_test"]
    cv_grid = ValidacionSoloReales(n_splits=cv_folds_grid, seed=seed)  # valida solo con SMS reales
    cv_interno = CVPorGrupos(ctx["grupo_de_texto"], n_splits=cv_folds_grid, seed=seed)

    def pipeline_de(modelo):
        return crear_pipeline(modelo, ctx["usar_logica_difusa"], ctx["quitar_stopwords"])

    def buscar(nombre, estimador, grilla):
        grilla = {f"modelo__{k}": v for k, v in grilla.items()}
        print(f"Grid Search: {nombre} ({np.prod([len(v) for v in grilla.values()])} combinaciones)...")
        gs = GridSearchCV(pipeline_de(estimador), grilla, scoring="f1", cv=cv_grid, n_jobs=-1)
        gs.fit(X_train, y_train, groups=grupos)
        mejores = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}
        print(f"  mejores parámetros: {mejores}  (F1 CV: {gs.best_score_:.4f})")
        return gs

    # SVM: se busca C con LinearSVC (rápido) y solo al final se calibra el mejor para
    # obtener probabilidades: SVC(probability=True) recalibra en cada fit y volvería
    # la búsqueda extremadamente lenta.
    gs_svm = buscar("SVM (kernel lineal)", LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000),
                    {"C": [0.1, 1, 5, 10, 50]})
    mejor_c = gs_svm.best_params_["modelo__C"]
    svm_afinado = CalibratedClassifierCV(
        pipeline_de(LinearSVC(C=mejor_c, class_weight="balanced", random_state=seed, max_iter=5000)),
        cv=cv_interno)
    svm_afinado.fit(X_train, y_train)

    rf_afinado = buscar("Random Forest", RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
                        {"n_estimators": [100, 200], "max_depth": [None, 30], "min_samples_leaf": [1, 2]}).best_estimator_
    xgb_afinado = buscar("XGBoost", XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1),
                         {"n_estimators": [150, 250], "max_depth": [4, 6], "learning_rate": [0.1, 0.2]}).best_estimator_
    ada_afinado = buscar("AdaBoost", AdaBoostClassifier(random_state=seed),
                         {"n_estimators": [100, 200], "learning_rate": [0.5, 1.0]}).best_estimator_

    print("\nConstruyendo ensamble (soft voting) SVM + RF afinados...")
    voting = VotingClassifier(estimators=[("svm", svm_afinado), ("rf", rf_afinado)], voting="soft")
    voting.fit(X_train, y_train)

    # Stacking: copias sin entrenar (clone) de los modelos afinados, con una Regresión
    # Logística como meta-modelo que aprende a combinarlos (en vez de solo promediarlos).
    print("Construyendo ensamble por Stacking (SVM + RF + XGBoost -> Regresión Logística)...")
    stacking = StackingClassifier(
        estimators=[("svm", clone(svm_afinado)), ("rf", clone(rf_afinado)), ("xgb", clone(xgb_afinado))],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        cv=cv_interno, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    modelos_nuevos = {
        "SVM afinado (Grid Search)": svm_afinado,
        "Random Forest afinado (Grid Search)": rf_afinado,
        "XGBoost afinado (Grid Search)": xgb_afinado,
        "AdaBoost afinado (Grid Search)": ada_afinado,
        "Ensamble SVM+RF (soft voting)": voting,
        "Ensamble Stacking (SVM+RF+XGB -> LogReg)": stacking,
    }
    print("\nF1 de validación cruzada (5 pliegues por grupos, validando solo con SMS reales) de cada candidato...")
    filas = []
    for nombre, modelo in modelos_nuevos.items():
        cv_f1 = cross_val_score(clone(modelo), X_train, y_train, groups=grupos, cv=ctx["cv"], scoring="f1", n_jobs=-1)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        filas.append({
            "Modelo": nombre,
            "F1 CV (5-fold, media)": cv_f1.mean(),
            "F1 CV (desv. estandar)": cv_f1.std(),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
    df_resultados = pd.DataFrame(filas).sort_values("F1 CV (5-fold, media)", ascending=False).reset_index(drop=True)
    return df_resultados, modelos_nuevos
