"""Intervalos de confianza por bootstrap, test de McNemar y corrección de Holm (sección 5.3.1).

Código tomado del notebook v9, sin cambios.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .modelos import RANDOM_STATE


# --- 5.3.1 ¿Las diferencias son significativas? Intervalos de confianza y test de McNemar ---
# Siguiendo a Rasol Al Saidat et al. (2026): (a) intervalo de confianza del 95 % por bootstrap
# para las métricas del modelo final, y (b) test de McNemar entre el modelo final y cada
# candidato, sobre las MISMAS predicciones del test, con corrección de Holm. Nada de esto se
# usa para elegir el modelo (eso ya se decidió con la validación cruzada): solo dice cuánta
# confianza merecen las diferencias que se ven en el test.


def _metricas(y_t, y_p):
    return {"Accuracy": accuracy_score(y_t, y_p), "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0), "F1-Score": f1_score(y_t, y_p, zero_division=0)}


def intervalo_bootstrap(y_t, y_p, grupos=None, n_boot=2000, seed=RANDOM_STATE):
    """IC del 95 % por percentiles. Con 'grupos' se remuestrean grupos completos (bootstrap
    por conglomerados): los SMS de un mismo grupo no son independientes, así que este
    intervalo es el más honesto."""
    rng = np.random.RandomState(seed)
    if grupos is None:
        unidades = [np.array([i]) for i in range(len(y_t))]
    else:
        _, inversa = np.unique(grupos, return_inverse=True)
        unidades = [np.where(inversa == g)[0] for g in range(inversa.max() + 1)]
    muestras = []
    for _ in range(n_boot):
        idx = np.concatenate([unidades[k] for k in rng.randint(0, len(unidades), len(unidades))])
        muestras.append(_metricas(y_t[idx], y_p[idx]))
    muestras = pd.DataFrame(muestras)
    return muestras.quantile(0.025), muestras.quantile(0.975)


def mcnemar_exacto(y_t, pred_a, pred_b):
    """b = SMS que solo A acierta; c = SMS que solo B acierta. Test binomial exacto sobre b y c."""
    b = int(((pred_a == y_t) & (pred_b != y_t)).sum())
    c = int(((pred_a != y_t) & (pred_b == y_t)).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def holm(p_valores):
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(p_valores, dtype=float)
    orden = np.argsort(p)
    ajustados = np.empty_like(p)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (len(p) - rango) * p[i]))
        ajustados[i] = acumulado
    return ajustados
