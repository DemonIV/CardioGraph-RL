# AGENT: eval_agent
"""Standart + özgün metrikler."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def compute_all_metrics(y_true, y_pred, y_prob):
    """F1 macro/weighted, AUC-ROC, confusion matrix.

    Parameters
    ----------
    y_true : (N,) int
    y_pred : (N,) int
    y_prob : (N, C) float — softmax olasılıkları
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    return {
        "f1_macro":         float(f1_macro),
        "f1_weighted":      float(f1_weighted),
        "auc_roc":          float(auc),
        "confusion_matrix": cm,
    }


def faithfulness_at_k(attention_scores, pathological_labels, k=3):
    """
    Özgün metrik: top-K attention atımlarının patolojik olma oranı.
    faithfulness@K = |top_k ∩ patolojik| / K

    Parameters
    ----------
    attention_scores    : (N,) float — düğüm başına birleştirilmiş attention
    pathological_labels : (N,) int/bool — 1 ise düğüm patolojik
    k                   : int

    Returns
    -------
    float in [0, 1]
    """
    scores = np.asarray(attention_scores, dtype=float)
    labels = np.asarray(pathological_labels, dtype=int)

    if len(scores) == 0:
        return 0.0

    k_eff     = min(k, len(scores))
    top_k_idx = np.argsort(scores)[-k_eff:]
    return float(labels[top_k_idx].sum()) / k_eff


def expected_calibration_error(y_prob, y_true, n_bins=10):
    """ECE kalibrasyon skoru.

    Parameters
    ----------
    y_prob  : (N,) float — tahmin güven skoru [0, 1]
    y_true  : (N,) int   — ikili doğruluk etiketi (1=doğru, 0=yanlış)
              veya ikili sınıf etiketi
    n_bins  : int

    Returns
    -------
    float — küçük = iyi kalibre
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_prob)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob > lo) & (y_prob <= hi)
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc  = float(y_true[mask].mean())
        ece     += (mask.sum() / n) * abs(avg_conf - avg_acc)

    return float(ece)
