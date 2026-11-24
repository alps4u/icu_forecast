"""Discrimination and watchlist metrics for the LightGBM model."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10) -> dict:
    """ROC-AUC, PR-AUC, Brier score, and precision/recall among the top `top_frac` highest-risk snapshots."""
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(top_frac * len(y)))
    flagged = y[np.argsort(-p, kind="stable")[:k]]
    return {
        "n": len(y),
        "prevalence": y.mean(),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision_top10": flagged.sum() / k,
        "recall_top10": flagged.sum() / y.sum(),
        "brier": brier_score_loss(y, p),
    }
