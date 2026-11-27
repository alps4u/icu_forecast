"""Discrimination and watchlist metrics for the LightGBM model."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def precision_at_recall(y: np.ndarray, p: np.ndarray, recall_target: float = 0.30) -> float:
    """Best precision achievable among thresholds whose recall is at least `recall_target`.

    This is the number the M4 success criterion (recall >= 0.30, precision >= 0.10) is actually
    judged against -- the top-10% watchlist's precision is capped near the positive rate (~0.9%
    prevalence => <= ~0.10 at best) almost by construction, so it cannot by itself say whether a
    model clears the criterion.
    """
    y, p = np.asarray(y), np.asarray(p)
    precision, recall, _ = precision_recall_curve(y, p)
    meets_target = recall >= recall_target
    if not meets_target.any():
        return float("nan")
    return float(precision[meets_target].max())


def evaluate(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10, recall_target: float = 0.30) -> dict:
    """ROC-AUC, PR-AUC, Brier score, precision/recall among the top `top_frac` highest-risk
    snapshots, and precision at recall >= `recall_target` (the success-criterion operating point).
    """
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
        f"precision_at_recall{int(recall_target * 100)}": precision_at_recall(y, p, recall_target),
        "brier": brier_score_loss(y, p),
    }
