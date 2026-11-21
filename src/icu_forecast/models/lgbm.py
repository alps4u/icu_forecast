"""LightGBM fit/plot helpers, shared by models.train (the 5-fold CV entry point).

Early stopping uses a fraction of the training patients (config.yaml: models.lgbm.early_stopping_val_frac).
"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def fit_lgbm(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: dict,
             metric: str = "average_precision") -> lgb.LGBMClassifier:
    settings = cfg["models"]["lgbm"]
    params = {k: v for k, v in settings.items() if k not in ("early_stopping_rounds", "early_stopping_val_frac")}
    params.update(objective="binary", metric=metric, n_jobs=-1, verbose=-1)

    fit_idx, es_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=settings["early_stopping_val_frac"], random_state=cfg["seed"],
    ).split(X, y, groups))
    model = lgb.LGBMClassifier(**params, random_state=cfg["seed"])
    model.fit(
        X.iloc[fit_idx], y.iloc[fit_idx],
        eval_set=[(X.iloc[es_idx], y.iloc[es_idx])],
        callbacks=[lgb.early_stopping(settings["early_stopping_rounds"], verbose=False), lgb.log_evaluation(250)],
    )
    return model


def plot_importance(model: lgb.LGBMClassifier, path, top: int = 20) -> None:
    gain = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    gain = (gain / gain.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(gain.index, gain.values * 100, color="#27ae60")
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("LightGBM, 24 h model: top features")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
