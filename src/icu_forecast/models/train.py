"""5-fold cross-validation for the LightGBM baseline, grouped and stratified by patient.

Folds are built once on the combined train+validation patient pool, stratified by
whether the patient ever had an unplanned ward-to-ICU transfer (the same variable
split.py stratifies the fixed split on), so every snapshot from one patient stays in
a single fold. The test split is never read here.

LightGBM is trained with `metric="auc"` (passed to fit_lgbm), so early stopping itself
tracks ROC-AUC on the held-out fraction, and ROC-AUC is the primary metric reported
here too. Chosen because the positive rate is well under 2% (the negative class
outnumbers the positive one by roughly two orders of magnitude) -- ROC-AUC is
computed within each class separately, so it is not distorted by the imbalance the
way accuracy would be. Secondary metrics (PR-AUC, recall/precision at the top 10%,
Brier) are reported too, since ROC-AUC alone can look deceptively strong while
precision in the clinically relevant top decile stays low -- see reports/data_pipeline.md
and the M4 report for that discussion. This script only trains LightGBM -- no
reference, logistic regression, or k-NN.

Categorical levels and the tree-model's missing-value handling are refit inside
every fold, on that fold's own training patients only -- nothing is shared across
folds except the fold assignment itself.

Run: python -m icu_forecast.models.train
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from icu_forecast.config import ROOT, load_config
from icu_forecast.models.evaluate import evaluate
from icu_forecast.models.lgbm import fit_lgbm
from icu_forecast.prepare import CATEGORICAL, LABELS, feature_columns, load_features


def plot_accuracy_vs_roc_auc(fold_results: dict[str, pd.DataFrame], path) -> None:
    """Accuracy per fold against the trivial 'always predict negative' floor, next to ROC-AUC for contrast.

    Accuracy at a 0.5 threshold is not a metric this project reports as a measure of model quality
    (see reports/data_pipeline.md and the M4 report) -- at <2% prevalence, 'always negative' already
    scores ~98-99%, so accuracy barely moves regardless of how good the model is. This figure exists
    to show that explicitly, not to recommend using accuracy for anything.
    """
    fig, axes = plt.subplots(1, len(fold_results), figsize=(6 * len(fold_results), 4.5), sharey=True)
    if len(fold_results) == 1:
        axes = [axes]
    for ax, (label, fm) in zip(axes, fold_results.items()):
        x = fm.fold
        ax.bar(x, fm.accuracy, color="#999999", label="Accuracy (0.5 threshold)")
        ax.axhline(fm.always_negative_accuracy.mean(), color="#999999", linestyle="--", linewidth=1,
                   label="'Always negative' accuracy floor")
        ax.plot(x, fm.roc_auc, color="#27ae60", marker="o", linewidth=2, label="ROC-AUC")
        ax.set_xlabel("Fold")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.03)
        ax.set_title(f"{label.replace('y_', '').replace('h', ' h')} horizon")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("Accuracy looks high and flat; ROC-AUC is where the real signal shows up")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def patient_fold_assignment(cfg: dict, n_splits: int) -> pd.DataFrame:
    """One row per patient in the train+val pool, with its fold id and 'ever transferred' flag."""
    cohort = pd.read_parquet(cfg["paths"]["processed"] / "cohort.parquet")
    split = pd.read_csv(cfg["paths"]["processed"] / "split.csv")
    adm = cohort.merge(split, on="subject_id", how="inner")
    patients = adm.groupby("subject_id").agg(
        ever_icu=("first_icu_time", lambda s: s.notna().any()),
        split=("split", "first"),
    )
    pool = patients[patients.split.isin(["train", "val"])].reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    pool["fold"] = -1
    for i, (_, fold_idx) in enumerate(skf.split(pool, pool.ever_icu)):
        pool.loc[fold_idx, "fold"] = i
    return pool[["subject_id", "fold"]]


def to_tree_frame_local(df: pd.DataFrame, numeric: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Same as prepare.to_tree_frame, but with categories computed fresh for this fold's training data."""
    X = df[numeric + list(categories)].copy()
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X


def main() -> None:
    cfg = load_config()
    n_splits = cfg["models"]["cv_folds"]
    df = load_features(cfg)
    numeric, _ = feature_columns(df)

    fold_map = patient_fold_assignment(cfg, n_splits)
    pool = df.merge(fold_map, on="subject_id", how="inner")
    print(f"CV pool: {pool.subject_id.nunique():,} patients, {len(pool):,} snapshots, {n_splits} folds")

    rows = []
    fold_results = {}
    for label in LABELS:
        print(f"\n=== {label} ===")
        fold_metrics = []
        for k in range(n_splits):
            train_fold = pool[pool.fold != k]
            val_fold = pool[pool.fold == k]

            categories = {c: sorted(train_fold[c].dropna().unique().tolist()) for c in CATEGORICAL}
            X_train = to_tree_frame_local(train_fold, numeric, categories)
            X_val = to_tree_frame_local(val_fold, numeric, categories)

            model = fit_lgbm(X_train, train_fold[label], train_fold.subject_id, cfg, metric="auc")
            p_val = model.predict_proba(X_val)[:, 1]
            y_val = val_fold[label].to_numpy()
            m = evaluate(y_val, p_val)
            m["fold"] = k
            # accuracy at a 0.5 threshold -- not used as a criterion anywhere in this project (see
            # reports/data_pipeline.md); computed here only so plot_accuracy_vs_roc_auc can show why.
            m["accuracy"] = ((p_val >= 0.5) == y_val).mean()
            m["always_negative_accuracy"] = 1 - m["prevalence"]
            fold_metrics.append(m)
            print(f"fold {k}: n={m['n']:,} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
                  f"recall_top10={m['recall_top10']:.4f} precision_top10={m['precision_top10']:.4f} "
                  f"accuracy={m['accuracy']:.4f} (always-negative floor={m['always_negative_accuracy']:.4f})")

        fm = pd.DataFrame(fold_metrics)
        fold_results[label] = fm
        summary = {"horizon": label, "n_folds": n_splits}
        for metric in ["roc_auc", "pr_auc", "recall_top10", "precision_top10", "brier", "accuracy"]:
            summary[f"{metric}_mean"] = fm[metric].mean()
            summary[f"{metric}_std"] = fm[metric].std()
        rows.append(summary)
        print(f"-> mean ROC-AUC across {n_splits} folds: {fm.roc_auc.mean():.4f} (+/- {fm.roc_auc.std():.4f})")
        print(f"-> mean accuracy: {fm.accuracy.mean():.4f}, vs always-negative floor "
              f"{fm.always_negative_accuracy.mean():.4f} -- barely higher despite real ROC-AUC/recall gains")

    results = pd.DataFrame(rows)
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "lgbm_5fold_cv.csv", index=False)
    print(f"\nwrote {out / 'lgbm_5fold_cv.csv'}")
    print(results.to_string(index=False, float_format="{:.4f}".format))

    fig_path = ROOT / "reports" / "figures" / "fig10_accuracy_vs_roc_auc.png"
    plot_accuracy_vs_roc_auc(fold_results, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
