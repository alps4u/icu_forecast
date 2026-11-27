"""M5: LightGBM hyperparameter search, by the same 5-fold patient-grouped CV as models/train.py.

Search: a random sample of config.yaml's models.lgbm_tuning grid (learning_rate, num_leaves,
min_child_samples, reg_lambda); every other LightGBM setting stays fixed at its M4 value in
models.lgbm. Settings are scored on the primary target only (models.lgbm_tuning.primary_label),
to keep the search affordable; the winning setting is then also scored on the secondary target.

Selection happens entirely on the training+validation CV pool (test split never read, same pool
as train.py); the test set stays locked until M6.

Writes:
- results/m5_tuning_log.csv       every setting tried: mean +/- std CV score, train and validation
- results/m5_comparison.csv       reference, the M4 LightGBM baseline, and the M5 tuned finalist
- results/m5_val_predictions.csv  out-of-fold validation predictions for all three, for the error
                                   analysis notebook
- reports/figures/fig_m5_lgbm_validation_curve.png   train vs. CV score against num_leaves, at the
                                                      finalist's other hyperparameters

Run: python -m icu_forecast.models.tune [--n-settings N] [--sample N]
"""

import argparse
import itertools
import random
import time

import matplotlib.pyplot as plt
import pandas as pd

from icu_forecast.config import ROOT, load_config
from icu_forecast.models.evaluate import evaluate
from icu_forecast.models.knn import sample_patients
from icu_forecast.models.lgbm import fit_lgbm
from icu_forecast.models.train import patient_fold_assignment, to_tree_frame_local
from icu_forecast.prepare import CATEGORICAL, LABELS, feature_columns, load_features
from sklearn.dummy import DummyClassifier

RESULTS = ROOT / "results"
CV_METRICS = ["roc_auc", "pr_auc", "precision_at_recall30", "recall_top10", "precision_top10", "brier"]


def sample_settings(grid: dict, n_settings: int, seed: int) -> list[dict]:
    """n_settings distinct combinations, drawn at random (without replacement) from the grid."""
    keys = ["learning_rate", "num_leaves", "min_child_samples", "reg_lambda"]
    all_combos = list(itertools.product(*(grid[k] for k in keys)))
    n = min(n_settings, len(all_combos))
    chosen = random.Random(seed).sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in chosen]


def build_fold_frames(pool: pd.DataFrame, numeric: list[str], label: str, n_splits: int) -> list[dict]:
    """Train/validation frames for each fold, built once and reused across every setting tried."""
    frames = []
    for k in range(n_splits):
        train_fold = pool[pool.fold != k]
        val_fold = pool[pool.fold == k]
        categories = {c: sorted(train_fold[c].dropna().unique().tolist()) for c in CATEGORICAL}
        frames.append({
            "fold": k,
            "X_train": to_tree_frame_local(train_fold, numeric, categories),
            "y_train": train_fold[label],  # kept as a Series: fit_lgbm slices it with .iloc
            "groups_train": train_fold.subject_id,
            "X_val": to_tree_frame_local(val_fold, numeric, categories),
            "y_val": val_fold[label].to_numpy(),
            "val_index": val_fold.index,
        })
    return frames


def fit_and_score(frames: list[dict], fit_one, with_predictions: bool = False) -> tuple[pd.DataFrame, pd.Series | None]:
    """Fit `fit_one(X, y, groups) -> fitted model` on every fold, score train + validation.

    If with_predictions, also returns the out-of-fold validation probability for every row
    (indexed as in the original pool), for the comparison table's prediction export.
    """
    rows, oof = [], (pd.Series(dtype=float) if with_predictions else None)
    for f in frames:
        model = fit_one(f["X_train"], f["y_train"], f["groups_train"])
        p_val = model.predict_proba(f["X_val"])[:, 1]
        p_train = model.predict_proba(f["X_train"])[:, 1]
        m_val, m_train = evaluate(f["y_val"], p_val), evaluate(f["y_train"], p_train)
        rows.append({"fold": f["fold"], **{f"val_{c}": m_val[c] for c in CV_METRICS},
                     **{f"train_{c}": m_train[c] for c in CV_METRICS}})
        if with_predictions:
            oof = pd.concat([oof, pd.Series(p_val, index=f["val_index"])])
    return pd.DataFrame(rows), oof


def summarize(fm: pd.DataFrame, prefix: str = "") -> dict:
    row = {}
    for metric in CV_METRICS:
        row[f"{prefix}val_{metric}_mean"] = fm[f"val_{metric}"].mean()
        row[f"{prefix}val_{metric}_std"] = fm[f"val_{metric}"].std()
        row[f"{prefix}train_{metric}_mean"] = fm[f"train_{metric}"].mean()
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-settings", type=int, help="override models.lgbm_tuning.n_settings")
    parser.add_argument("--sample", type=int,
                        help="subsample ~N snapshots (whole patients) from the CV pool, for a fast dry run")
    args = parser.parse_args(argv)

    cfg = load_config()
    tcfg = cfg["models"]["lgbm_tuning"]
    label = tcfg["primary_label"]
    select_by = tcfg["select_by"]
    n_splits = cfg["models"]["cv_folds"]
    seed = cfg["seed"]

    df = load_features(cfg)
    numeric, _ = feature_columns(df)
    pool = df.merge(patient_fold_assignment(cfg, n_splits), on="subject_id", how="inner")
    if args.sample:
        pool = sample_patients(pool, args.sample, seed)
    print(f"CV pool: {pool.subject_id.nunique():,} patients, {len(pool):,} snapshots, {n_splits} folds, "
          f"tuning on {label}")

    frames = build_fold_frames(pool, numeric, label, n_splits)

    def lgbm_fitter(overrides: dict):
        run_cfg = {**cfg, "models": {**cfg["models"], "lgbm": {**cfg["models"]["lgbm"], **overrides}}}
        return lambda X, y, groups: fit_lgbm(X, y, groups, run_cfg, metric="auc")

    # --- reference: always "no transfer" (same reference as knn.py/train.py's floor) ---
    ref_fm, ref_oof = fit_and_score(
        frames, lambda X, y, groups: DummyClassifier(strategy="most_frequent").fit(X, y), with_predictions=True)
    print(f"reference: val {select_by} {ref_fm[f'val_{select_by}'].mean():.4f}")

    # --- M4 baseline: today's models.lgbm settings, unchanged, scored by the same folds ---
    t0 = time.time()
    base_fm, base_oof = fit_and_score(frames, lgbm_fitter({}), with_predictions=True)
    print(f"M4 baseline (untuned): val {select_by} {base_fm[f'val_{select_by}'].mean():.4f} "
          f"[{time.time() - t0:.0f}s]")

    # --- random search ---
    n_settings = args.n_settings or tcfg["n_settings"]
    settings = sample_settings(tcfg, n_settings, seed)
    print(f"\nrandom search: {len(settings)} settings drawn from a "
          f"{len(tcfg['learning_rate'])}x{len(tcfg['num_leaves'])}x"
          f"{len(tcfg['min_child_samples'])}x{len(tcfg['reg_lambda'])} grid, selecting by {select_by}\n")

    log_rows = []
    for i, overrides in enumerate(settings):
        t0 = time.time()
        fm, _ = fit_and_score(frames, lgbm_fitter(overrides))
        row = {"setting_id": i, **overrides, **summarize(fm), "seconds": time.time() - t0}
        log_rows.append(row)
        print(f"[{i + 1}/{len(settings)}] {overrides} -> val {select_by} "
              f"{row[f'val_{select_by}_mean']:.4f} (+/- {row[f'val_{select_by}_std']:.4f}) "
              f"[{row['seconds']:.0f}s]")

    log = pd.DataFrame(log_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log.to_csv(RESULTS / "m5_tuning_log.csv", index=False)
    print(f"\nwrote {RESULTS / 'm5_tuning_log.csv'} ({len(log)} settings)")

    best_row = log.loc[log[f"val_{select_by}_mean"].idxmax()]
    best = {k: best_row[k] for k in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda")}
    best["num_leaves"], best["min_child_samples"] = int(best["num_leaves"]), int(best["min_child_samples"])
    print(f"\nbest setting (by mean validation {select_by}): {best}")

    # --- finalist: refit the winner once more, this time keeping validation predictions ---
    tuned_fm, tuned_oof = fit_and_score(frames, lgbm_fitter(best), with_predictions=True)

    # --- validation curve: num_leaves against train/CV score, other hyperparameters at `best` ---
    curve_rows = []
    for nl in tcfg["num_leaves"]:
        overrides = {**best, "num_leaves": nl}
        fm, _ = fit_and_score(frames, lgbm_fitter(overrides))
        curve_rows.append({"num_leaves": nl, f"val_{select_by}_mean": fm[f"val_{select_by}"].mean(),
                           f"val_{select_by}_std": fm[f"val_{select_by}"].std(),
                           f"train_{select_by}_mean": fm[f"train_{select_by}"].mean()})
    curve = pd.DataFrame(curve_rows).sort_values("num_leaves")
    curve.to_csv(RESULTS / "m5_lgbm_validation_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(curve.num_leaves, curve[f"val_{select_by}_mean"], yerr=curve[f"val_{select_by}_std"],
               marker="s", capsize=4, label="validation (5-fold CV)")
    ax.plot(curve.num_leaves, curve[f"train_{select_by}_mean"], marker="o", label="training")
    ax.axvline(best["num_leaves"], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("num_leaves"); ax.set_ylabel(f"mean {select_by}")
    ax.set_title(f"LightGBM ({label}): {select_by} against num_leaves\n"
                f"(learning_rate={best['learning_rate']}, min_child_samples={best['min_child_samples']}, "
                f"reg_lambda={best['reg_lambda']})")
    ax.legend()
    fig.tight_layout()
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_m5_lgbm_validation_curve.png", dpi=120)
    plt.close(fig)
    print(f"wrote {fig_dir / 'fig_m5_lgbm_validation_curve.png'}")

    # --- secondary target: the winning setting applied to y_48h, not re-tuned ---
    label2 = [l for l in LABELS if l != label][0]
    frames2 = build_fold_frames(pool, numeric, label2, n_splits)
    tuned_fm2, tuned_oof2 = fit_and_score(frames2, lgbm_fitter(best), with_predictions=True)

    # --- comparison table ---
    comparison = pd.DataFrame([
        {"model": "Reference: always no-transfer", "horizon": label, **summarize(ref_fm)},
        {"model": "LightGBM, M4 baseline (untuned)", "horizon": label, **summarize(base_fm)},
        {"model": "LightGBM, M5 tuned", "horizon": label, **best, **summarize(tuned_fm)},
        {"model": "LightGBM, M5 tuned", "horizon": label2, **best, **summarize(tuned_fm2)},
    ])
    comparison.to_csv(RESULTS / "m5_comparison.csv", index=False)
    print(f"wrote {RESULTS / 'm5_comparison.csv'}")
    print("\n" + comparison[["model", "horizon", f"val_{select_by}_mean", f"val_{select_by}_std",
                            f"train_{select_by}_mean"]].to_string(index=False, float_format="{:.4f}".format))

    # --- out-of-fold validation predictions, for the error-analysis notebook ---
    preds = pool[["subject_id", "hadm_id", "snapshot_time", "y_24h", "y_48h"]].copy()
    preds["p_reference"] = ref_oof
    preds["p_lgbm_m4_baseline"] = base_oof
    preds[f"p_lgbm_m5_tuned_{label}"] = tuned_oof
    preds[f"p_lgbm_m5_tuned_{label2}"] = tuned_oof2
    preds.to_csv(RESULTS / "m5_val_predictions.csv", index=False)
    print(f"wrote {RESULTS / 'm5_val_predictions.csv'} ({len(preds):,} rows)")


if __name__ == "__main__":
    main()
