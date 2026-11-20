"""k-NN classifier for 24h / 48h ward-to-ICU transfer: knn.ipynb rewritten on the project pipeline.

Metric: accuracy is reported as in the notebook, but with 1-2% positives "never transfer" already
scores ~98%, so the best k is chosen by PR-AUC (or --select-by roc_auc), from predict_proba.

Settings: the `knn:` block of config.yaml (k_grid, cv_folds, cv_n, train_curve_n); --k overrides k_grid.

Size: k-NN scores every query against every training row, so --n-train / --n-val (default from
`knn.train_curve_n` in config.yaml) subsample whole patients. Pass 0 to use the full split.

Run: python -m icu_forecast.models.knn [--label y_48h] [--n-train 20000] [--n-val 20000] [--cv]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from icu_forecast.config import load_config
from icu_forecast.prepare import LABELS, build_preprocessor, feature_columns, load_features
WATCHLIST_SHARE = 0.10


def build_unscaled(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """prepare.build_preprocessor() without the StandardScaler: the notebook's "unscaled" k-NN."""
    # k-NN cannot take NaN: fill with the training median, plus a 0/1 "was missing" column per feature
    # (an unordered lab is itself informative).
    num = SimpleImputer(strategy="median", add_indicator=True)
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # Categoricals become 0/1 columns; levels under 0.1% of training rows (or unseen) share one column.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001, sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def knn_pipeline(numeric, categorical, k, metric="euclidean", weights="uniform", scaled=True) -> Pipeline:
    # Preprocessing lives inside the pipeline, so .fit() learns medians / means / levels from the
    # rows it is fitted on only (training split, or the fitting folds in CV): no leakage.
    pre = build_preprocessor(numeric, categorical) if scaled else build_unscaled(numeric, categorical)
    return Pipeline([("pre", pre), ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights))])


def sample_patients(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """About n snapshots, taking whole patients at random; n <= 0 keeps everything."""
    if n <= 0 or len(df) <= n:
        return df
    # Sample patients, not rows, so a patient's snapshots are never split between kept and dropped.
    patients = df.subject_id.drop_duplicates().sample(frac=1.0, random_state=seed)
    sizes = df.groupby("subject_id").size().reindex(patients)
    # Keep patients in shuffled order until their snapshots add up to about n.
    keep = sizes.index[sizes.cumsum() <= n]
    return df[df.subject_id.isin(keep)]

def evaluate(y: np.ndarray, score: np.ndarray, share: float = WATCHLIST_SHARE) -> dict:
    """Ranking metrics plus recall / precision of a watchlist flagging the top `share` of snapshots.

    Snapshots tied at the watchlist cut-off are split at random; the top-share metrics are
    the expected value of that draw, so models with coarse or constant scores are not
    favoured or penalised by row order.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    k = int(np.ceil(share * len(y)))
    cutoff = np.sort(score)[::-1][k - 1]
    above, tied = score > cutoff, score == cutoff
    hits = y[above].sum() + (k - above.sum()) * y[tied].mean()
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        f"recall_top{int(share * 100)}": hits / max(int(y.sum()), 1),
        f"precision_top{int(share * 100)}": hits / k,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", choices=LABELS, default="y_24h", help="target (the notebook's TARGET)")
    parser.add_argument("--n-train", type=int, help="training snapshots used (0 = all; default knn.train_curve_n)")
    parser.add_argument("--n-val", type=int, help="validation snapshots used (0 = all; default knn.train_curve_n)")
    parser.add_argument("--k", type=int, nargs="+", help="values of k to try (default knn.k_grid)")
    parser.add_argument("--select-by", choices=["pr_auc", "roc_auc", "accuracy"], default="pr_auc")
    parser.add_argument("--cv", action="store_true", help="also run the grouped k-fold CV over k on train + val")
    args = parser.parse_args()

    cfg = load_config()
    # Settings come from the knn: block of config.yaml; a missing key raises KeyError (no silent defaults).
    kcfg = cfg["knn"]
    seed = cfg["seed"]
    k_grid = args.k or kcfg["k_grid"]
    n_default = kcfg["train_curve_n"]
    # Omitted -> config default; 0 -> the full split (sample_patients keeps everything).
    n_train = n_default if args.n_train is None else args.n_train
    n_val = n_default if args.n_val is None else args.n_val
    out = cfg["paths"]["artifacts"] / "knn" / args.label
    out.mkdir(parents=True, exist_ok=True)

    # --- data: features.py output + patient-level split, via prepare.py (test split never read) ---
    # features.parquet joined with split.csv: every snapshot is tagged train / val / test by patient.
    df = load_features(cfg)
    numeric, categorical = feature_columns(df)
    FEATURES, TARGET = numeric + categorical, args.label
    # Only the train and val splits are ever selected; test stays untouched.
    train = sample_patients(df[df.split == "train"], n_train, seed)
    val = sample_patients(df[df.split == "val"], n_val, seed)
    X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    X_val, y_val = val[FEATURES], val[TARGET].to_numpy()
    print(f"{TARGET}: {len(numeric)} numeric + {len(categorical)} categorical features; "
          f"train {len(train):,} / val {len(val):,} snapshots")

    shares = pd.DataFrame({
        "training": pd.Series(y_train).value_counts(normalize=True),
        "validation": pd.Series(y_val).value_counts(normalize=True),
    }).round(4)
    shares.loc["records"] = [len(y_train), len(y_val)]
    shares.to_csv(out / "shares.csv")
    print("\n" + shares.to_string())

    results = []

    def score(model, label, **extra) -> dict:
        """Validation accuracy + ranking metrics of a fitted model, added to the results table."""
        # One neighbour search: column 1 = probability of ICU transfer (share of transfers among the
        # k neighbours, 1/distance-weighted if weights="distance").
        proba = model.predict_proba(X_val)
        pred = model.classes_[proba.argmax(axis=1)]  # same as model.predict(X_val), without a second search
        # Accuracy is kept for the notebook, but ranking metrics (ROC-AUC, PR-AUC, top-10% watchlist)
        # are what matter with 1-2% positives.
        row = {"model": label, **extra, "accuracy": accuracy_score(y_val, pred),
               **evaluate(y_val, proba[:, 1])}
        results.append(row)
        print(f"{label}: accuracy {row['accuracy']:.4f}, ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.4f}")
        return row

    # --- reference and scaled vs unscaled at k = 10 ---
    print()
    # Floor to beat: always "no transfer", so ROC-AUC = 0.5 and PR-AUC = the positive rate.
    score(DummyClassifier(strategy="most_frequent").fit(X_train, y_train), "reference: most frequent class")
    # Distances are dominated by large-valued features unless they are standardized.
    score(knn_pipeline(numeric, categorical, 10, scaled=False).fit(X_train, y_train), "k-NN, k = 10, unscaled")
    knn_scaled = knn_pipeline(numeric, categorical, 10).fit(X_train, y_train)
    score(knn_scaled, "k-NN, k = 10, standardized")

    # --- the scaler inside the pipeline learned its means from the training data only ---
    pre = knn_scaled.named_steps["pre"]
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    means = pd.DataFrame({
        # scaler.mean_ also covers the missing-indicator columns; keep the numeric features only.
        "scaler mean": scaler.mean_[:len(numeric)],
        "training mean (imputed)": imputer.transform(X_train[numeric])[:, :len(numeric)].mean(axis=0),
        "validation mean (imputed)": imputer.transform(X_val[numeric])[:, :len(numeric)].mean(axis=0),
    }, index=numeric)
    means.to_csv(out / "scaler_means.csv")

    # --- k sweep: training vs validation ---
    curve = []
    for k in k_grid:
        model = knn_pipeline(numeric, categorical, k).fit(X_train, y_train)
        # Training scores are optimistic (each training row is its own nearest neighbour, decisive at
        # k = 1); the gap to validation shows over-fitting. Predicting all training rows is the
        # costliest step of the script.
        p_tr, p_va = model.predict_proba(X_train), model.predict_proba(X_val)
        tr, va = evaluate(y_train, p_tr[:, 1]), evaluate(y_val, p_va[:, 1])
        curve.append({"k": k,
                      "train_accuracy": accuracy_score(y_train, model.classes_[p_tr.argmax(axis=1)]),
                      "val_accuracy": accuracy_score(y_val, model.classes_[p_va.argmax(axis=1)]),
                      "train_roc_auc": tr["roc_auc"], "val_roc_auc": va["roc_auc"],
                      "train_pr_auc": tr["pr_auc"], "val_pr_auc": va["pr_auc"]})
        print(f"k = {k}: val ROC-AUC {va['roc_auc']:.4f}, PR-AUC {va['pr_auc']:.4f}")
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "k_sweep.csv", index=False)
    # Best k is chosen on validation only; idxmax returns the first (smallest) k on ties.
    best_k = int(curve.k[curve[f"val_{args.select_by}"].idxmax()])  # smallest k with the best score
    print(f"\nBest k on the validation set by {args.select_by}: {best_k}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["accuracy", "roc_auc", "pr_auc"]):
        ax.plot(curve.k, curve[f"train_{m}"], marker="o", label="training")
        ax.plot(curve.k, curve[f"val_{m}"], marker="s", label="validation")
        ax.axvline(best_k, color="grey", linestyle="--")
        ax.set_xscale("log"); ax.set_xlabel("k"); ax.set_ylabel(m); ax.legend()
    fig.suptitle(f"{TARGET}: k-NN against k, standardized features")
    fig.tight_layout(); fig.savefig(out / "k_sweep.png", dpi=120); plt.close(fig)

    # --- metric x weights at the best k ---
    print()
    # 2 metrics x 2 weightings at the best k. manhattan is ~6x slower than euclidean here (no
    # matrix-multiplication shortcut in the brute-force search).
    for metric in ["euclidean", "manhattan"]:
        for weights in ["uniform", "distance"]:
            model = knn_pipeline(numeric, categorical, best_k, metric, weights).fit(X_train, y_train)
            score(model, f"k-NN, k = {best_k}, {metric}, {weights}", k=best_k, metric=metric, weights=weights)

    res = pd.DataFrame(results)
    res.to_csv(out / "results.csv", index=False)
    # Only the metric x weights rows have a "metric" value; the winner feeds best_params.json and --cv.
    combos = res.dropna(subset=["metric"])
    best = combos.loc[combos[args.select_by].idxmax()]
    best_params = {"label": TARGET, "k": best_k, "metric": best.metric, "weights": best.weights,
                   "select_by": args.select_by, "n_train": len(train), "n_val": len(val)}
    with open(out / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # --- optional: grouped k-fold CV over k on dev = train + val (whole patients), using the
    #     metric / weights that won the comparison above ---
    if args.cv:
        cv_metric, cv_weights = best.metric, best.weights
        print(f"\nCV with metric={cv_metric}, weights={cv_weights}")
        # Dev set = train + val (never test), cv_n snapshots of whole patients; independent of --n-train.
        dev = sample_patients(df[df.split.isin(["train", "val"])], kcfg["cv_n"], seed)
        X_dev, y_dev, groups = dev[FEATURES], dev[TARGET].to_numpy(), dev.subject_id.to_numpy()
        # groups = patient: a patient never sits in both the fitting and the held-out folds;
        # stratified: each fold keeps about the same share of transfers.
        folds = StratifiedGroupKFold(n_splits=kcfg["cv_folds"], shuffle=True, random_state=seed)
        cv_rows = []
        for k in k_grid:
            s = []
            for tr_idx, va_idx in folds.split(X_dev, y_dev, groups):
                # A fresh pipeline per fold: preprocessing is re-fitted on the fitting folds only.
                model = knn_pipeline(numeric, categorical, k, cv_metric, cv_weights).fit(X_dev.iloc[tr_idx], y_dev[tr_idx])
                p = model.predict_proba(X_dev.iloc[va_idx])
                s.append(accuracy_score(y_dev[va_idx], model.classes_[p.argmax(axis=1)])
                         if args.select_by == "accuracy" else evaluate(y_dev[va_idx], p[:, 1])[args.select_by])
            cv_rows.append({"k": k, "metric": cv_metric, "weights": cv_weights, "mean": np.mean(s), "std": np.std(s)})
            print(f"CV k = {k}: {args.select_by} {np.mean(s):.4f} +/- {np.std(s):.4f}")
        cv = pd.DataFrame(cv_rows)
        cv.to_csv(out / "cv_k.csv", index=False)
        # Reported only; best_params.json keeps the validation choice. Compare means against the std:
        # differences smaller than the fold-to-fold spread are not meaningful.
        cv_best = int(cv.k[cv["mean"].idxmax()])
        print(f"Best k by grouped CV ({args.select_by}): {cv_best}")

        plt.figure(figsize=(8, 4))
        plt.errorbar(cv.k, cv["mean"], yerr=cv["std"], fmt="s-", capsize=5)
        plt.axvline(cv_best, color="grey", linestyle="--")
        plt.xscale("log"); plt.xlabel("k"); plt.ylabel(f"CV {args.select_by}")
        plt.title(f"{TARGET}: grouped CV against k ({cv_metric}, {cv_weights})")
        plt.tight_layout(); plt.savefig(out / "cv_k.png", dpi=120); plt.close()

    print("\n" + res.drop(columns=["n"]).to_string(index=False, float_format="{:.4f}".format))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
