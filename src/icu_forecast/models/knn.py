"""k-NN classifier for 24h / 48h ward-to-ICU transfer: knn.ipynb rewritten on the project pipeline.

Metric: accuracy is reported as in the notebook, but with 1-2% positives "never transfer" already
scores ~98%, so k is chosen by models.knn.select_by (ROC-AUC; or --select-by), from predict_proba.

Settings: the models.knn block of config.yaml (metric, k_grid, weights, select_by, cv_train_n);
--k overrides k_grid.

Size: k-NN scores every query against every training row, so by default it uses patient samples of
models.knn.cv_train_n (20,000) snapshots: the training and validation splits of the validation step,
and the training part of every CV fold (the held-out CV fold is always scored in full). --full uses
every row instead (hours rather than minutes); --n-train / --n-val override the validation step only.

--cv: LightGBM's patient 5-fold CV (train.patient_fold_assignment); every k is scored on the
validation folds, and the k with the best mean select_by goes, with training and validation metrics
per fold, to reports/tables/knn_fold_metrics_<label>.csv (read by run_baselines.py).

Run: python -m icu_forecast.models.knn [--label y_48h] [--cv] [--full]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from icu_forecast.config import ROOT, load_config  
from icu_forecast.prepare import LABELS, build_preprocessor, feature_columns, load_features
from icu_forecast.models.evaluate import evaluate  # Cho: same metrics function as LightGBM (models/evaluate.py)
from icu_forecast.models.train import patient_fold_assignment  # Cho: same patient 5-fold assignment as LightGBM

METRICS = ["roc_auc", "pr_auc", "recall_top10", "precision_top10", "brier"]  # Cho: as reported by train.py

def knn_pipeline(numeric, categorical, k, metric="euclidean", weights="distance") -> Pipeline:  
    # Preprocessing lives inside the pipeline to make sure no data leakage
    pre = build_preprocessor(numeric, categorical)
    return Pipeline([("pre", pre), ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1))])

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

def main(argv=None) -> None:  
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", choices=LABELS, default="y_24h", help="target (the notebook's TARGET)")
    parser.add_argument("--n-train", type=int, help="training snapshots used (0 = all; default knn.cv_train_n)") 
    parser.add_argument("--n-val", type=int, help="validation snapshots used (0 = all; default knn.cv_train_n)")  
    parser.add_argument("--k", type=int, nargs="+", help="values of k to try (default knn.k_grid)")
    parser.add_argument("--select-by", choices=["pr_auc", "roc_auc", "accuracy"],
                        help="default models.knn.select_by")  # Cho: default from config.yaml (roc_auc)
    parser.add_argument("--cv", action="store_true", help="also run the grouped k-fold CV over k on train + val")
    parser.add_argument("--full", action="store_true",
                        help="use every row (full splits, full CV training folds) instead of cv_train_n samples")  # Cho:
    args = parser.parse_args(argv)  

    cfg = load_config()
    kcfg = cfg["models"]["knn"]
    args.select_by = args.select_by or kcfg["select_by"]  # Cho: roc_auc, LightGBM's primary metric
    seed = cfg["seed"]
    k_grid = args.k or kcfg["k_grid"]
    n_default = 0 if args.full else kcfg["cv_train_n"]  # Cho: 20,000-snapshot samples by default, every row with --full
    metric = kcfg["metric"]
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
    shares.to_csv(out / "knn_shares.csv")
    print("\n" + shares.to_string())

    results = []

    def score(model, label, **extra) -> dict:
        """Validation accuracy + ranking metrics of a fitted model, added to the results table."""
        # One neighbour search: column 1 = probability of ICU transfer 
        # (share of transfers among the k neighbours, 1/distance-weighted if weights="distance").
        proba = model.predict_proba(X_val)
        pred = model.classes_[proba.argmax(axis=1)]  # same as model.predict(X_val)
        # Ranking metrics (ROC-AUC, PR-AUC, top-10% watchlist) matter due to severe class imbalance.
        # Accuracy is kept for the remarks.
        row = {"model": label, **extra, "accuracy": accuracy_score(y_val, pred),
               **evaluate(y_val, proba[:, 1])}
        tr = evaluate(y_train, model.predict_proba(X_train)[:, 1])
        row.update({"train_roc_auc": tr["roc_auc"], "train_pr_auc": tr["pr_auc"]})
        results.append(row)
        print(f"{label}: accuracy {row['accuracy']:.4f}, ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.4f}")

        return row

    # Reference with majority class
    print()
    # Floor to beat: always "no transfer", so ROC-AUC = 0.5 and PR-AUC = the positive rate.
    score(DummyClassifier(strategy="most_frequent").fit(X_train, y_train), "reference: most frequent class")

    # Best k: training vs validation 
    curve = []
    for k in k_grid:
        model = knn_pipeline(numeric, categorical, k, metric).fit(X_train, y_train)  # Cho: metric from config
        # Training scores are optimistic (each training row is its own nearest neighbour, decisive at k = 1)
        # The gap to validation shows over-fitting. 
        p_tr, p_va = model.predict_proba(X_train), model.predict_proba(X_val)
        tr, va = evaluate(y_train, p_tr[:, 1]), evaluate(y_val, p_va[:, 1])
        curve.append({"k": k,
                      "train_accuracy": accuracy_score(y_train, model.classes_[p_tr.argmax(axis=1)]),
                      "val_accuracy": accuracy_score(y_val, model.classes_[p_va.argmax(axis=1)]),
                      "train_roc_auc": tr["roc_auc"], "val_roc_auc": va["roc_auc"],
                      "train_pr_auc": tr["pr_auc"], "val_pr_auc": va["pr_auc"]})
        print(f"k = {k}: val ROC-AUC {va['roc_auc']:.4f}, PR-AUC {va['pr_auc']:.4f}")
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "knn_best_k.csv", index=False)
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
    fig.tight_layout(); fig.savefig(out / "knn_best_k.png", dpi=120); plt.close(fig)

    # Weights at the best k
    print()
    for weights in kcfg["weights"]:  # Cho: metric fixed to models.knn.metric 
        model = knn_pipeline(numeric, categorical, best_k, metric, weights).fit(X_train, y_train)  
        score(model, f"k-NN, k = {best_k}, {metric}, {weights}", k=best_k, metric=metric, weights=weights)  

    res = pd.DataFrame(results)
    res.to_csv(out / "knn_results.csv", index=False)
    # Only the metric x weights rows have a "metric" value; the winner feeds knn_best_params.json and --cv.
    combos = res.dropna(subset=["metric"])
    best = combos.loc[combos[args.select_by].idxmax()]
    best_params = {"label": TARGET, "k": best_k, "metric": best.metric, "weights": best.weights,
                   "select_by": args.select_by, "n_train": len(train), "n_val": len(val)}
    with open(out / "knn_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Optional: grouped k-fold CV over k on dev = train + val (whole patients), using best metric/weights
    if args.cv:
        cv_metric, cv_weights = metric, best.weights  
        print(f"\nCV with metric={cv_metric}, weights={cv_weights}")
        # Cho: LightGBM's folds (train.patient_fold_assignment) on the full train + val pool (test never read)
        n_splits = cfg["models"]["cv_folds"]  
        pool = df.merge(patient_fold_assignment(cfg, n_splits), on="subject_id", how="inner")  # Cho:
        preds = pool[["subject_id", "hadm_id", "snapshot_time", "fold", TARGET]].copy()  # Cho: validation predictions
        cv_n = 0 if args.full else kcfg["cv_train_n"]  # Cho: CV training sample size, every row with --full
        cv_rows, fold_rows = [], []     # Cho: fold_rows = validation metrics per fold and k
        for k in k_grid:
            s = []
            for f in range(n_splits):  # Cho: fold f held out, as in train.py
                fit = sample_patients(pool[pool.fold != f], cv_n, seed)  # Cho: ~20,000 training snapshots (all with --full)
                va = pool[pool.fold == f]  # Cho: full validation fold, the same rows LightGBM is scored on
                # A fresh pipeline per fold: preprocessing is re-fitted on the fitting sample only.
                model = knn_pipeline(numeric, categorical, k, cv_metric, cv_weights).fit(fit[FEATURES], fit[TARGET]) 
                p = model.predict_proba(va[FEATURES])  # Cho: sklearn predict_proba on the whole fold
                preds.loc[va.index, f"p_k{k}"] = p[:, 1]  # Cho: P(transfer) for this k
                y_va = va[TARGET].to_numpy()  
                m_va = evaluate(y_va, p[:, 1])  # Cho: all validation metrics
                acc = ((p[:, 1] >= 0.5) == y_va).mean()  # Cho: 0.5 threshold, as train.py
                fold_rows.append({"model": f"k-NN (k = {k}, {cv_weights})", "horizon": TARGET, "fold": f, "k": k,
                                  "weights": cv_weights, "n_train": len(fit), "n_val": len(va),
                                  **{f"val_{c}": m_va[c] for c in METRICS}, "val_accuracy": acc})  # Cho: training metrics below, chosen k only
                s.append(acc if args.select_by == "accuracy" else m_va[args.select_by])  # Cho:
            cv_rows.append({"k": k, "metric": cv_metric, "weights": cv_weights, "mean": np.mean(s), "std": np.std(s)})
            print(f"CV k = {k}: {args.select_by} {np.mean(s):.4f} +/- {np.std(s):.4f}")
        folds_df = pd.DataFrame(fold_rows)  
        folds_df.to_csv(out / "knn_cv_folds.csv", index=False)  # Cho: every k, every fold
        cols = [f"val_{c}" for c in METRICS] + ["val_accuracy"]  # Cho: validation metrics for every k
        summ = folds_df.groupby("k")[cols].agg(["mean", "std"])  # Cho: sync with lgbm_5fold_cv.csv
        summ.columns = [f"{c}_{stat}" for c, stat in summ.columns]  
        cv = pd.DataFrame(cv_rows).merge(summ.reset_index(), on="k")  # Cho: train / val mean and std per k
        cv.to_csv(out / "knn_cv_k.csv", index=False)
        preds.to_parquet(out / "knn_val_predictions.parquet", index=False)  # Cho: out-of-fold validation predictions
        cv_best = int(cv.k[cv["mean"].idxmax()])
        print(f"Best k by grouped CV ({args.select_by}): {cv_best}")
        # Cho: training metrics only at the chosen k (predicting every training row is the costliest step,
        # Cho: hours with --full). predict_proba on the training rows: each row is its own nearest neighbour,
        # Cho: so with weights="distance" the training scores are 1.0.
        best_rows = folds_df[folds_df.k == cv_best].copy() 
        for f in range(n_splits):  # Cho: the same training sample as above (same seed)
            fit = sample_patients(pool[pool.fold != f], cv_n, seed) 
            model = knn_pipeline(numeric, categorical, cv_best, cv_metric, cv_weights).fit(fit[FEATURES], fit[TARGET])  
            m_tr = evaluate(fit[TARGET].to_numpy(), model.predict_proba(fit[FEATURES])[:, 1])  
            for c in METRICS:  
                best_rows.loc[best_rows.fold == f, f"train_{c}"] = m_tr[c] 
        tables = ROOT / "reports" / "tables" 
        tables.mkdir(parents=True, exist_ok=True)  
        best_rows.to_csv(tables / f"knn_fold_metrics_{TARGET}.csv", index=False)  

        plt.figure(figsize=(8, 4))
        plt.errorbar(cv.k, cv["mean"], yerr=cv["std"], fmt="s-", capsize=5)
        plt.axvline(cv_best, color="grey", linestyle="--")
        plt.xscale("log"); plt.xlabel("k"); plt.ylabel(f"CV {args.select_by}")
        plt.title(f"{TARGET}: grouped CV against k ({cv_metric}, {cv_weights})")
        plt.tight_layout(); plt.savefig(out / "knn_cv_k.png", dpi=120); plt.close()

    print("\n" + res.drop(columns=["n"]).to_string(index=False, float_format="{:.4f}".format))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
