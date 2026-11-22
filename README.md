# Forecasting ICU Bed Demand via Patient Deterioration Modeling
IE7275 Group A project — *I See You, ICU*.
Authors: Ka Won Cho, Minh Tuong Le, Shanti Ram Pokhrel.

## 1. Problem Statement & Intended Users

### Problem Statement
Intensive Care Units (ICUs) face chronic bed capacity and staffing bottlenecks, particularly in healthcare systems strained by demographic aging. Unplanned transfers from general hospital wards into ICUs create severe, unpredictable surges in critical care demand. Conventional capacity planning relies heavily on historical rolling averages (e.g., historical bed utilization), which completely ignore the real-time physiological deterioration of currently admitted inpatients. Conversely, bedside early warning systems (e.g., NEWS2) offer very short lead times (4–12 hours), which is insufficient for forward operational planning.

This project develops a predictive model that scores each ward patient's short-term risk of unplanned ICU transfer, focused on frequently admitted inpatients. The project addresses the following research question:
> **For frequently admitted inpatients, can we predict the risk of unplanned ICU transfer within short-term windows (24 and 48 hours), and how much operational and clinical value does a clinical watchlist provide when it flags the 10% of cases with the highest predicted risk?**

### Intended Users
<<<<<<< HEAD
- **Rapid Response Teams & Ward Clinicians:** Utilize the patient-level risk watchlist to identify rapidly decompensating frequent admitters up to 24 hours prior to acute failure for timely bedside intervention.
=======
- **Primary Users (Rapid Response Teams & Ward Clinicians):** Use the patient-level 24h/48h risk watchlist to identify rapidly decompensating frequent admitters ahead of acute failure, for timely bedside intervention.
- **Secondary Users (Hospital Operation Managers & Bed Coordinators):** Use the same patient-level risk scores, aggregated across the ward, to anticipate near-term ICU bed demand and proactively adjust nursing rosters, respiratory equipment, and elective surgical schedules.
>>>>>>> 5a78272e8220ca7e3ad239476b38b6855f9c7ce1

---

## 2. Data Sources & Licensing

### Source Dataset
- **Database:** Medical Information Mart for Intensive Care IV (**MIMIC-IV v3.1**)
- **URL:** [https://physionet.org/content/mimiciv/3.1/](https://physionet.org/content/mimiciv/3.1/)
- **Provider:** PhysioNet / Massachusetts Institute of Technology (MIT) Computational Physiology Lab.
- **Download Date:** 2026-09-23
- **Institutional Origin:** Beth Israel Deaconess Medical Center (BIDMC), Boston, MA.

### Dataset Access & Acquisition
- **License:** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/3.1/) |
                under PhysioNet Restricted Data Use Agreement (DUA) / Creative Commons Attribution 4.0 International (CC BY 4.0).
    **The data are not included in this repository and must not be redistributed.** 
    The license above applies to the data only; the MIT license in [`LICENSE`](LICENSE) covers the code.
- **Access Requirements:** Access requires credentialing on PhysioNet, completion of the CITI "Data or Specimens Only Research" training, and signing the DUA.
- **Data Modules Used:**
  - `hosp.patients`, `hosp.admissions`, `hosp.transfers`, `hosp.labevents`, `hosp.prescriptions`, `hosp.diagnoses_icd`, `hosp.omr`, `hosp.services`
  - `icu.icustays`

### How to obtain the data
1. Create a PhysioNet account and complete credentialing.
2. Complete the CITI course *"Data or Specimens Only Research"* and upload the certificate.
3. Sign the data use agreement on the MIMIC-IV 3.1 project page.
4. Download the files and place them, unchanged (still gzipped), under `data/raw/`:

---

## 3. Installation

Requires **Python 3.12** (`pyproject.toml` pins `>=3.12,<3.13`). About 16 GB of RAM is
recommended; `labevents` (2.6 GB compressed) is streamed in blocks.

```bash
git clone https://github.com/alps4u/icu_forecast.git
cd icu_forecast

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt    # pinned versions; also installs this package (-e .)
pip install jupyter seaborn        # only needed for the notebooks
```

### Key Dependencies (`requirements.txt`)
```text
google-cloud-bigquery>=3.10.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=1.7.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.25.0
```

---

## 4. Reproducing the Results

To reproduce all results and figures reported in the project study, follow the pipeline steps sequentially:

### Step 1: Data Access
Download dataset from Physionet site:

### Step 2: Clean Data and Extract Feature Matrix & Cohorts

```bash
pip install -r requirements.txt   
python -m icu_forecast.clean
python -m icu_forecast.split
python -m icu_forecast.features
python -m icu_forecast.prepare
python -m icu_forecast.export_csv
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_pca.ipynb
```

### Step 3: Baseline Models (validation split)
** Now we have features.csv and features.parquet, let's train with LGBM: 

```bash
python -m icu_forecast.models.knn --cv --label y_24h
python -m icu_forecast.models.knn --cv --label y_48h
```

All models are fitted on the training split and scored on validation; `--test` (reference,
logreg) also scores the held-out test split and is meant for the final evaluation only.

`models/knn.py` notes:

- By default it uses about 20,000 training and 20,000 validation snapshots (whole patients,
  `train_curve_n`); `--full` uses the full splits.


## 5. Repository structure

```
.
├── README.md                 # this file
├── LICENSE                   # MIT license for the code
├── config.yaml               # all paths, seed, split ratios, cohort and feature settings
├── pyproject.toml            # package metadata (installs src/icu_forecast)
├── requirements.txt          # pinned dependencies
│
├── src/icu_forecast/         # pipeline package (run with python -m icu_forecast.<module>)
│   ├── config.py             # loads config.yaml, resolves paths from the repo root
│   ├── clean.py              # cleans raw MIMIC-IV tables, builds the cohort
│   ├── split.py              # patient-level stratified train/val/test split
│   ├── features.py           # 08:00 snapshots, 24 h / 48 h labels, 96 features
│   ├── prepare.py            # imputation, scaling, encoding fitted on train only
<<<<<<< HEAD
│   └── export_csv.py         # exports processed parquet files to CSV
│   └── models/               # baseline models (run with python -m icu_forecast.models.<module>)
│       ├── knn.py            # k-NN classifier: knn.ipynb steps on prepare.py features and split
│       └── knn.py            # k-NN classifier: knn.ipynb steps on prepare.py features and split
│
├── code/                     # earlier copy of the pipeline scripts (superseded by src/)
=======
│   ├── export_csv.py         # exports processed parquet files to CSV
│   └── models/
│       ├── train.py          # M4: the one command -- 5-fold patient-grouped CV for LightGBM
│       ├── knn.py            # k-NN classifier: knn.ipynb steps on prepare.py features and split
│       ├── lgbm.py           # LightGBM fit/plot helpers, used by train.py
│       └── evaluate.py       # shared discrimination/watchlist metrics
>>>>>>> 5a78272e8220ca7e3ad239476b38b6855f9c7ce1
│
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   ├── 02_pca.ipynb          # PCA on the routine lab panel
│   └── 03_logres.ipynb       # Logistic regression model
│
├── sql/
│   └── 01_pipeline_validation.sql   # SQL cross-check of the cohort definition
│
├── reports/
│   ├── IE7275_GroupA_Project_Proposal_M3.pdf   # milestone report (.doc source alongside)
│   ├── IE7275_GroupA_Project_Proposal_M4.pdf   # milestone report (.doc source alongside)
│   ├── data_pipeline.md      # before/after counts for every pipeline step
│   ├── tables/                # M4: committed result tables (small CSVs, not gitignored)
│   │   └── lgbm_5fold_cv.csv  # per-fold and mean +/- std metrics from models/train.py
│   └── figures/              # figures used in the reports
│
<<<<<<< HEAD
├── artifacts/                # not tracked by Git; model outputs
│   └── knn/<label>/          # knn.py results for y_24h / y_48h
=======
├── artifacts/                 # not tracked by Git; trained models (.joblib), rebuilt by models/train.py
>>>>>>> 5a78272e8220ca7e3ad239476b38b6855f9c7ce1
│
└── data/                     # not tracked by Git; created locally
    ├── raw/                  # original MIMIC-IV files (read-only)
    ├── processed/            # parquet outputs of the pipeline
    └── processed_csv/        # CSV export (step 5)
```

## 6. License

- **Code:** MIT License — see [`LICENSE`](LICENSE).
- **Data:** MIMIC-IV v3.1 is distributed by PhysioNet under the PhysioNet Credentialed Health
  Data License 1.5.0. No patient-level data is stored in this repository.

## 7. Citation

Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R.
(2024). *MIMIC-IV* (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58
