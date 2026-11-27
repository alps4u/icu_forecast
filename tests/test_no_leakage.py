"""M5 leakage checks: no record in more than one split, no code reads the test split, and the
cross-validation folds used by train.py / knn.py never draw on test-split patients.

Requires data/processed/split.csv and features.parquet (run the pipeline first; see README).

Run: pytest tests/test_no_leakage.py
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from icu_forecast.config import ROOT, load_config
from icu_forecast.models.train import patient_fold_assignment

SRC = ROOT / "src" / "icu_forecast"
# split.py is where the "test" label is assigned; it does not train on it.
FILES_ALLOWED_TO_MENTION_TEST_SPLIT = {"split.py", "config.py"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def split(cfg):
    return pd.read_csv(cfg["paths"]["processed"] / "split.csv")


def test_every_patient_has_exactly_one_split_label(split):
    assert split.subject_id.is_unique, "a patient must appear in exactly one row of split.csv"
    assert set(split.split.unique()) <= {"train", "val", "test"}


def test_cv_pool_excludes_test_split_patients(cfg, split):
    """patient_fold_assignment() builds the folds train.py and knn.py --cv train on; none of its
    patients may be from the locked test split."""
    test_patients = set(split.loc[split.split == "test", "subject_id"])
    pool = patient_fold_assignment(cfg, cfg["models"]["cv_folds"])
    leaked = set(pool.subject_id) & test_patients
    assert not leaked, f"{len(leaked)} test-split patient(s) leaked into the CV pool"


def test_cv_folds_partition_the_pool_disjointly(cfg):
    """Every CV-pool patient gets exactly one fold id, and every fold is used."""
    n_splits = cfg["models"]["cv_folds"]
    pool = patient_fold_assignment(cfg, n_splits)
    assert pool.subject_id.is_unique, "a patient must not appear in more than one fold"
    assert set(pool.fold.unique()) == set(range(n_splits))


def test_no_source_file_selects_the_test_split():
    """Static guard: no training or tuning code should filter rows by split == 'test'."""
    pattern = re.compile(r"""split\s*==\s*['"]test['"]""")
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name in FILES_ALLOWED_TO_MENTION_TEST_SPLIT:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"these files read the test split: {offenders}"
