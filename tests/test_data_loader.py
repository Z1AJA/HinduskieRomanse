import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ProjectDataLoader


def test_prepare_classification_divorce():
    loader = ProjectDataLoader()
    prepared = loader.prepare("classification_divorce")
    assert prepared.X_train.shape[0] > 0
    assert prepared.X_test.shape[0] > 0
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1]
    assert prepared.metadata["problem_type"] == "classification"


def test_prepare_regression_years():
    loader = ProjectDataLoader()
    prepared = loader.prepare("regression_years")
    assert prepared.y_train.ndim == 2
    assert prepared.metadata["problem_type"] == "regression"