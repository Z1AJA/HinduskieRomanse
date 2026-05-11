from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_TASKS, PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR

UM_RESULTS_DIR = RESULTS_DIR / "um"
UM_PLOTS_DIR = UM_RESULTS_DIR / "plots"
UM_RANDOM_SEEDS = [RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2]

DEFAULT_UM_TASKS = [
    DEFAULT_TASKS["classification"],
    DEFAULT_TASKS["regression"],
]


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    task_name: str
    model_key: str
    owner: str
    tuned_param: str
    values: list
    fixed_params: dict
    primary_metric: str
    notes: str



UM_EXPERIMENTS: list[ExperimentDefinition] = [
    ExperimentDefinition(
        experiment_id="classification_divorce_decision_tree_max_depth",
        task_name="classification_divorce",
        model_key="decision_tree_classification",
        owner="person2",
        tuned_param="max_depth",
        values=[3, 5, 10, 20],
        fixed_params={"min_samples_leaf": 1},
        primary_metric="balanced_accuracy",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="regression_years_decision_tree_max_depth",
        task_name="regression_years",
        model_key="decision_tree_regression",
        owner="person2",
        tuned_param="max_depth",
        values=[3, 5, 10, 20],
        fixed_params={"min_samples_leaf": 1},
        primary_metric="rmse",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="classification_divorce_random_forest_n_estimators",
        task_name="classification_divorce",
        model_key="random_forest_classification",
        owner="person2",
        tuned_param="n_estimators",
        values=[50, 100, 200, 300],
        fixed_params={"max_depth": 10},
        primary_metric="balanced_accuracy",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="regression_years_random_forest_n_estimators",
        task_name="regression_years",
        model_key="random_forest_regression",
        owner="person2",
        tuned_param="n_estimators",
        values=[50, 100, 200, 300],
        fixed_params={"max_depth": 10},
        primary_metric="rmse",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="classification_divorce_knn_n_neighbors",
        task_name="classification_divorce",
        model_key="knn_classification",
        owner="person3",
        tuned_param="n_neighbors",
        values=[3, 5, 11, 21],
        fixed_params={"weights": "uniform"},
        primary_metric="balanced_accuracy",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="regression_years_knn_n_neighbors",
        task_name="regression_years",
        model_key="knn_regression",
        owner="person3",
        tuned_param="n_neighbors",
        values=[3, 5, 11, 21],
        fixed_params={"weights": "uniform"},
        primary_metric="rmse",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="classification_divorce_gradient_boosting_learning_rate",
        task_name="classification_divorce",
        model_key="gradient_boosting_classification",
        owner="person3",
        tuned_param="learning_rate",
        values=[0.01, 0.05, 0.1, 0.2],
        fixed_params={"n_estimators": 100},
        primary_metric="balanced_accuracy",
        notes="Parametr do raportu głównego.",
    ),
    ExperimentDefinition(
        experiment_id="regression_years_gradient_boosting_learning_rate",
        task_name="regression_years",
        model_key="gradient_boosting_regression",
        owner="person3",
        tuned_param="learning_rate",
        values=[0.01, 0.05, 0.1, 0.2],
        fixed_params={"n_estimators": 100},
        primary_metric="rmse",
        notes="Parametr do raportu głównego.",
    ),
]