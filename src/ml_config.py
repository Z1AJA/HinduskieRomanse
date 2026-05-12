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
    # --- DECISION TREE ---
    ExperimentDefinition("classification_divorce_decision_tree_max_depth", "classification_divorce", "decision_tree_classification", "person2", "max_depth", [3, 5, 10, 20], {"min_samples_leaf": 1}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_decision_tree_max_depth", "regression_years", "decision_tree_regression", "person2", "max_depth", [3, 5, 10, 20], {"min_samples_leaf": 1}, "rmse", ""),

    # --- RANDOM FOREST ---
    ExperimentDefinition("classification_divorce_random_forest_n_estimators", "classification_divorce", "random_forest_classification", "person2", "n_estimators", [50, 100, 200, 300], {"max_depth": 10}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_random_forest_n_estimators", "regression_years", "random_forest_regression", "person2", "n_estimators", [50, 100, 200, 300], {"max_depth": 10}, "rmse", ""),
    ExperimentDefinition("classification_divorce_random_forest_max_depth", "classification_divorce", "random_forest_classification", "person2", "max_depth", [5, 10, 15, None], {"n_estimators": 100}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_random_forest_max_depth", "regression_years", "random_forest_regression", "person2", "max_depth", [5, 10, 15, None], {"n_estimators": 100}, "rmse", ""),
    ExperimentDefinition("classification_divorce_random_forest_min_samples_split", "classification_divorce", "random_forest_classification", "person2", "min_samples_split", [2, 5, 10, 20], {"n_estimators": 100, "max_depth": 10}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_random_forest_min_samples_split", "regression_years", "random_forest_regression", "person2", "min_samples_split", [2, 5, 10, 20], {"n_estimators": 100, "max_depth": 10}, "rmse", ""),

    # --- KNN ---
    ExperimentDefinition("classification_divorce_knn_n_neighbors", "classification_divorce", "knn_classification", "person3", "n_neighbors", [3, 5, 11, 21], {"weights": "uniform"}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_knn_n_neighbors", "regression_years", "knn_regression", "person3", "n_neighbors", [3, 5, 11, 21], {"weights": "uniform"}, "rmse", ""),
    ExperimentDefinition("classification_divorce_knn_weights", "classification_divorce", "knn_classification", "person3", "weights", ["uniform", "distance"], {"n_neighbors": 11}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_knn_weights", "regression_years", "knn_regression", "person3", "weights", ["uniform", "distance"], {"n_neighbors": 11}, "rmse", ""),
    ExperimentDefinition("classification_divorce_knn_p", "classification_divorce", "knn_classification", "person3", "p", [1, 2], {"n_neighbors": 11, "weights": "uniform"}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_knn_p", "regression_years", "knn_regression", "person3", "p", [1, 2], {"n_neighbors": 11, "weights": "uniform"}, "rmse", ""),

    # --- GRADIENT BOOSTING ---
    ExperimentDefinition("classification_divorce_gradient_boosting_learning_rate", "classification_divorce", "gradient_boosting_classification", "person3", "learning_rate", [0.01, 0.05, 0.1, 0.2], {"n_estimators": 100}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_gradient_boosting_learning_rate", "regression_years", "gradient_boosting_regression", "person3", "learning_rate", [0.01, 0.05, 0.1, 0.2], {"n_estimators": 100}, "rmse", ""),
    ExperimentDefinition("classification_divorce_gradient_boosting_n_estimators", "classification_divorce", "gradient_boosting_classification", "person3", "n_estimators", [50, 100, 200, 300], {"learning_rate": 0.1}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_gradient_boosting_n_estimators", "regression_years", "gradient_boosting_regression", "person3", "n_estimators", [50, 100, 200, 300], {"learning_rate": 0.1}, "rmse", ""),
    ExperimentDefinition("classification_divorce_gradient_boosting_max_depth", "classification_divorce", "gradient_boosting_classification", "person3", "max_depth", [2, 3, 5], {"n_estimators": 100, "learning_rate": 0.1}, "balanced_accuracy", ""),
    ExperimentDefinition("regression_years_gradient_boosting_max_depth", "regression_years", "gradient_boosting_regression", "person3", "max_depth", [2, 3, 5], {"n_estimators": 100, "learning_rate": 0.1}, "rmse", ""),
]