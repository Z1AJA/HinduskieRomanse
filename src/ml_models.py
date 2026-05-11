from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


EstimatorFactory = Callable[..., object]


@dataclass(frozen=True)
class ModelSpec:
    method_name: str
    family: str
    owner: str
    problem_type: str
    factory: EstimatorFactory
    default_params: dict


def _decision_tree_classifier(**kwargs):
    return DecisionTreeClassifier(**kwargs)


def _decision_tree_regressor(**kwargs):
    return DecisionTreeRegressor(**kwargs)


def _random_forest_classifier(**kwargs):
    return RandomForestClassifier(**kwargs)


def _random_forest_regressor(**kwargs):
    return RandomForestRegressor(**kwargs)


def _knn_classifier(**kwargs):
    return KNeighborsClassifier(**kwargs)


def _knn_regressor(**kwargs):
    return KNeighborsRegressor(**kwargs)


def _gb_classifier(**kwargs):
    return GradientBoostingClassifier(**kwargs)


def _gb_regressor(**kwargs):
    return GradientBoostingRegressor(**kwargs)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree_classification": ModelSpec(
        method_name="decision_tree",
        family="tree",
        owner="person2",
        problem_type="classification",
        factory=_decision_tree_classifier,
        default_params={"criterion": "gini"},
    ),
    "decision_tree_regression": ModelSpec(
        method_name="decision_tree",
        family="tree",
        owner="person2",
        problem_type="regression",
        factory=_decision_tree_regressor,
        default_params={"criterion": "squared_error"},
    ),
    "random_forest_classification": ModelSpec(
        method_name="random_forest",
        family="tree_ensemble",
        owner="person2",
        problem_type="classification",
        factory=_random_forest_classifier,
        default_params={"criterion": "gini", "n_jobs": -1},
    ),
    "random_forest_regression": ModelSpec(
        method_name="random_forest",
        family="tree_ensemble",
        owner="person2",
        problem_type="regression",
        factory=_random_forest_regressor,
        default_params={"criterion": "squared_error", "n_jobs": -1},
    ),
    "knn_classification": ModelSpec(
        method_name="knn",
        family="distance",
        owner="person3",
        problem_type="classification",
        factory=_knn_classifier,
        default_params={"weights": "uniform", "metric": "minkowski"},
    ),
    "knn_regression": ModelSpec(
        method_name="knn",
        family="distance",
        owner="person3",
        problem_type="regression",
        factory=_knn_regressor,
        default_params={"weights": "uniform", "metric": "minkowski"},
    ),
    "gradient_boosting_classification": ModelSpec(
        method_name="gradient_boosting",
        family="boosting",
        owner="person3",
        problem_type="classification",
        factory=_gb_classifier,
        default_params={"n_estimators": 100},
    ),
    "gradient_boosting_regression": ModelSpec(
        method_name="gradient_boosting",
        family="boosting",
        owner="person3",
        problem_type="regression",
        factory=_gb_regressor,
        default_params={"n_estimators": 100},
    ),
}
