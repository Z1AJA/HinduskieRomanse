from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, RANDOM_STATE, TASK_CONFIGS
from .ml_config import DEFAULT_UM_TASKS, UM_EXPERIMENTS, UM_PLOTS_DIR, UM_RANDOM_SEEDS, UM_RESULTS_DIR, ExperimentDefinition
from .ml_metrics import evaluate_classification, evaluate_regression
from .ml_models import MODEL_REGISTRY


def load_processed_dataset(task_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    npz_path = PROCESSED_DATA_DIR / f"{task_name}.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{task_name}_metadata.json"

    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Brakuje przygotowanych danych dla tasku {task_name}. Najpierw odpal src.run_data_prep."
        )

    data = np.load(npz_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return data["X_train"], data["X_test"], data["y_train"], data["y_test"], metadata


class UMExperimentRunner:
    def __init__(self, seeds: list[int] | None = None, results_dir: Path = UM_RESULTS_DIR, plots_dir: Path = UM_PLOTS_DIR):
        self.seeds = seeds or UM_RANDOM_SEEDS
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run_definition(self, definition: ExperimentDefinition) -> pd.DataFrame:
        X_train, X_test, y_train, y_test, metadata = load_processed_dataset(definition.task_name)
        problem_type = metadata["problem_type"]
        model_spec = MODEL_REGISTRY[definition.model_key]

        rows: list[dict] = []
        for value in definition.values:
            for repeat_idx, seed in enumerate(self.seeds, start=1):
                params = dict(model_spec.default_params)
                params.update(definition.fixed_params)
                params[definition.tuned_param] = value

                if "random_state" not in params and model_spec.method_name != "knn":
                    params["random_state"] = seed

                estimator = model_spec.factory(**params)
                estimator.fit(X_train, y_train.reshape(-1))

                y_train_pred = estimator.predict(X_train)
                y_test_pred = estimator.predict(X_test)

                if problem_type == "classification":
                    train_metrics = evaluate_classification(y_train, y_train_pred)
                    test_metrics = evaluate_classification(y_test, y_test_pred)
                else:
                    train_metrics = evaluate_regression(y_train, y_train_pred)
                    test_metrics = evaluate_regression(y_test, y_test_pred)

                row = {
                    "experiment_id": definition.experiment_id,
                    "task_name": definition.task_name,
                    "problem_type": problem_type,
                    "owner": definition.owner,
                    "method_name": model_spec.method_name,
                    "model_key": definition.model_key,
                    "tuned_param": definition.tuned_param,
                    "param_value": value,
                    "repeat": repeat_idx,
                    "random_seed": seed,
                }
                for metric_name, metric_value in train_metrics.items():
                    row[f"train_{metric_name}"] = metric_value
                for metric_name, metric_value in test_metrics.items():
                    row[f"test_{metric_name}"] = metric_value
                rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / f"{definition.experiment_id}.csv"
        df.to_csv(csv_path, index=False)
        self._save_summary(definition, df)
        self._save_plot(definition, df)
        return df

    def _save_summary(self, definition: ExperimentDefinition, df: pd.DataFrame) -> None:
        grouped = df.groupby("param_value", dropna=False).mean(numeric_only=True).reset_index()
        summary_path = self.results_dir / f"{definition.experiment_id}_summary.csv"
        grouped.to_csv(summary_path, index=False)

    def _save_plot(self, definition: ExperimentDefinition, df: pd.DataFrame) -> None:
        grouped = df.groupby("param_value", dropna=False).mean(numeric_only=True).reset_index()
        metric = definition.primary_metric
        train_col = f"train_{metric}"
        test_col = f"test_{metric}"
        if train_col not in grouped.columns or test_col not in grouped.columns:
            return

        plt.figure(figsize=(8, 5))
        plt.plot(grouped["param_value"], grouped[train_col], marker="o", label=f"train_{metric}")
        plt.plot(grouped["param_value"], grouped[test_col], marker="o", label=f"test_{metric}")
        plt.title(definition.experiment_id)
        plt.xlabel(definition.tuned_param)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = self.plots_dir / f"{definition.experiment_id}_{metric}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def run_many(self, definitions: Iterable[ExperimentDefinition]) -> pd.DataFrame:
        all_frames = []
        for definition in definitions:
            all_frames.append(self.run_definition(definition))
        if not all_frames:
            return pd.DataFrame()
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = self.results_dir / "um_all_results.csv"
        combined.to_csv(combined_path, index=False)
        return combined


def select_definitions(owner: str | None = None, task_name: str | None = None) -> list[ExperimentDefinition]:
    selected = UM_EXPERIMENTS
    if owner:
        selected = [d for d in selected if d.owner == owner]
    if task_name:
        selected = [d for d in selected if d.task_name == task_name]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uruchamianie eksperymentów UM w ramach części 2 projektu.")
    parser.add_argument("--owner", choices=["person2", "person3"], help="Uruchom tylko eksperymenty przypisane do danej osoby.")
    parser.add_argument("--task", choices=DEFAULT_UM_TASKS, help="Uruchom eksperymenty tylko dla jednego tasku.")
    parser.add_argument("--list", action="store_true", help="Pokaż listę eksperymentów bez uruchamiania.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    definitions = select_definitions(owner=args.owner, task_name=args.task)

    if args.list:
        for d in definitions:
            print(f"{d.owner}: {d.experiment_id} -> {d.tuned_param} = {d.values}")
        return

    if not definitions:
        raise SystemExit("Brak eksperymentów do uruchomienia dla podanych filtrów.")

    runner = UMExperimentRunner()
    combined = runner.run_many(definitions)
    print(f"Zapisano {len(combined)} rekordów do {runner.results_dir}")


if __name__ == "__main__":
    main()
