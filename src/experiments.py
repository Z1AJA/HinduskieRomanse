
from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import PROCESSED_DATA_DIR, RESULTS_DIR, TASK_CONFIGS
from .data_loader import load_npz_dataset
from .metrics import accuracy, mse
from .neural_network import NeuralNetwork


def get_default_results_path(task_name: str) -> Path:
    output_dir = RESULTS_DIR / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{task_name}_results.csv"


def run_experiments(task_name: str, experiment_grid: dict[str, list[Any]], n_repeats: int = 3) -> None:
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Nieznany task: {task_name}")

    problem_type = TASK_CONFIGS[task_name]["problem_type"]
    npz_path = PROCESSED_DATA_DIR / f"{task_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Brak pliku z danymi: {npz_path}. Uruchom najpierw run_data_prep.py.")

    data = load_npz_dataset(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    input_dim = X_train.shape[1]
    csv_path = get_default_results_path(task_name)
    write_header = not csv_path.exists()

    keys = list(experiment_grid.keys())
    values = list(experiment_grid.values())
    combinations = list(itertools.product(*values))

    run_group_id = int(time.time())

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "task", "run_id", "param_name", "param_value",
                "train_metric", "test_metric",
                "epochs", "learning_rate", "batch_size",
                "hidden_layers", "hidden_units"
            ])

        for combo in combinations:
            params = dict(zip(keys, combo))

            epochs = params.get("epochs", 100)
            learning_rate = params.get("learning_rate", 0.01)
            batch_size = params.get("batch_size", 32)
            activation = params.get("activation", "relu")
            weight_init = params.get("weight_init", "xavier")
            hidden_layer_sizes = params.get("hidden_layer_sizes", [16, 8])

            output_dim = 1
            layer_sizes = [input_dim] + list(hidden_layer_sizes) + [output_dim]
            hidden_layers_count = len(hidden_layer_sizes)
            hidden_units_str = "-".join(map(str, hidden_layer_sizes))

            if len(keys) == 1:
                param_name = keys[0]
                param_value = str(combo[0])
            else:
                param_name = "grid"
                param_value = str(params)

            for repeat in range(n_repeats):
                seed = 42 + repeat
                model = NeuralNetwork(
                    layer_sizes=layer_sizes,
                    problem_type=problem_type,
                    activation=activation,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    weight_init=weight_init,
                    random_state=seed,
                )

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                if problem_type == "classification":
                    train_metric = accuracy(y_train, y_train_pred)
                    test_metric = accuracy(y_test, y_test_pred)
                else:
                    train_metric = mse(y_train, y_train_pred)
                    test_metric = mse(y_test, y_test_pred)

                writer.writerow([
                    task_name,
                    f"{run_group_id}_{repeat}",
                    param_name,
                    param_value,
                    train_metric,
                    test_metric,
                    epochs,
                    learning_rate,
                    batch_size,
                    hidden_layers_count,
                    hidden_units_str
                ])
                f.flush()
                print(f"[{task_name}] Powtórzenie {repeat+1}/{n_repeats} | "
                      f"Parametr: {param_name}={param_value} | "
                      f"Train: {train_metric:.4f}, Test: {test_metric:.4f}")


def generate_plots(task_name: str) -> None:
    csv_path = get_default_results_path(task_name)
    if not csv_path.exists():
        print(f"Brak pliku z wynikami dla {task_name}. Nie generuję wykresów.")
        return

    df = pd.read_csv(csv_path)

    for param in df['param_name'].unique():
        if param == "grid":
            continue

        subset = df[df['param_name'] == param]
        grouped = subset.groupby('param_value')[['train_metric', 'test_metric']].mean().reset_index()

        try:
            grouped['param_value_num'] = pd.to_numeric(grouped['param_value'])
            grouped = grouped.sort_values('param_value_num')
        except ValueError:
            grouped = grouped.sort_values('param_value')

        plt.figure(figsize=(8, 5))
        plt.plot(grouped['param_value'].astype(str), grouped['train_metric'], marker='o', label='Train')
        plt.plot(grouped['param_value'].astype(str), grouped['test_metric'], marker='s', label='Test')
        plt.title(f"Wpływ parametru {param} na wyniki ({task_name})")
        plt.xlabel(param)

        metric_name = "Accuracy" if "classification" in task_name else "MSE"
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_path = RESULTS_DIR / "experiments" / f"{task_name}_{param}_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Zapisano wykres: {plot_path}")


if __name__ == "__main__":
    tasks = ["classification_divorce", "regression_years"]
    
    EXPERIMENT_SCHEMAS_EXACT = {
        "test_hidden_layer": {"hidden_layer_sizes": [[8], [16, 8], [32, 16], [32, 16, 8]]},
        "test_epochs": {"epochs": [10, 50, 100, 200]},
        "test_batch_size": {"batch_size": [16, 32, 64, 128]},
        "test_learning_rate": {"learning_rate": [0.001, 0.01, 0.05, 0.1]}
    }
    
    for task in tasks:
        print(f"\nUruchamiam domknięcie eksperymentów dla tasku: {task}")
        for exp_name, grid in EXPERIMENT_SCHEMAS_EXACT.items():
            print(f"-> {exp_name}")
            try:
                run_experiments(task_name=task, experiment_grid=grid, n_repeats=3)
            except FileNotFoundError:
                print(f"UWAGA: Brakuje danych dla {task}. Odpal najpierw np. python3 -m src.run_data_prep --task {task}")
            
        generate_plots(task)
        
    print("\nEksperymenty dla obu zadań zakończone i zabezpieczone sukcesem!")
