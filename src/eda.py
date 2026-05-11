from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import RAW_DATA_PATH, RESULTS_DIR


def main() -> None:
    output_dir = RESULTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "divorce_status_distribution": df["Divorce_Status"].value_counts(normalize=True).round(4).to_dict(),
        "marital_satisfaction_distribution": df["Marital_Satisfaction"].value_counts(normalize=True).round(4).to_dict(),
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fig_specs = [
        ("Divorce_Status", "bar", "divorce_status_balance.png", "Rozkład klas: Divorce_Status"),
        ("Marital_Satisfaction", "bar", "marital_satisfaction_balance.png", "Rozkład klas: Marital_Satisfaction"),
        ("Children_Count", "hist", "children_count_hist.png", "Rozkład Children_Count"),
        ("Years_Since_Marriage", "hist", "years_since_marriage_hist.png", "Rozkład Years_Since_Marriage"),
        ("Age_at_Marriage", "hist", "age_at_marriage_hist.png", "Rozkład Age_at_Marriage"),
    ]

    for col, kind, filename, title in fig_specs:
        plt.figure(figsize=(7, 4))
        if kind == "bar":
            df[col].value_counts().plot(kind="bar")
            plt.xlabel(col)
            plt.ylabel("Liczba obserwacji")
        else:
            df[col].hist(bins=20)
            plt.xlabel(col)
            plt.ylabel("Liczba obserwacji")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    print(f"EDA zapisane do: {output_dir}")


if __name__ == "__main__":
    main()
