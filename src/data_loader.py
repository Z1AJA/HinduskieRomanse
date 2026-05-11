from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_TEST_SIZE, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_PATH, TASK_CONFIGS


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]
    preprocessor: ColumnTransformer


class ProjectDataLoader:
    """
    Jedno źródło prawdy dla przygotowania danych w projekcie.
    Zasada: reszta zespołu importuje to miejsce, a nie robi własnych wariacji preprocessingu.
    """

    REQUIRED_COLUMNS = [
        "ID",
        "Marriage_Type",
        "Age_at_Marriage",
        "Gender",
        "Education_Level",
        "Caste_Match",
        "Religion",
        "Parental_Approval",
        "Urban_Rural",
        "Dowry_Exchanged",
        "Marital_Satisfaction",
        "Divorce_Status",
        "Children_Count",
        "Income_Level",
        "Years_Since_Marriage",
        "Spouse_Working",
        "Inter-Caste",
        "Inter-Religion",
    ]

    def __init__(self, csv_path: Path | str = RAW_DATA_PATH) -> None:
        self.csv_path = Path(csv_path)

    def load_raw_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        self.validate_dataframe(df)
        return df

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Brakuje wymaganych kolumn: {missing}")

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Nieznany task: {task_name}. Dostępne: {list(TASK_CONFIGS)}")
        return TASK_CONFIGS[task_name]

    def get_features_and_target(self, df: pd.DataFrame, task_name: str) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        cfg = self.get_task_config(task_name)
        target_col = cfg["target"]
        drop_columns = set(cfg.get("drop_columns", [])) | {target_col}

        X = df[[c for c in df.columns if c not in drop_columns]].copy()
        y = df[target_col].copy()

        if cfg["problem_type"] == "classification":
            mapping = cfg["target_mapping"]
            y = y.map(mapping)
            if y.isna().any():
                raise ValueError("Mapowanie targetu dało NaN. Sprawdź wartości klas.")
            y = y.astype(int)

        return X, y, cfg

    def build_preprocessor(self, X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(steps=num_steps)
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        return preprocessor

    def prepare(
        self,
        task_name: str,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = RANDOM_STATE,
        scale_numeric: bool = True,
    ) -> PreparedDataset:
        df = self.load_raw_data()
        X, y, cfg = self.get_features_and_target(df, task_name)
        stratify = y if cfg.get("stratify", False) else None

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        preprocessor = self.build_preprocessor(X_train_df, scale_numeric=scale_numeric)
        X_train = preprocessor.fit_transform(X_train_df).astype(np.float32)
        X_test = preprocessor.transform(X_test_df).astype(np.float32)

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)

        if cfg["problem_type"] == "regression":
            y_train_array = y_train_array.astype(np.float32).reshape(-1, 1)
            y_test_array = y_test_array.astype(np.float32).reshape(-1, 1)
        else:
            y_train_array = y_train_array.astype(np.int64)
            y_test_array = y_test_array.astype(np.int64)

        metadata = {
            "task_name": task_name,
            "problem_type": cfg["problem_type"],
            "target": cfg["target"],
            "target_mapping": cfg.get("target_mapping"),
            "test_size": test_size,
            "random_state": random_state,
            "scale_numeric": scale_numeric,
            "input_dim": int(X_train.shape[1]),
            "train_size": int(X_train.shape[0]),
            "test_size_rows": int(X_test.shape[0]),
            "original_features": X.columns.tolist(),
            "feature_names_after_preprocessing": feature_names,
            "recommended_metrics": cfg.get("recommended_metrics", []),
            "notes": cfg.get("notes", ""),
        }

        if cfg["problem_type"] == "classification":
            train_unique, train_counts = np.unique(y_train_array, return_counts=True)
            test_unique, test_counts = np.unique(y_test_array, return_counts=True)
            metadata["class_distribution_train"] = {int(k): int(v) for k, v in zip(train_unique, train_counts)}
            metadata["class_distribution_test"] = {int(k): int(v) for k, v in zip(test_unique, test_counts)}

        return PreparedDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train_array,
            y_test=y_test_array,
            feature_names=feature_names,
            metadata=metadata,
            preprocessor=preprocessor,
        )

    def save_prepared_dataset(
        self,
        prepared: PreparedDataset,
        output_dir: Path | str = PROCESSED_DATA_DIR,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        task_name = prepared.metadata["task_name"]
        npz_path = output_dir / f"{task_name}.npz"
        metadata_path = output_dir / f"{task_name}_metadata.json"
        features_path = output_dir / f"{task_name}_feature_names.txt"

        np.savez_compressed(
            npz_path,
            X_train=prepared.X_train,
            X_test=prepared.X_test,
            y_train=prepared.y_train,
            y_test=prepared.y_test,
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(prepared.metadata, f, indent=2, ensure_ascii=False)

        with open(features_path, "w", encoding="utf-8") as f:
            for name in prepared.feature_names:
                f.write(name + "\n")

        return {
            "npz_path": npz_path,
            "metadata_path": metadata_path,
            "features_path": features_path,
        }


def load_npz_dataset(path: Path | str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}
