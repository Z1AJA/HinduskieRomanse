from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "marriage_data_india.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20

TASK_CONFIGS = {
    "classification_divorce": {
        "problem_type": "classification",
        "target": "Divorce_Status",
        "drop_columns": ["ID"],
        "target_mapping": {"No": 0, "Yes": 1},
        "stratify": True,
        "recommended_metrics": ["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
        "notes": "Mocno niezbalansowany target. Sam accuracy nie wystarczy."
    },
    "classification_satisfaction": {
        "problem_type": "classification",
        "target": "Marital_Satisfaction",
        "drop_columns": ["ID"],
        "target_mapping": {"Low": 0, "Medium": 1, "High": 2},
        "stratify": True,
        "recommended_metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
        "notes": "Opcjonalny task wieloklasowy."
    },
        "regression_years": {
        "problem_type": "regression",
        "target": "Years_Since_Marriage",
        "drop_columns": ["ID"],
        "stratify": False,
        "recommended_metrics": ["mse", "rmse", "mae", "r2"],
        "notes": "Główny target regresyjny w projekcie."
    },
}

DEFAULT_TASKS = {
    "classification": "classification_divorce",
    "regression": "regression_years",
}
