import argparse
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.ml_experiments import load_processed_dataset
RESULTS_DIR = Path("results/um")
PLOTS_DIR = RESULTS_DIR / "plots"

def run_feature_importance():
    print("Ładowanie danych do klasyfikacji rozwodów...")
    X_train, X_test, y_train, y_test, metadata = load_processed_dataset("classification_divorce")
    
    feature_names = metadata.get("feature_names_after_preprocessing", [f"Feature_{i}" for i in range(X_train.shape[1])])
    
    print("Trenowanie algorytmu Random Forest")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Pobieranie wskaźników Feature Importance")
    importances = rf.feature_importances_

    clean_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
    indices = np.argsort(importances)[::-1]
    sorted_names = [clean_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    csv_path = RESULTS_DIR / "classification_divorce_feature_importance.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Importance'])
        for name, imp in zip(sorted_names, sorted_importances):
            writer.writerow([name, imp])
    print(f"Zapisano CSV: {csv_path}")
    
    top_n = 10
    top_names = sorted_names[:top_n]
    top_imps = sorted_importances[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_names[::-1], top_imps[::-1], color='darkred')
    plt.title(f"Top {top_n} najważniejszych czynników decydujących o rozwodzie\n(Algorytm: Random Forest)")
    plt.xlabel("Waga Cechy")
    plt.ylabel("Cecha demograficzna")
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "classification_divorce_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano Wykres: {plot_path}")
    
if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_feature_importance()
