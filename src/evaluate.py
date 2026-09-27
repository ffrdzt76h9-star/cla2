import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained best model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    best_model = joblib.load('models/best_model.pkl')

    with open('models/best_model_name.txt') as f:
        best_model_name = f.read().strip()

    roc_data = joblib.load('models/roc_data.pkl')
    results_df = pd.read_csv('outputs/model_comparison.csv', index_col=0)

    # 2. Make predictions
    y_pred = best_model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation Report ---")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    os.makedirs('outputs', exist_ok=True)

    # 3. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Disease", "Disease"]
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    # 4. ROC-AUC Comparison across all trained models
    plt.figure(figsize=(9, 6))
    for model_name, data in roc_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f"{model_name} (AUC = {data['auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Comparison of ML Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_auc_comparison.png')
    plt.close()

    # 5. Accuracy Comparison across all trained models
    plt.figure(figsize=(9, 5))
    plt.bar(results_df.index, results_df["Accuracy"])
    plt.title("Accuracy Comparison of ML Models")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png')
    plt.close()

    # 6. Error Analysis Export (false negatives / false positives)
    # Reload raw data so we keep the original (unscaled) feature values
    raw_df = pd.read_csv('data/raw/heart_disease_uci.csv')
    raw_df['target'] = (raw_df['num'] > 0).astype(int)
    raw_df = raw_df.drop(['num', 'id', 'dataset'], axis=1)

    from sklearn.model_selection import train_test_split
    X = raw_df.drop('target', axis=1)
    y = raw_df['target']
    _, X_test_raw, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mismatched = X_test_raw.copy()
    mismatched["actual"] = y_test
    mismatched["predicted"] = y_pred

    false_negatives = mismatched[(mismatched["actual"] == 1) & (mismatched["predicted"] == 0)]
    false_positives = mismatched[(mismatched["actual"] == 0) & (mismatched["predicted"] == 1)]

    false_negatives.to_csv('outputs/false_negatives.csv', index=False)
    false_positives.to_csv('outputs/false_positives.csv', index=False)

    print("Evaluation complete. Plots and error analysis files saved to outputs/!")


if __name__ == "__main__":
    run_evaluation()
