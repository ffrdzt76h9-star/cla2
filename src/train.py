import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
    }


def evaluate_model(model, X_test_final, y_test, model_name):
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n{model_name}")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    return {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1": f1, "ROC_AUC": roc_auc, "fpr": fpr, "tpr": tpr
    }


def run_training():
    print("Starting Model Training...")

    # 1. Load processed training/test data
    X_train_final = np.load('data/processed/X_train_final.npy')
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # 2. Train & compare every candidate model
    models = get_models()
    metrics_results = {}
    trained_models = {}
    roc_data = {}

    for model_name, model in models.items():
        model.fit(X_train_final, y_train)
        result = evaluate_model(model, X_test_final, y_test, model_name)

        metrics_results[model_name] = {
            "Accuracy": result["Accuracy"], "Precision": result["Precision"],
            "Recall": result["Recall"], "F1": result["F1"], "ROC_AUC": result["ROC_AUC"]
        }
        roc_data[model_name] = {"fpr": result["fpr"], "tpr": result["tpr"], "auc": result["ROC_AUC"]}
        trained_models[model_name] = model

    results_df = pd.DataFrame(metrics_results).T
    results_df = results_df.sort_values(by="F1", ascending=False)

    # 3. Best Model Selection
    best_model_name = results_df["F1"].idxmax()
    best_model = trained_models[best_model_name]

    print("\nModel comparison (sorted by F1):")
    print(results_df)
    print("\nBest Model:", best_model_name)

    # 4. Save Artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(roc_data, 'models/roc_data.pkl')
    results_df.to_csv('outputs/model_comparison.csv')

    with open('models/best_model_name.txt', 'w') as f:
        f.write(best_model_name)

    print("Model training complete. Best model saved to disk!")


if __name__ == "__main__":
    run_training()
