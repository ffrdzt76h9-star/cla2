import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def run_mlflow_training():
    print("[INFO] --- Starting MLflow Experiment Tracking Run ---")

    # 1. Set up MLflow experiment
    mlflow.set_experiment("Heart_Disease_Prediction")

    # 2. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # 3. Define Model Parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    with mlflow.start_run(run_name="RandomForest_Baseline_Tracking"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 4. Train Model
        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_metrics(metrics)

        # 6. Log the preprocessors so runs are fully reproducible
        mlflow.log_artifact("models/scaler.pkl", artifact_path="preprocessing")
        mlflow.log_artifact("models/ohe.pkl", artifact_path="preprocessing")

        # 7. Log the trained model
        mlflow.sklearn.log_model(sk_model=rf_model, name="random_forest_model")

        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print("[SUCCESS] Run tracked in MLflow experiment 'Heart_Disease_Prediction'.")


if __name__ == "__main__":
    run_mlflow_training()
