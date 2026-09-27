import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    data_path = 'data/raw/heart_disease_uci.csv'
    df = pd.read_csv(data_path)

    # 2. Creating the Target Variable
    # 'num' ranges 0-4 in the raw data; turn it into a binary target
    df['target'] = (df['num'] > 0).astype(int)

    # 3. Data Cleaning
    df = df.drop(['num', 'id', 'dataset'], axis=1)

    X = df.drop('target', axis=1)
    y = df['target']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Separate Column Types
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns

    # 6. Impute Missing Values (median for numeric, mode for categorical)
    for col in num_cols:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

    for col in cat_cols:
        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

    # 7. Scale & Encode
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_train_encoded = ohe.fit_transform(X_train[cat_cols])
    X_test_encoded = ohe.transform(X_test[cat_cols])

    # Combine Features
    X_train_final = np.hstack((X_train_scaled, X_train_encoded))
    X_test_final = np.hstack((X_test_scaled, X_test_encoded))

    # 8. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')

    # Save Metadata
    metadata = {
        "dataset_name": "Heart Disease UCI",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing complete. Processed data and preprocessors saved to disk!")


if __name__ == "__main__":
    run_preprocessing()
