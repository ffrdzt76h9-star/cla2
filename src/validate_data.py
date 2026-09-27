import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_heart_disease_schema():
    """Defines the strict schema specification for the Heart Disease UCI dataset."""
    return pa.DataFrameSchema({
        "id": Column(pa.Int, Check.ge(0)),
        "age": Column(pa.Int, Check.in_range(0, 120)),
        "sex": Column(pa.String, Check.isin(["Male", "Female"])),
        "dataset": Column(pa.String, Check.isin(
            ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
        )),
        "cp": Column(pa.String, Check.isin(
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
        )),
        "trestbps": Column(pa.Float, Check.ge(0.0), nullable=True),
        "chol": Column(pa.Float, Check.ge(0.0), nullable=True),
        "fbs": Column(pa.Bool, nullable=True),
        "restecg": Column(pa.String, Check.isin(
            ["normal", "st-t abnormality", "lv hypertrophy"]
        ), nullable=True),
        "thalch": Column(pa.Float, Check.ge(0.0), nullable=True),
        "exang": Column(pa.Bool, nullable=True),
        "oldpeak": Column(pa.Float, nullable=True),
        "slope": Column(pa.String, Check.isin(
            ["upsloping", "flat", "downsloping"]
        ), nullable=True),
        "ca": Column(pa.Float, Check.in_range(0.0, 3.0), nullable=True),
        "thal": Column(pa.String, Check.isin(
            ["normal", "fixed defect", "reversable defect"]
        ), nullable=True),
        "num": Column(pa.Int, Check.in_range(0, 4))
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_heart_disease_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    # When run directly, it only validates the raw production data
    data_path = 'data/raw/heart_disease_uci.csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
