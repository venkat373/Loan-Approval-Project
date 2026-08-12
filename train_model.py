"""
Train the loan-approval model and save everything the app needs to make
predictions on new, raw user input: the model, the scaler, the label
encoders, and the exact feature column order.

Run this once (locally or wherever you have the dataset):
    python train_model.py

It expects `loan_approval_dataset.csv` in the same folder (the same file
used in the notebook). It writes artifacts/model.pkl, artifacts/scaler.pkl,
artifacts/encoders.pkl, artifacts/feature_columns.pkl, and
artifacts/metrics.json.
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "loan_approval_dataset.csv"
ARTIFACTS_DIR = "artifacts"

CATEGORICAL_COLS = ["education", "self_employed"]
SCALE_COLS = [
    "income_annum",
    "loan_amount",
    "loan_term_years",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.rename(
        columns={
            "loan_term": "loan_term_years",
            "bank_assets_value": "bank_asset_value",
        },
        inplace=True,
    )
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # loan_id is just a row identifier, not a predictive feature -> drop it.
    # (The notebook accidentally left it in x_train; we intentionally
    # exclude it here so the model isn't memorizing IDs.)
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])
    return df


def main():
    df = load_and_clean(DATA_PATH)
    print("Rows, columns:", df.shape)
    print(df.isnull().sum())

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df["loan_status"] = target_encoder.fit_transform(df["loan_status"])
    # Make sure "Approved" maps to 1 for a sane, human-readable prediction.
    approved_idx = list(target_encoder.classes_).index("Approved")
    if approved_idx != 1:
        df["loan_status"] = 1 - df["loan_status"]
    encoders["loan_status"] = target_encoder

    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    x = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    feature_columns = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Best params:", grid.best_params_)
    print("Test accuracy:", acc)
    print(classification_report(y_test, y_pred))

    import os
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, f"{ARTIFACTS_DIR}/model.pkl")
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{ARTIFACTS_DIR}/encoders.pkl")
    joblib.dump(feature_columns, f"{ARTIFACTS_DIR}/feature_columns.pkl")
    joblib.dump(SCALE_COLS, f"{ARTIFACTS_DIR}/scale_columns.pkl")

    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump({"test_accuracy": acc, "best_params": grid.best_params_,
                    "report": report}, f, indent=2)

    print(f"\nSaved model + preprocessing artifacts to ./{ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
