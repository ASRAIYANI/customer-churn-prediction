
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load and validate data
df = pd.read_csv(DATA_PATH)

if df["churn"].isna().any():
    raise ValueError("The churn target contains missing values.")

if set(df["churn"].unique()) != {0, 1}:
    raise ValueError("Expected churn labels 0 and 1.")

X = df.drop(columns=["churn", "customer_id"])
y = df["churn"].astype(int)

# Split: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Rows:", len(X_train), "train |", len(X_val),
      "validation |", len(X_test), "test")

# Identify feature types using training data
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Preprocessing is fitted on training data only
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# Train models and compare on validation data
models = {}
validation_results = []

for name, estimator in candidates.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_prob = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, val_prob),
    }

    validation_results.append(metrics)
    models[name] = pipeline

    print(f"\nValidation results: {name}")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

# Choose model using validation F1 only
validation_df = pd.DataFrame(validation_results)
validation_df = validation_df.sort_values("f1", ascending=False)

best_name = validation_df.iloc[0]["model"]
best_model = models[best_name]

# Evaluate the selected model on the untouched test set
test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "selected_model": best_name,
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall": recall_score(y_test, test_pred, zero_division=0),
    "f1": f1_score(y_test, test_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, test_prob),
}

print("\n===== VALIDATION MODEL COMPARISON =====")
print(validation_df.to_string(index=False))

print("\n===== FINAL TEST RESULTS =====")
for key, value in test_metrics.items():
    if key != "selected_model":
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nTest confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_test, test_pred))

print("\nTest classification report:")
print(classification_report(y_test, test_pred, zero_division=0))

# Save model and reports
joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")

validation_df.to_csv(
    REPORT_DIR / "validation_model_comparison.csv", index=False
)

with open(REPORT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

metadata = {
    "selected_model": best_name,
    "target_column": "churn",
    "excluded_columns": ["customer_id"],
    "feature_columns": X.columns.tolist(),
    "positive_class": 1,
    "random_state": 42,
    "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
    "selection_metric": "validation F1",
}

with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model:", MODEL_DIR / "churn_model.joblib")
print("Saved validation comparison:", REPORT_DIR / "validation_model_comparison.csv")
print("Saved final test metrics:", REPORT_DIR / "test_metrics.json")
