
from pathlib import Path
import pandas as pd

# Locate the project directory and dataset
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print(f"Number of customers: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n===== FIRST 5 CUSTOMERS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== MISSING VALUE PERCENTAGES =====")
print((df.isnull().mean() * 100).round(2))

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== CHURN COUNTS =====")
print(df["churn"].value_counts(dropna=False))

print("\n===== CHURN PERCENTAGES =====")
print(
    (df["churn"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)

print("\n===== NUMERIC SUMMARY =====")
print(df.describe())

print("\n===== CATEGORICAL SUMMARY =====")
print(df.describe(include="object"))
