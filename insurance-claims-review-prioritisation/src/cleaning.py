"""Clean the raw claims data and engineer business-oriented features."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/insurance_claims.csv"
OUT = ROOT / "data/processed/insurance_claims_clean.csv"

NUMERIC_COLUMNS = [
    "claim_amount", "annual_premium", "previous_claims", "days_to_report",
    "documents_submitted", "customer_service_calls", "incident_hour",
    "repair_estimate_variance",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["claim_to_premium_ratio"] = df["claim_amount"] / df["annual_premium"].replace(0, float("nan"))
    df["report_delay_band"] = pd.cut(
        df["days_to_report"], bins=[-1, 1, 7, 30], labels=["0-1 days", "2-7 days", "8-30 days"])
    df["claim_amount_band"] = pd.cut(
        df["claim_amount"], bins=[-float("inf"), 10000, 30000, 75000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"])
    df["high_claim_ratio"] = (df["claim_to_premium_ratio"] > 4).astype("Int64")
    df.loc[df["claim_to_premium_ratio"].isna(), "high_claim_ratio"] = pd.NA
    return df


if __name__ == "__main__":
    cleaned = clean_data(pd.read_csv(RAW))
    cleaned.to_csv(OUT, index=False)
    print(cleaned.head())
    print("\nMissing values retained for train-only pipeline imputation:", int(cleaned.isna().sum().sum()))
