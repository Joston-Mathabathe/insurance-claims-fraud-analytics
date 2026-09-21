import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/insurance_claims.csv")

numeric_columns = [
    "claim_amount", "annual_premium", "previous_claims",
    "days_to_report", "documents_submitted",
    "customer_service_calls", "incident_hour",
    "repair_estimate_variance"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].median())

df["claim_to_premium_ratio"] = df["claim_amount"] / df["annual_premium"]

df["report_delay_band"] = pd.cut(
    df["days_to_report"],
    bins=[-1, 1, 7, 30],
    labels=["0-1 days", "2-7 days", "8-30 days"]
)

df["claim_amount_band"] = pd.qcut(
    df["claim_amount"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"]
)

df["high_claim_ratio"] = (df["claim_to_premium_ratio"] > 4).astype(int)

df.to_csv("data/processed/insurance_claims_clean.csv", index=False)

print(df.head())
print(df.isna().sum())
