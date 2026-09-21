import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/insurance_claims_clean.csv")

print("Overall review flag rate:", df["fraud_review_flag"].mean())

print("\nReview rate by policy type:")
print(df.groupby("policy_type")["fraud_review_flag"].mean().sort_values(ascending=False))

print("\nReview rate by reporting delay:")
print(df.groupby("report_delay_band")["fraud_review_flag"].mean().sort_values(ascending=False))

plt.figure(figsize=(7, 5))
df["fraud_review_flag"].value_counts().sort_index().plot(kind="bar")
plt.title("Fraud Review Flag Distribution")
plt.xlabel("Flag")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()
