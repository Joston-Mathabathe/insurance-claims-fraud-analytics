"""Exploratory analysis: where do review flags concentrate? Saves tables and charts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/insurance_claims_clean.csv")
OUT = ROOT / "reports"
FIG = ROOT / "visualisations"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
T = "fraud_review_flag"
base = df[T].mean()
print(f"Overall review-flag rate: {base:.2%} ({int(df[T].sum())} of {len(df):,} claims)")

df["previous_claims_band"] = pd.cut(df["previous_claims"], [-1, 0, 2, 10], labels=["0", "1-2", "3+"])
segments = {
    "policy_type": "review_rate_by_policy.csv",
    "report_delay_band": "review_rate_by_reporting_delay.csv",
    "claim_amount_band": "review_rate_by_claim_amount.csv",
    "high_claim_ratio": "review_rate_by_claim_ratio.csv",
    "previous_claims_band": "review_rate_by_previous_claims.csv",
    "new_policy": "review_rate_by_new_policy.csv",
}
for col, fname in segments.items():
    tbl = (df.groupby(col, observed=True)[T].agg(claims="size", flagged="sum", review_flag_rate="mean")
           .reset_index())
    tbl["lift_vs_overall"] = tbl["review_flag_rate"] / base
    tbl.to_csv(OUT / fname, index=False)
    print(f"\n{col}\n{tbl.round(3).to_string(index=False)}")


def bar(col, fname, title, xlabel):
    tbl = df.groupby(col, observed=True)[T].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tbl.plot(kind="bar", ax=ax, color="#2e86c1")
    ax.axhline(base * 100, color="red", linestyle="--", label=f"Overall {base:.1%}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Review-flag rate (%)")
    ax.legend()
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


bar("policy_type", "review_rate_by_policy.png", "Review-flag rate by policy type", "Policy type")
bar("report_delay_band", "review_rate_by_reporting_delay.png", "Review-flag rate by reporting delay", "Days to report")
bar("claim_amount_band", "review_rate_by_claim_amount.png", "Review-flag rate by claim amount", "Claim amount band (fixed R thresholds)")
bar("high_claim_ratio", "review_rate_by_claim_ratio.png", "Review-flag rate: claim above 4x annual premium?", "High claim-to-premium ratio (1 = yes)")

fig, ax = plt.subplots(figsize=(6, 4.5))
df[T].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#7f8c8d", "#c0392b"])
ax.set_title("Review-flag distribution (class imbalance)")
ax.set_xlabel("Flag (1 = flagged for review)")
ax.set_ylabel("Number of claims")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(FIG / "fraud_review_distribution.png", dpi=160)
plt.close(fig)
