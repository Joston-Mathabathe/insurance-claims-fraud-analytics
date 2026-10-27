"""Translate the review-queue ranking into workload tiers and a financial case.

ASSUMPTIONS (illustrative - the data is synthetic; currency units are labelled "R"):
- Each claim reviewed costs R1,000 of investigator time.
- For scenario analysis ONLY, a reviewed claim with a synthetic flag is assumed to yield 25% of claim value. This is NOT an empirically supported recovery rate.
Both are varied in reports/review_economics_sensitivity.csv.

Three illustrative ways to build the review queue are compared on the same out-of-fold scores (in-sample policy exploration, not independent policy validation):
random review, ranking by model score, and ranking by expected recoverable value
(score x claim amount), which is how a claims manager would actually prioritise.

Outputs: reports/review_tiers.csv, review_economics.csv, review_economics_sensitivity.csv,
         rule_indicator.csv ; visualisations/review_net_benefit.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
COST_PER_REVIEW = 1000
RECOVERY_SHARE = 0.25
TIER_CUTS = {"Priority review": 0.05, "Standard review": 0.20}   # cumulative queue shares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = ROOT / "visualisations"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/insurance_claims_clean.csv")
T = "fraud_review_flag"
features = [
    "customer_age", "policy_tenure_years", "policy_type", "annual_premium", "claim_amount",
    "previous_claims", "days_to_report", "documents_submitted", "customer_service_calls",
    "incident_hour", "repair_estimate_variance", "new_policy", "claim_to_premium_ratio",
    "high_claim_ratio",
]
categorical = ["policy_type"]
numeric = [c for c in features if c not in categorical]
pipe = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])),
    ("model", LogisticRegression(max_iter=2000)),
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
df["score"] = cross_val_predict(pipe, df[features], df[T], cv=skf, method="predict_proba")[:, 1]
df["expected_value"] = df["score"] * df["claim_amount"].fillna(0)
# Missing claim amounts have unknown financial exposure; zero is a conservative
# scenario placeholder, not an imputation for model training.
df["flagged_value"] = df["claim_amount"].fillna(0) * df[T]
n, base = len(df), df[T].mean()
total_flags, total_value = df[T].sum(), df["flagged_value"].sum()

# ------------------------------------------------------------ 1. Workload tiers (value-aware queue)
ranked = df.sort_values("expected_value", ascending=False).reset_index(drop=True)
edges = [0] + [int(n * c) for c in TIER_CUTS.values()] + [n]
labels = list(TIER_CUTS) + ["Fast-track (no extra review)"]
actions = {
    "Priority review": "Senior investigator, full review before payment",
    "Standard review": "Standard desk review or sampled checks",
    "Fast-track (no extra review)": "Normal straight-through processing",
}
rows = []
for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
    s = ranked.iloc[lo:hi]
    rows.append({
        "tier": lab, "claims": len(s), "share_of_claims": len(s) / n,
        "flags": int(s[T].sum()), "flag_rate": s[T].mean(),
        "lift_vs_overall": s[T].mean() / base,
        "share_of_all_flags": s[T].sum() / total_flags,
        "share_of_flagged_claim_value": s["flagged_value"].sum() / total_value,
        "recommended_action": actions[lab],
    })
tiers = pd.DataFrame(rows)
tiers.to_csv(OUT / "review_tiers.csv", index=False)

# ------------------------------------------------------------ 2. Queue economics
shares = np.round(np.arange(0.01, 1.001, 0.01), 2)


def curve(order_col, cost=COST_PER_REVIEW, rec=RECOVERY_SHARE, rng=None):
    if order_col == "random":                      # expected result of reviewing a random share
        return np.array([rec * total_value * s - cost * int(n * s) for s in shares])
    r = df.sort_values(order_col, ascending=False)
    cum_val = r["flagged_value"].cumsum().to_numpy()
    return np.array([rec * cum_val[int(n * s) - 1] - cost * int(n * s) for s in shares])


strategies = {"Random review": "random", "Ranked by model score": "score",
              "Ranked by expected recoverable value": "expected_value"}
econ_rows, curves = [], {}
for label, col in strategies.items():
    c = curve(col)
    curves[label] = c
    i = int(np.argmax(c))
    econ_rows.append({"strategy": label, "best_queue_share": shares[i],
                      "claims_reviewed": int(n * shares[i]), "hypothetical_net_benefit_R": c[i]})
econ = pd.DataFrame(econ_rows)
econ.to_csv(OUT / "review_economics.csv", index=False)

sens = []
for cost in (500, 1000, 1500):
    for rec in (0.15, 0.25, 0.35):
        for label, col in strategies.items():
            c = curve(col, cost, rec)
            i = int(np.argmax(c))
            sens.append({"cost_per_review_R": cost, "recovery_share": rec, "strategy": label,
                         "best_queue_share": shares[i], "hypothetical_net_benefit_R": max(c[i], 0.0),
                         "pays_off": bool(c[i] > 0)})
pd.DataFrame(sens).to_csv(OUT / "review_economics_sensitivity.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for label, c in curves.items():
    ax.plot(shares * 100, c / 1e6, label=label)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Share of claims reviewed (%)")
ax.set_ylabel("Hypothetical net benefit (R million)")
ax.set_title(f"Illustrative review-queue economics (R{COST_PER_REVIEW:,} per review, {int(RECOVERY_SHARE*100)}% recovery)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "review_net_benefit.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 3. Pre-specified multi-indicator rule (from sql/)
rule = (df["claim_to_premium_ratio"] > 4) & (df["previous_claims"] >= 3) & (df["repair_estimate_variance"] > 0.35)
k, hits = int(rule.sum()), int(df.loc[rule, T].sum())
p = hits / k
z = 1.96
centre = (p + z**2 / (2 * k)) / (1 + z**2 / k)
half = z * np.sqrt(p * (1 - p) / k + z**2 / (4 * k**2)) / (1 + z**2 / k)
pd.DataFrame([{"claims_matching_rule": k, "flagged": hits, "flag_rate": p, "overall_flag_rate": base,
               "lift_vs_overall": p / base, "wilson_95_ci_low": centre - half,
               "wilson_95_ci_high": centre + half}]).to_csv(OUT / "rule_indicator.csv", index=False)

pd.set_option("display.width", 220)
print("Tiers\n", tiers.drop(columns="recommended_action").round(3).to_string(index=False))
print("\nEconomics (base)\n", econ.round(0).to_string(index=False))
s = pd.DataFrame(sens)
print("\nSensitivity - best strategy net benefit R by scenario")
print(s.pivot_table(index=["cost_per_review_R", "recovery_share"], columns="strategy", values="hypothetical_net_benefit_R").round(0).to_string())
print("\nRule\n", pd.read_csv(OUT / "rule_indicator.csv").round(3).to_string(index=False))
print("\nTotal flagged claim value R:", round(total_value), " avg claim R:", round(df.claim_amount.mean()))
