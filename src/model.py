"""Model the synthetic review flag as a *prioritisation* problem.

With only ~4.8% of claims flagged, accuracy and a 0.5 threshold are not meaningful. Models are
therefore judged on how well they RANK claims: ROC-AUC, average precision (baseline = flag rate)
and how many flagged claims fall in the top 20% of the review queue. Everything uses out-of-fold
predictions so no claim is scored by a model that saw its own label.

Outputs (reports/): model_comparison.csv, cross_validation.csv, permutation_test.csv,
    logistic_odds_ratios.csv, lift_table.csv, decile_calibration.csv, project_summary.json
Charts (visualisations/): gains_chart.png, odds_ratios.png, decile_calibration.png
"""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
QUEUE_SHARE = 0.20           # review capacity used for the headline "top 20%" metric
N_PERMUTATIONS = int(os.getenv("FRAUD_PERMUTATIONS", "30"))  # set to 200 for a stronger significance check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = ROOT / "visualisations"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/insurance_claims_clean.csv")
target = "fraud_review_flag"
features = [
    "customer_age", "policy_tenure_years", "policy_type", "annual_premium", "claim_amount",
    "previous_claims", "days_to_report", "documents_submitted", "customer_service_calls",
    "incident_hour", "repair_estimate_variance", "new_policy", "claim_to_premium_ratio",
    "high_claim_ratio",
]
X, y = df[features], df[target]
categorical = ["policy_type"]
numeric = [c for c in features if c not in categorical]


def make_pipe(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", pre), ("model", model)])


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1),
    }


def capture_at(y_true, score, share=QUEUE_SHARE):
    """Share of all flagged claims found in the top `share` of claims ranked by score."""
    k = int(len(y_true) * share)
    order = np.argsort(-np.asarray(score), kind="stable")
    return float(np.asarray(y_true)[order[:k]].sum() / np.asarray(y_true).sum())


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ------------------------------------------------------------ 1. Development-only model selection
# The final test set is never used to choose a model, tune a queue, or fit a preprocessor.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
cv_rows, train_oof = [], {}
for name, model in make_models().items():
    pipe = make_pipe(model)
    auc = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="roc_auc")
    ap = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="average_precision")
    train_oof[name] = cross_val_predict(
        pipe, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    cv_rows.append({
        "model": name, "roc_auc_mean": auc.mean(), "roc_auc_std": auc.std(),
        "average_precision_mean": ap.mean(), "average_precision_std": ap.std(),
        f"flags_found_in_top_{int(QUEUE_SHARE*100)}pct": capture_at(y_tr, train_oof[name]),
    })
cv_df = pd.DataFrame(cv_rows).sort_values("roc_auc_mean", ascending=False)
cv_df["baseline_average_precision"] = y_tr.mean()
cv_df.to_csv(OUT / "cross_validation.csv", index=False)
best = cv_df.iloc[0]["model"]

# ------------------------------------------------------------ 2. Final untouched hold-out evaluation
selected_pipe = make_pipe(make_models()[best]).fit(X_tr, y_tr)
test_score = selected_pipe.predict_proba(X_te)[:, 1]
holdout = pd.DataFrame([{
    "model": best, "roc_auc": roc_auc_score(y_te, test_score),
    "average_precision": average_precision_score(y_te, test_score),
    f"flags_found_in_top_{int(QUEUE_SHARE*100)}pct": capture_at(y_te, test_score),
    "test_claims": len(y_te),
}])
holdout.to_csv(OUT / "model_comparison.csv", index=False)

# The all-data cross-fitted scores below are descriptive portfolio simulations,
# NOT independent validation. They are not used for final model selection.
score = cross_val_predict(
    make_pipe(make_models()[best]), X, y, cv=skf, method="predict_proba")[:, 1]

# ------------------------------------------------------------ 3. Is the signal real? Permutation test
rng = np.random.default_rng(RANDOM_STATE)
lr_pipe = make_pipe(make_models()["Logistic Regression"])
observed = cross_val_score(lr_pipe, X_tr, y_tr, cv=skf, scoring="roc_auc").mean()
perm = np.array([cross_val_score(lr_pipe, X_tr, rng.permutation(y_tr.to_numpy()), cv=skf, scoring="roc_auc").mean()
                 for _ in range(N_PERMUTATIONS)])
p_value = (1 + (perm >= observed).sum()) / (1 + N_PERMUTATIONS)
pd.DataFrame([{"observed_cv_roc_auc": observed, "shuffled_label_mean": perm.mean(),
               "shuffled_label_95th_percentile": np.percentile(perm, 95),
               "permutations": N_PERMUTATIONS, "p_value": p_value}]).to_csv(OUT / "permutation_test.csv", index=False)

# ------------------------------------------------------------ 4. Drivers (Logistic Regression)
fit = make_pipe(make_models()["Logistic Regression"]).fit(X_tr, y_tr)
names = [n.split("__", 1)[1] for n in fit.named_steps["preprocessor"].get_feature_names_out()]
coefs = fit.named_steps["model"].coef_[0]
odds = pd.DataFrame({"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
odds = odds.reindex(odds["coefficient"].abs().sort_values(ascending=False).index)
odds.to_csv(OUT / "logistic_odds_ratios.csv", index=False)
top = odds.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(top["feature"], top["odds_ratio"], color=["#c0392b" if v > 1 else "#2e86c1" for v in top["odds_ratio"]])
ax.axvline(1, color="black", linewidth=0.8)
ax.set_xlabel("Odds ratio per 1 SD (numeric) or vs other levels (categories)")
ax.set_title("Largest drivers of the review flag - Logistic Regression")
fig.tight_layout()
fig.savefig(FIG / "odds_ratios.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 5. Lift / gains (out-of-fold ranking)
rank = df.assign(score=score).sort_values("score", ascending=False).reset_index(drop=True)
n, total_flags = len(rank), rank[target].sum()
total_value = (rank["claim_amount"] * rank[target]).sum()
lift_rows = []
for share in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
    k = int(n * share)
    s = rank.iloc[:k]
    hits = int(s[target].sum())
    lift_rows.append({
        "queue_share": share, "claims_reviewed": k, "flags_found": hits,
        "share_of_flags_found": hits / total_flags, "precision": hits / k,
        "lift_vs_random": (hits / k) / y.mean(),
        "share_of_flagged_claim_value": (s["claim_amount"] * s[target]).sum() / total_value,
    })
lift = pd.DataFrame(lift_rows)
lift.to_csv(OUT / "lift_table.csv", index=False)

x = np.r_[0, [r["queue_share"] for r in lift_rows]]
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(x, np.r_[0, lift["share_of_flags_found"]], marker="o", label=f"{best} (out-of-fold)")
ax.plot([0, 1], [0, 1], "k--", label="Random review")
ax.set_xlabel("Share of claims reviewed (highest score first)")
ax.set_ylabel("Share of flagged claims found")
ax.set_title("Gains chart: review-queue effectiveness")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "gains_chart.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 6. Calibration by decile
dec = rank.assign(decile=pd.qcut(rank["score"].rank(method="first"), 10, labels=range(1, 11)))
cal = dec.groupby("decile", observed=True).agg(
    claims=(target, "size"), mean_predicted=("score", "mean"), observed_flag_rate=(target, "mean")).reset_index()
cal.to_csv(OUT / "decile_calibration.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
w = 0.38
ax.bar(cal["decile"].astype(int) - w / 2, cal["mean_predicted"] * 100, w, label="Mean predicted")
ax.bar(cal["decile"].astype(int) + w / 2, cal["observed_flag_rate"] * 100, w, label="Observed flag rate")
ax.set_xlabel("Score decile (1 = lowest risk, 10 = highest)")
ax.set_ylabel("%")
ax.set_title("Calibration by decile (out-of-fold)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "decile_calibration.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ Summary
b = cv_df.iloc[0]
summary = {
    "dataset": "Synthetic insurance claims dataset",
    "records": int(len(df)),
    "flagged_claims": int(total_flags),
    "synthetic_review_flag_rate": round(float(y.mean()), 4),
    "best_model_by_cv_roc_auc": best,
    "holdout_roc_auc": round(float(roc_auc_score(y_te, test_score)), 4),
    "holdout_average_precision": round(float(average_precision_score(y_te, test_score)), 4),
    "cv_roc_auc_mean_std": [round(float(b.roc_auc_mean), 4), round(float(b.roc_auc_std), 4)],
    "cv_average_precision_vs_baseline": [round(float(b.average_precision_mean), 4), round(float(y.mean()), 4)],
    "flags_found_in_top_20pct_of_queue": round(float(b[f"flags_found_in_top_{int(QUEUE_SHARE*100)}pct"]), 4),
    "permutation_test_p_value": round(float(p_value), 4),
    "purpose": "Synthetic review-flag prioritisation only; not a fraud-detection or production claim.",
}
(OUT / "project_summary.json").write_text(json.dumps(summary, indent=2))

pd.set_option("display.width", 200)
print("Hold-out\n", holdout.round(3).to_string(index=False))
print("\nCV\n", cv_df.round(3).to_string(index=False))
print("\nPermutation test\n", pd.read_csv(OUT / "permutation_test.csv").round(4).to_string(index=False))
print("\nTop drivers\n", odds.head(8).round(3).to_string(index=False))
print("\nLift\n", lift.round(3).to_string(index=False))
print("\nDecile calibration\n", cal.round(3).to_string(index=False))
print(json.dumps(summary, indent=2))
