"""Lightweight checks for data quality and reproducible outputs."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class PortfolioChecks(unittest.TestCase):
    def test_synthetic_dataset_and_target(self):
        df = pd.read_csv(ROOT / "data/raw/insurance_claims.csv")
        self.assertEqual(len(df), 6000)
        self.assertTrue(set(df["fraud_review_flag"].dropna().unique()).issubset({0, 1}))
        self.assertEqual(int(df["fraud_review_flag"].sum()), 287)

    def test_cleaned_features_and_missingness(self):
        df = pd.read_csv(ROOT / "data/processed/insurance_claims_clean.csv")
        self.assertEqual(len(df), 6000)
        self.assertIn("claim_to_premium_ratio", df)
        self.assertTrue(df["claim_amount"].isna().sum() > 0)  # train-only imputation

    def test_independent_holdout_report(self):
        summary = json.loads((ROOT / "reports/project_summary.json").read_text())
        self.assertEqual(summary["records"], 6000)
        self.assertGreater(summary["holdout_roc_auc"], 0.5)
        self.assertLess(summary["holdout_roc_auc"], 1.0)

    def test_scenario_outputs_are_finite(self):
        df = pd.read_csv(ROOT / "reports/review_economics.csv")
        self.assertTrue(df["hypothetical_net_benefit_R"].notna().all())
        self.assertEqual(len(df), 3)

if __name__ == "__main__":
    unittest.main()
