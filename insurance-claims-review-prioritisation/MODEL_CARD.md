# Model card — Synthetic insurance review-flag prioritisation

## Scope
Educational demonstration of claims **review prioritisation**. `fraud_review_flag` is a synthetic referral flag, not a fraud finding or a verified recovery outcome. No real claim should be declined or investigated automatically based on this model.

## Data and intended decision
6,000 simulated claims, 287 flagged. The business decision is how to allocate a constrained **human** investigation queue, not whether a customer committed fraud.

## Method
Logistic Regression, Decision Tree and Random Forest. Numeric missing values are imputed and standardised **inside training folds**; policy type is imputed and one-hot encoded inside the pipeline. Fixed-seed stratified 80/20 train/test split. Selection by five-fold CV on the **80% development set**; final test evaluation only for the selected model.

## Reproduced results
- **Independent hold-out (1,200 claims; 57 flagged): ROC-AUC 0.585; average precision 0.063; capture at top 20% 29.8%.** Small flagged sample means substantial uncertainty.
- Development CV ROC-AUC: 0.617 ± 0.038 (Logistic Regression selected); development OOF capture at top 20% 30.4%.
- Baseline review-flag prevalence: 0.0478.
- Descriptive full-portfolio cross-fitted ranking: 31.7% of review flags in top 20% of claims.
- Permutation-test result depends on configured number of permutations; see `reports/permutation_test.csv`.

## Known limitations
Weak predictive separation, synthetic labels, potential measurement bias, no fairness evaluation, no temporal validation, no production calibration, no real financial recovery observations. Cross-fitted portfolio queue metrics and scenario-optimised queue size are **not** independent policy evaluation.

## Governance requirements before real use
Representative claims, outcome definitions based on verified investigations, documented feature availability, independent temporal validation, fairness and privacy review, human decision authority, monitoring and review rights.
