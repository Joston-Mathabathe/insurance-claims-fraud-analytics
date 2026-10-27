# Power BI Dashboard Specification

**Status:** This is a dashboard design specification with example DAX; **no `.pbix` file is included**. See README **Recommended next steps**. All records and review flags are synthetic.

## Page 1: Claims Portfolio Overview
- Total claims, claim value, average claim amount, simulated review-flag rate
- Claim count and amount by policy type; review-flag rate by reporting delay
- Slicers: policy type, claim amount band (fixed thresholds: R10,000, R30,000 and R75,000)

## Page 2: Investigation Prioritisation
- Claim amount, claim-to-premium ratio, previous claims, reporting delay, documentation count
- Review capacity and flag concentration by queue share; no automatic fraud accusations

## Page 3: Model Performance
- **Independent hold-out**: ROC-AUC, average precision, capture rate at top 20% and review-flag prevalence baseline
- Development-only CV results separately labelled; lift and gains chart identified as descriptive cross-fitted portfolio results
- Avoid headline accuracy on the heavily imbalanced synthetic flag target
- Caveat: only 57 flagged claims in the hold-out; estimates are uncertain

## Page 4: Review Queue and Hypothetical Economics
- Use `reports/review_tiers.csv`, `reports/review_economics.csv`, `reports/review_economics_sensitivity.csv` and `reports/lift_table.csv`
- Review queue share, estimated workload, descriptive flag capture, hypothetical benefit by strategy
- Cost-per-review and assumed recovery-share scenario slicers; clearly label financial results **hypothetical**
- Optimised queue size is selected and evaluated on the same synthetic portfolio, **not independent policy validation**

## Business questions
- How much investigator capacity is available, and what trade-offs does prioritisation create?
- Does the model improve ranking over random review or simple rules?
- How sensitive is the scenario to review costs and assumed recoveries?
- How are customer fairness, appeal rights and human oversight protected?

## Example DAX measures
```DAX
Total Claims = COUNTROWS(insurance_claims_clean)
Total Claim Value = SUM(insurance_claims_clean[claim_amount])
Average Claim Amount = AVERAGE(insurance_claims_clean[claim_amount])
Flagged Claims = CALCULATE(COUNTROWS(insurance_claims_clean), insurance_claims_clean[fraud_review_flag] = 1)
Review Flag Rate = DIVIDE([Flagged Claims], [Total Claims])
Flagged Claim Value = CALCULATE(SUM(insurance_claims_clean[claim_amount]), insurance_claims_clean[fraud_review_flag] = 1)
```

Import model metrics from `reports/model_comparison.csv` and `reports/cross_validation.csv` instead of attempting to derive ROC-AUC or average precision from the claims table. Review-cost and recovery assumptions are hypothetical; do not label simulated benefits as realised savings.

*Footnote for every page: Synthetic data; review flags are not confirmed fraud; not for real claim decisions.*
