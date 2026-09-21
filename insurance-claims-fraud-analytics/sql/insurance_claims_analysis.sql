-- Insurance Claims & Fraud Analytics
-- Dataset is synthetic. This SQL demonstrates business analysis.

-- 1. Overall claims and review-flag rate
SELECT
    COUNT(*) AS total_claims,
    SUM(fraud_review_flag) AS flagged_claims,
    AVG(fraud_review_flag) AS review_flag_rate
FROM insurance_claims_clean;

-- 2. Review rate by policy type
SELECT
    policy_type,
    COUNT(*) AS claims,
    SUM(fraud_review_flag) AS flagged_claims,
    AVG(fraud_review_flag) AS review_flag_rate
FROM insurance_claims_clean
GROUP BY policy_type
ORDER BY review_flag_rate DESC;

-- 3. Review rate by reporting delay
SELECT
    report_delay_band,
    COUNT(*) AS claims,
    AVG(fraud_review_flag) AS review_flag_rate
FROM insurance_claims_clean
GROUP BY report_delay_band
ORDER BY review_flag_rate DESC;

-- 4. Claims with multiple risk indicators
SELECT
    claim_id,
    policy_type,
    claim_amount,
    claim_to_premium_ratio,
    previous_claims,
    days_to_report,
    documents_submitted,
    repair_estimate_variance,
    fraud_review_flag
FROM insurance_claims_clean
WHERE claim_to_premium_ratio > 4
  AND previous_claims >= 3
  AND repair_estimate_variance > 0.35
ORDER BY claim_to_premium_ratio DESC;

-- 5. Average claim amount by policy type
SELECT
    policy_type,
    COUNT(*) AS claims,
    AVG(claim_amount) AS avg_claim_amount
FROM insurance_claims_clean
GROUP BY policy_type
ORDER BY avg_claim_amount DESC;
