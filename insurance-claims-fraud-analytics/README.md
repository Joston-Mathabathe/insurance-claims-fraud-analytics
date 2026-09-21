# Insurance Claims & Fraud Analytics

## Overview
A beginner-friendly portfolio project demonstrating how Python, pandas, NumPy, Matplotlib, scikit-learn, SQL and Power BI can support insurance claims investigation.

The project combines **Business Management + Data Science**.

## Business Problem
An insurer needs to identify claims that may warrant additional investigation while managing false positives, operational workload, customer impact and financial exposure.

## Important
The dataset and target are **synthetic**. The results do not represent a real insurer and must not be presented as real-world fraud rates or insurance findings.

## Tools
- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- SQL
- Power BI

## Project Structure
```text
insurance-claims-fraud-analytics/
├── data/
│   ├── raw/insurance_claims.csv
│   ├── processed/insurance_claims_clean.csv
│   └── data_dictionary.csv
├── notebooks/insurance_claims_fraud_analysis.ipynb
├── src/
│   ├── cleaning.py
│   ├── eda.py
│   └── model.py
├── sql/insurance_claims_analysis.sql
├── reports/
├── visualisations/
├── BUSINESS_ANALYSIS.md
├── BUSINESS_DECISION_TABLE.md
├── POWER_BI_DASHBOARD.md
├── MODEL_CARD.md
├── INTERVIEW_PREP.md
├── requirements.txt
└── .gitignore
```

## Business Management Layer
The project includes stakeholder analysis, business requirements, KPIs, decision-support logic, operational considerations and responsible-use guidance.

## Technical Workflow
Raw data -> Cleaning -> Feature engineering -> EDA -> SQL -> Machine learning -> Evaluation -> Business interpretation -> Power BI

## Responsible Analytics
A fraud flag should be treated as a reason for further investigation, not as proof of fraud. Real insurance use would require governance, validated outcomes, fairness assessment, regulatory compliance and human oversight.
