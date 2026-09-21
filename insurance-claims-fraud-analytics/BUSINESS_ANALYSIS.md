# Business Analysis & Decision Framework

## Business Problem
Insurance claims teams need to prioritise claims that may require additional investigation. The objective is not to label customers as fraudulent automatically, but to use analytics to identify unusual patterns and support human review.

## Business Objectives
- Understand claim patterns associated with the synthetic review flag.
- Identify useful risk indicators.
- Build a classification model for investigation prioritisation.
- Measure false positives and false negatives.
- Translate model results into operational and management considerations.

## Stakeholders
| Stakeholder | Need |
|---|---|
| Claims Team | Prioritise claims for investigation |
| Fraud/Risk Team | Identify unusual patterns and monitor risk |
| Finance | Understand potential financial exposure |
| Customer Service | Manage customer impact of additional reviews |
| Management | Monitor KPIs and operational capacity |
| Data/Analytics | Build and monitor analytical solutions |

## KPIs
- Total claims
- Average claim amount
- Review-flag rate
- Claims by policy type
- Review rate by policy type
- Average claim-to-premium ratio
- Average reporting delay
- Precision
- Recall
- F1-score
- ROC-AUC

## Business Interpretation
A higher recall can help identify more claims that belong to the flagged class, but it can also increase false positives and investigation workload. Precision indicates how many flagged claims are actually in the target class. A real insurer would choose operating thresholds based on financial costs, investigation capacity, customer impact, policy and governance requirements.

## Decision Workflow
Claims data -> Data quality checks -> Risk indicators -> Predictive model -> Prioritised review queue -> Human investigation -> Outcome -> Monitoring

## Responsible Use
The synthetic target is a portfolio modelling device, not evidence of real fraud. A real implementation would require validated investigation outcomes, legal and regulatory controls, fairness assessment, model governance, secure data handling and human oversight.
