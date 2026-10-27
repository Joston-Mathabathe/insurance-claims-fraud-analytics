# Business analysis: investigator capacity and claim-review governance

## Management question
How can a claims operations team prioritise **additional human review** without equating a synthetic flag with proven fraud?

## Objectives and KPIs
| Business objective | KPI | Guardrail |
|---|---|---|
| Allocate scarce investigator time | Claims reviewed, queue share, flags captured | Workload and service-level agreements |
| Improve prioritisation | Lift, average precision, ROC-AUC | Compare with random queue and existing rules |
| Manage financial exposure | Scenario recovery minus investigation costs | No claim of actual recoveries |
| Protect customer outcomes | Overrides, appeal rates, processing delays | No automatic fraud accusations or denials |

## Results and interpretation
The synthetic portfolio has 6,000 claims, of which 287 carry a **simulated review flag** (4.78%). Logistic Regression has development CV ROC-AUC 0.617 and independent hold-out ROC-AUC 0.585. This is **weak** separation: the model should not be positioned as a fraud detector.

Cross-fitted portfolio scores place about 31.7% of flags in the top 20% of the score-ranked queue. Under hypothetical assumptions of R1,000 per investigation and a 25% recovery share of flagged claim value, ranking by score × claim amount produces about **R616,825** scenario net benefit at an optimised 20% review share. The optimisation and evaluation occur on the same synthetic portfolio: this is **not an independently validated benefit**.

## Stakeholders
- **Claims operations manager:** staffing, review capacity, turnaround time and queue policy
- **Investigators:** evidence collection, case review and human judgment
- **Finance:** validated review costs, recovery amounts and sensitivity analysis
- **Data science / model risk:** independent validation, drift, documentation and performance
- **Compliance / customer protection:** fair treatment, privacy, explainability and appeals

## Decision recommendation
Treat the model as a **research prototype**. Compare with simple rules, pilot a *human-reviewed* prioritisation workflow only after independent validation, and collect actual investigation outcomes and costs. Do not automate adverse claim decisions or promise savings from synthetic labels.

## Risks and controls
1. Synthetic flag is not fraud confirmation; document label provenance.
2. Weak discrimination can create unnecessary investigations; monitor false referrals.
3. Financial assumptions are hypothetical; test cost and recovery scenarios.
4. Queue-share optimisation on the same portfolio is optimistic; validate on later data.
5. Check subgroup fairness, operational burden, privacy and customer recourse.
