# Business decision table

| Evidence | What it means | Management action |
|---|---|---|
| 287/6,000 synthetic claims flagged (4.78%) | Review resources are scarce relative to volume | Define a capacity budget and baseline queue |
| Development CV ROC-AUC 0.617 ± 0.038 | Some weak ranking signal | Keep the model experimental |
| Independent hold-out ROC-AUC 0.585 | Generalisation is limited | Do not use for automated adverse decisions |
| Hold-out average precision 0.063 vs 0.048 prevalence | Modest improvement over prevalence baseline | Compare against simple triage rules |
| 31.7% of flags in top 20% of cross-fitted queue | Descriptive prioritisation gain | Validate prospectively before operational use |
| Hypothetical value-ranked scenario ~R616,825 | A *sensitivity illustration*, not real recovered cash | Obtain actual investigator costs and verified outcomes |
| Human review and governance required | Customer harm and fairness risks remain | Define escalation, audit trail and appeal rights |

The scenario uses synthetic labels and an optimised queue share evaluated on the same dataset. It is **not** independent proof of business impact.
