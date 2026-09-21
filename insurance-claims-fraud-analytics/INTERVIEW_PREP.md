# Interview Preparation

## 60-Second Explanation
I built a synthetic insurance claims and fraud analytics project to explore how data can support claims investigation. I first cleaned the claims data and created business-oriented indicators such as claim-to-premium ratio and reporting-delay bands. I used SQL and exploratory analysis to identify patterns, then compared Logistic Regression, Decision Tree and Random Forest models. I evaluated them using precision, recall, F1-score and ROC-AUC rather than relying only on accuracy. From a Business Management perspective, I also considered investigation capacity, customer impact, financial exposure and responsible use. The model is intended to prioritise claims for human review, not automatically declare fraud.

## Questions

### Why did you choose this problem?
Insurance combines risk management, customer service, financial exposure and data-driven decision-making.

### Why is this a Business Management project as well as a Data Science project?
The technical model is only part of the solution. I also identified stakeholders, KPIs, operational constraints and the business consequences of false positives and false negatives.

### Why not automatically reject a flagged claim?
A model prediction is not proof of fraud. Claims should be investigated using appropriate evidence and governed processes.

### Why use precision and recall?
They show different types of model error. Recall measures how many target cases are identified, while precision measures how accurate the flagged group is.

### What would you improve with real data?
I would use validated investigation outcomes, test for data leakage, assess fairness, validate the model on later data, monitor performance over time and involve claims, fraud, compliance and risk stakeholders.

### What would management want to know?
Management would want to know the financial exposure, review workload, investigation outcomes, model performance and customer impact—not just the model's technical score.
