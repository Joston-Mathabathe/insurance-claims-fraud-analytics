# Synthetic dataset provenance and reproducibility

`data/raw/insurance_claims.csv` contains 6,000 synthetic (simulated) insurance-claim records, of which 287 carry a synthetic `fraud_review_flag`. It is not real customer or insurer data.

**The generation method, code and random seed are not documented.** The raw CSV therefore cannot be re-created from source code, and this repository does not include or claim a data generator. The file is treated as a fixed input. Relationships in it may reflect how it was simulated rather than real insurance behaviour. Do not treat the synthetic review flags as verified fraud, verified investigation outcomes or observed recoveries.

The analysis **is** reproducible from the versioned raw CSV: install `requirements.txt`, run `python src/run_all.py`, then `python -m unittest discover -s tests -v`.

Future work could replace this file with an independently documented synthetic generator and compare its distributions, missingness and label construction.
