# PaySim Fraud Detection: Rule-Based Baselines vs Machine Learning

Reproducible fraud-detection case study comparing high-signal business rules against lightweight supervised machine learning baselines on the PaySim synthetic financial dataset.

## Why This Repo Exists

This project is a fallback portfolio piece for Data Scientist / ML Engineer applications.

The repo is intentionally scoped to show four things clearly:

- business understanding of a fraud-detection problem
- reproducible code-first ML workflow
- honest evaluation on an imbalanced classification task
- ability to turn notebook exploration into reusable project code

## Problem Statement

The PaySim dataset simulates mobile-money transactions and includes a small minority of fraudulent events. The challenge is not simply to build a classifier, but to compare:

- simple, explainable domain rules
- lightweight supervised models
- the trade-off between precision, recall, and operational usefulness

## Dataset

- Source: [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Expected local path: `data/raw/PS_20174392719_1491204439457_log.csv`
- License and usage terms for the dataset follow Kaggle / dataset-owner rules

This repository does not commit the full dataset. You download it locally, place it under `data/raw/`, and run the pipeline yourself.

## Project Approach

The repo follows a hybrid strategy:

1. Start with rule discovery from transaction behavior.
2. Turn the strongest fraud heuristics into a formal rule baseline.
3. Build reusable feature engineering on top of those domain signals.
4. Compare the rule baseline against simple supervised models.

Current supervised baselines:

- Logistic Regression with class weighting
- Random Forest with class weighting

## Rule Baseline Insight

The original exploratory analysis in the legacy notebook found a strong fraud pattern:

- `TRANSFER` where `amount == oldbalanceOrg`
- `CASH_OUT` where `amount == oldbalanceOrg`

In the original notebook-based analysis, those simple rules captured roughly `97.6%` of known fraud cases in the PaySim data while remaining fully interpretable.

This refactored repo turns that insight into tested code under `src/rules/baseline_rules.py` so it can be compared against machine-learning baselines instead of living only inside a notebook.

## What The Repo Contains

```text
.
|-- README.md
|-- requirements.txt
|-- pytest.ini
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_rule_baseline.ipynb
|   |-- 03_model_experiments.ipynb
|   `-- legacy/
|-- reports/
|   |-- figures/
|   `-- metrics/
|-- src/
|   |-- config.py
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- rules/
|   `-- utils/
`-- tests/
```

## Core Workflow

1. `src/data/load_data.py` loads and validates the PaySim schema.
2. `src/data/split.py` creates deterministic train / validation / test splits.
3. `src/rules/baseline_rules.py` computes the interpretable fraud heuristic.
4. `src/features/build_features.py` builds model-ready features.
5. `src/models/train.py` trains the ML baselines and exports comparable metrics.
6. `src/models/evaluate.py` reports fraud-appropriate metrics:
   `precision`, `recall`, `F1`, and `PR-AUC`

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the Kaggle dataset and place the CSV at:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Run the training pipeline:

```powershell
.\.venv\Scripts\python.exe -m src.models.train --input data/raw/PS_20174392719_1491204439457_log.csv
```

Optional quick local smoke run:

```powershell
.\.venv\Scripts\python.exe -m src.models.train --input data/raw/PS_20174392719_1491204439457_log.csv --sample-size 50000 --output-dir reports/metrics/smoke
```

## Outputs

After training, the repo writes:

- JSON metrics under `reports/metrics/`
- confusion-matrix figures under `reports/figures/`

The exported metrics compare:

- rule baseline
- logistic regression
- random forest

## Notebooks

The notebooks are supporting material, not the source of truth:

- `notebooks/01_eda.ipynb`: quick data overview
- `notebooks/02_rule_baseline.ipynb`: interpretable fraud rules
- `notebooks/03_model_experiments.ipynb`: code-first experiment walkthrough
- `notebooks/legacy/01_original_rule_discovery.ipynb`: original notebook retained for history

## Testing

The test suite focuses on engineering discipline rather than high coverage:

- schema validation
- deterministic splitting
- rule-baseline correctness
- feature construction
- metric computation
- training smoke test

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Limitations

- This is not a production fraud platform.
- The current repo keeps the model family intentionally small.
- Final benchmark values depend on the exact local dataset run.
- Rule baselines can be strong on this dataset but may not generalize to real-world systems without deeper validation.

## Why This Is Still Useful In A Portfolio

This project is not trying to win on model complexity. Its value is that it shows:

- you can derive useful signals from domain behavior
- you can formalize those signals into tested code
- you can compare interpretable baselines against ML baselines honestly
- you can structure a small ML project like software, not just like a notebook dump
