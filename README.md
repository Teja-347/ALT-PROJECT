# Personalized Learning Path Prediction

Applied Machine Learning for Text Analytics (ALT) BTech mini-project.

## Overview

This project predicts the **next EdNet KT2 interaction type** (`enter`, `respond`, or
`submit`) from a student's interaction history and the current item's available
metadata. It demonstrates a leakage-aware learning workflow: features for an event
are calculated from interactions that occurred earlier for the same student, never
from later events or the event's target label.

The project supports two modes:

* **Synthetic demonstration mode**: small, deterministic data for tests and a
  laptop-friendly dashboard demonstration. It is always labelled synthetic.
* **EdNet mode**: reads locally downloaded EdNet KT2 per-student CSV files. The
  full dataset is not included because the repository is several gigabytes when
  uncompressed.

This is a predictive project, not a causal study. Feature importance indicates
model association and must not be interpreted as proof that a behavior causes an
outcome.

## Objectives and features

* Load and validate EdNet KT2-compatible records.
* Handle missing values, duplicates, and common type inconsistencies.
* Build chronological, student-history features without future leakage.
* Compare Logistic Regression, Random Forest, and optional XGBoost.
* Report accuracy, weighted precision/recall, macro-F1, and confusion matrices.
* Persist the preprocessing pipeline and trained models.
* Keep synthetic and EdNet models/reports in separate artifact directories.
* Record bounded-load data-quality and held-out evaluation metadata.
* Provide SHAP explanations where compatible, with a model-importance fallback.
* Explore data and predictions through a Streamlit dashboard.

## Dataset and preparation

EdNet is published by Riiid at <https://github.com/riiid/ednet>. This implementation
uses **KT2**, whose documented fields are `timestamp`, `action_type`, `item_id`,
`source`, `user_answer`, and `platform`; the student ID is represented by the
per-user filename in the download. KT2 contains question-solving actions:
`enter`, `respond`, and `submit`.

Download EdNet KT2 from the source project and extract the per-user CSV files under
`data/raw/ednet-kt2/`. Do not commit the archive or extracted data. The loader
accepts a directory, a single CSV, or a ZIP archive and caps the number of student
CSV files at 50 by default for laptop-friendly experiments, with `--max-users` or
`max_users` available to adjust the limit. It discovers CSV files rather than
requiring specific filenames, skips malformed files individually, and reports files
processed, rows returned, duplicates removed, and data-quality errors.

The EdNet dataset is distributed under the Creative Commons Attribution-
NonCommercial 4.0 International terms as stated by the source project. Those terms
apply to the dataset, not to this original project code. Confirm the current
dataset terms and attribution requirements in the official repository before
redistribution or commercial use.

## Installation on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installing XGBoost or SHAP is not possible on a particular machine, the project
continues with the required scikit-learn models and the documented fallback
explainability path.

## Training and evaluation

Synthetic smoke run:

```powershell
python -m src.train --synthetic --output-dir models
python -m src.evaluate --synthetic --model-dir models --output-dir reports
```

The dashboard uses `models/synthetic/` and `reports/synthetic/` for its synthetic
demonstration artifacts, so they cannot be confused with real-data artifacts.

EdNet sample run:

```powershell
python -m src.train --data-dir data/raw/ednet-kt2 --max-users 1000 --max-rows 100000 --output-dir models
python -m src.evaluate --data-dir data/raw/ednet-kt2 --model-dir models --output-dir reports
```

For dashboard use, select **Local EdNet KT2** and enter
`data/raw/ednet-kt2`. The dashboard uses separate `models/ednet/` and
`reports/ednet/` directories and never substitutes synthetic data for a missing
real dataset. A real-data score is only shown after real-data training completes.

## Streamlit

```powershell
python -m streamlit run app.py
```

Use the sidebar to select synthetic demonstration data or a local EdNet path.
Predictions and charts display the mode so synthetic results are not mistaken for
EdNet results.

## Tests

```powershell
python -m pytest -q
```

## Project structure

```text
app.py
requirements.txt
data/
  README.md
src/
  data_loader.py
  preprocessing.py
  feature_engineering.py
  target_creation.py
  train.py
  evaluate.py
  predict.py
  explainability.py
models/
  synthetic/
  ednet/
reports/
  synthetic/
  ednet/
tests/
```

## Results, limitations, and future scope

Evaluation reports are generated from the data actually supplied to the commands;
no scores are hard-coded. Results from synthetic data only validate the workflow and
are not evidence about real student behavior. EdNet timestamps are shifted by the
publisher, the target is an interaction event rather than a pedagogical diagnosis,
and a bounded sample may not represent the full population. Future work could add
KT3/KT4 content activities, skill-level targets using the contents files, temporal
validation across later sessions, calibration, fairness analysis, and deployment
monitoring.

## Team members

* G Teja Swaroop — 2420030347
* Sai Vivek — 2420030018
* G Prasanth — 2420090072
