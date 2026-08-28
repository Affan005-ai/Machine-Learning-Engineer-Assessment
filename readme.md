# Spotter Freight Rate Assessment

This project predicts `posted_rate` for future freight loads using a reproducible pandas and scikit-learn pipeline.

## Structure

```text
data/                 Input CSV files
src/clean.py          Shared data cleaning
src/features.py       Shared feature engineering
src/model.py          Random Forest model and persistence
eda.ipynb             Exploratory analysis
train.py              Time-based validation and final training
predict.py            Validation and December predictions
score.py              Submission-format checker
REPORT.md             Method and results report
```

## Setup and run

From Command Prompt:

```cmd
cd /d D:\Affan\assessment
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python train.py
python predict.py
python score.py --predictions validation_predictions.csv --december-predictions data\december-chart-inputs.csv
```

The training script uses a chronological January-August versus September-October holdout, compares a log-target Ridge baseline with a tuned Random Forest, and then retrains the selected Random Forest on all labeled rows. The prediction script creates `validation_predictions.csv` and fills `predicted_rate` in the December input file.

Open `eda.ipynb` with the `.venv` interpreter to review the data-quality checks and exploratory plots. The generated model in `artifacts/` is ignored because it is too large for GitHub's standard file limit and can be recreated with `python train.py`.

## Scoring

Install the requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.
