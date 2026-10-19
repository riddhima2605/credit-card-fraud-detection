# Credit Card Fraud Detector

A Flask web app that scores credit card transactions for fraud using a tuned XGBoost model
trained on the public credit card fraud dataset (284,807 transactions, 492 frauds).

## Features
- Upload a CSV of transactions and get a fraud probability + prediction for each row
- "Try demo data" button using real test-set transactions
- Top contributing features per transaction (XGBoost native contributions)
- Input validation: empty files, wrong file type, missing columns, non-numeric or missing values, oversized files
- JSON endpoint: `POST /api/predict` (form field `file`)

## Why a CSV upload and not a form?
`V1`-`V28` are anonymised PCA features, so a normal user cannot type them in from a real card
transaction. The app therefore accepts CSV records with columns `Time, V1..V28, Amount`.
An optional `Class` column is ignored for prediction (used only to show actual labels in the demo).

## Model
Dataset: Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
| Item | Value |
|---|---|
| Model | XGBoost (200 trees, depth 5, learning rate 0.05) |
| Preprocessing | StandardScaler fitted on the training set |
| Decision threshold | 0.45, chosen on a validation set (not the test set) |
| Library versions | xgboost 3.4.1, scikit-learn 1.6.1 |

## Setup (Windows)
Put these files in one folder, together with the saved files from the notebook:
`fraud_model.json`, `scaler_fresh.joblib`, `config.json`, `sample_transactions.csv`.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Run the tests
```
python test_app.py
```

## Project files
- `app.py` - Flask application
- `templates/index.html` - web page
- `test_app.py` - automated tests (valid input, invalid input, column order)
- `requirements.txt` - pinned dependencies

## Screenshots
Add screenshots here: home page, demo results, an error message for an invalid file.
