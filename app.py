"""Credit Card Fraud Detector - Flask app.

Loads the trained XGBoost model + fitted scaler (saved from the notebook),
accepts a CSV of transactions (columns Time, V1..V28, Amount), and returns a
fraud probability and prediction for every row.
"""
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb_lib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model files can sit next to app.py, or inside an "app_files" sub-folder
ART_DIR = BASE_DIR

MAX_ROWS = 5000
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---- load saved artefacts once, at startup ----
model = xgb_lib.XGBClassifier()
model.load_model(os.path.join(ART_DIR, "fraud_model.json"))

scaler = joblib.load(os.path.join(ART_DIR, "scaler_fresh.joblib"))

with open(os.path.join(ART_DIR, "config.json")) as f:
    config = json.load(f)

FEATURES = config["feature_names"]      # fixed feature order used in training
THRESHOLD = float(config["threshold"])  # chosen on the validation set


class InputError(Exception):
    """Raised for problems with the user's input (shown as a friendly message)."""


def read_and_validate(raw_bytes):
    """Turn uploaded bytes into a clean DataFrame with exactly FEATURES columns."""
    if not raw_bytes or not raw_bytes.strip():
        raise InputError("The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        raise InputError("Could not read the file as a CSV. Please upload a valid .csv file.")

    if df.empty:
        raise InputError("The CSV has a header but no transaction rows.")
    if len(df) > MAX_ROWS:
        raise InputError(f"Too many rows ({len(df)}). The limit is {MAX_ROWS} rows per upload.")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise InputError(f"Missing {len(missing)} required column(s): {shown}")

    actual = df["Class"] if "Class" in df.columns else None
    features = df[FEATURES].apply(pd.to_numeric, errors="coerce")  # bad text -> NaN
    if features.isna().any().any():
        bad_rows = features.index[features.isna().any(axis=1)][:5] + 2  # +2: header + 1-based
        raise InputError(
            "Some values are missing or not numeric "
            f"(first problem rows in the file: {', '.join(map(str, bad_rows))})."
        )
    if not np.isfinite(features.to_numpy()).all():
        raise InputError("Some values are infinite. Please check the file.")
    return features, actual


def predict(features):
    """Apply the SAME scaler as training, then predict. Returns probs + top features."""
    scaled = scaler.transform(features)  # DataFrame in training column order
    probs = model.predict_proba(scaled)[:, 1]

    # per-row feature contributions straight from XGBoost (no extra package needed)
    contribs = model.get_booster().predict(xgb_lib.DMatrix(scaled), pred_contribs=True)[:, :-1]
    top = []
    for row in contribs:
        idx = np.argsort(np.abs(row))[::-1][:3]
        top.append(", ".join(f"{FEATURES[i]} ({'+' if row[i] > 0 else '-'})" for i in idx))
    return probs, top


def build_results(features, actual):
    probs, top = predict(features)
    rows = []
    for i, p in enumerate(probs):
        is_fraud = bool(p >= THRESHOLD)
        row = {
            "row": i + 1,
            "amount": round(float(features["Amount"].iloc[i]), 2),
            "probability": round(float(p), 4),
            "prediction": "FRAUD" if is_fraud else "Genuine",
            "is_fraud": is_fraud,
            "top_features": top[i],
            "actual": None,
        }
        if actual is not None:
            try:
                row["actual"] = "FRAUD" if int(actual.iloc[i]) == 1 else "Genuine"
            except (ValueError, TypeError):
                row["actual"] = None
        rows.append(row)

    summary = {
        "total": len(rows),
        "flagged": sum(r["is_fraud"] for r in rows),
        "threshold": THRESHOLD,
        "has_actual": actual is not None and all(r["actual"] for r in rows),
    }
    if summary["has_actual"]:
        summary["correct"] = sum((r["prediction"] == r["actual"]) for r in rows)
    return rows, summary


def render_results(raw_bytes, source):
    try:
        features, actual = read_and_validate(raw_bytes)
        rows, summary = build_results(features, actual)
    except InputError as e:
        return render_template("index.html", error=str(e), threshold=THRESHOLD), 400
    except Exception:
        app.logger.exception("Unexpected error while predicting")
        return render_template(
            "index.html",
            error="Something went wrong while processing the file. Please check it and try again.",
            threshold=THRESHOLD,
        ), 500
    return render_template("index.html", rows=rows, summary=summary, source=source, threshold=THRESHOLD)


@app.route("/")
def home():
    return render_template("index.html", threshold=THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict_upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose a CSV file first.", threshold=THRESHOLD), 400
    if not file.filename.lower().endswith(".csv"):
        return render_template("index.html", error="Only .csv files are accepted.", threshold=THRESHOLD), 400
    return render_results(file.read(), source=file.filename)


@app.route("/demo")
def demo():
    with open(os.path.join(ART_DIR, "sample_transactions.csv"), "rb") as f:
        return render_results(f.read(), source="demo data (sample_transactions.csv)")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON version: POST a CSV as form field 'file'."""
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="No file uploaded (form field name must be 'file')."), 400
    try:
        features, actual = read_and_validate(file.read())
        rows, summary = build_results(features, actual)
    except InputError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("Unexpected error in /api/predict")
        return jsonify(error="Internal error while predicting."), 500
    return jsonify(summary=summary, predictions=rows)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error=f"File too large (limit {MAX_UPLOAD_MB} MB).", threshold=THRESHOLD), 413


if __name__ == "__main__":
    app.run(debug=False, port=5000)
