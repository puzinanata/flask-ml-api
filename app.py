import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from peewee import Model, IntegerField, CharField, FloatField, SqliteDatabase
from playhouse.db_url import connect


app = Flask(__name__)


# =========================================================
# DATABASE
# =========================================================

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    db = connect(DATABASE_URL)
else:
    db = SqliteDatabase("updates.db")


class BaseModel(Model):
    class Meta:
        database = db


class GroundTruthUpdate(BaseModel):
    date_str = CharField()
    port_code = IntegerField()
    traffic = CharField()
    true_value = FloatField()


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "model_package.pkl"
model_package = joblib.load(MODEL_PATH)

pipeline = model_package["pipeline"]
history_store = model_package["history_store"]
last_train_date = pd.Timestamp(model_package["last_train_date"])
traffic_values = model_package["traffic_values"]


# =========================================================
# HELPERS
# =========================================================

def make_feature_row(port_code, traffic, date_value, lag_1, lag_2, lag_3):
    month = date_value.month

    return pd.DataFrame([{
        "Code": int(port_code),
        "traffic": traffic,
        "year": date_value.year,
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3
    }])


def load_updates_from_db():
    global last_train_date

    query = GroundTruthUpdate.select().order_by(GroundTruthUpdate.id)

    for row in query:
        key = (row.port_code, row.traffic)

        if key not in history_store:
            history_store[key] = []

        history_store[key].append(row.true_value)

        if len(history_store[key]) > 3:
            history_store[key] = history_store[key][-3:]

        row_date = pd.to_datetime(row.date_str, format="%b %Y")

        if row_date > last_train_date:
            last_train_date = row_date


def recursive_forecast(port_code, traffic):
    key = (int(port_code), traffic)

    if key not in history_store:
        return {"error": "No history for this port_code and traffic"}, 422

    history = history_store[key].copy()

    if len(history) < 3:
        return {"error": f"Not enough history for port_code={port_code} and traffic='{traffic}'"}, 422

    future_dates = pd.date_range(
        start=last_train_date + pd.DateOffset(months=1),
        periods=6,
        freq="MS"
    )

    predictions = []

    for future_date in future_dates:
        lag_1 = history[-1]
        lag_2 = history[-2]
        lag_3 = history[-3]

        X = make_feature_row(
            port_code=port_code,
            traffic=traffic,
            date_value=future_date,
            lag_1=lag_1,
            lag_2=lag_2,
            lag_3=lag_3
        )

        pred = pipeline.predict(X)[0]
        pred = int(round(max(0, pred)))

        predictions.append(pred)
        history.append(pred)

    return {
        "port_code": int(port_code),
        "traffic": traffic,
        "prediction": " ".join(map(str, predictions))
    }, 200


# =========================================================
# INIT DB
# =========================================================

db.connect(reuse_if_open=True)
db.create_tables([GroundTruthUpdate], safe=True)
load_updates_from_db()


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return "API is running"


# ---------------------------
# PREDICT
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON provided"}), 422

    if "port_code" not in data or "traffic" not in data:
        return jsonify({"error": "port_code and traffic required"}), 422

    try:
        port_code = int(data["port_code"])
    except Exception:
        return jsonify({"error": "port_code must be int"}), 422

    traffic = data["traffic"]

    if not isinstance(traffic, str):
        return jsonify({"error": "traffic must be string"}), 422

    if traffic not in traffic_values:
        return jsonify({"error": f"traffic must be one of {traffic_values}"}), 422

    result, status_code = recursive_forecast(port_code, traffic)
    return jsonify(result), status_code


# ---------------------------
# UPDATE
# ---------------------------
@app.route("/update", methods=["POST"])
def update():
    global last_train_date

    data = request.get_json()

    required = ["date", "port_code", "traffic", "true_value"]

    if not data or not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 422

    try:
        date_parsed = pd.to_datetime(data["date"], format="%b %Y")
    except Exception:
        return jsonify({"error": "date must be like 'Sep 2025'"}), 422

    try:
        port_code = int(data["port_code"])
    except Exception:
        return jsonify({"error": "port_code must be int"}), 422

    traffic = data["traffic"]

    if not isinstance(traffic, str):
        return jsonify({"error": "traffic must be string"}), 422

    if traffic not in traffic_values:
        return jsonify({"error": f"traffic must be one of {traffic_values}"}), 422

    try:
        value = float(data["true_value"])
    except Exception:
        return jsonify({"error": "true_value must be numeric"}), 422

    key = (port_code, traffic)

    if key not in history_store:
        history_store[key] = []

    history_store[key].append(value)

    if len(history_store[key]) > 3:
        history_store[key] = history_store[key][-3:]

    if date_parsed > last_train_date:
        last_train_date = date_parsed

    existing = GroundTruthUpdate.get_or_none(
        (GroundTruthUpdate.date_str == data["date"]) &
        (GroundTruthUpdate.port_code == port_code) &
        (GroundTruthUpdate.traffic == traffic)
    )

    if existing:
        existing.true_value = value
        existing.save()
    else:
        GroundTruthUpdate.create(
            date_str=data["date"],
            port_code=port_code,
            traffic=traffic,
            true_value=value
        )

    return jsonify({
        "date": data["date"],
        "port_code": port_code,
        "traffic": traffic,
        "true_value": value
    }), 200


# ---------------------------
# LIST DB CONTENTS
# ---------------------------
@app.route("/list-db-contents", methods=["GET"])
def list_db_contents():
    rows = []
    query = GroundTruthUpdate.select().order_by(GroundTruthUpdate.id)

    for row in query:
        rows.append({
            "id": row.id,
            "date": row.date_str,
            "port_code": row.port_code,
            "traffic": row.traffic,
            "true_value": row.true_value
        })

    return jsonify(rows), 200


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)