import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model package
MODEL_PATH = "model_package.pkl"
model_package = joblib.load(MODEL_PATH)

pipeline = model_package["pipeline"]
history_store = model_package["history_store"]
last_train_date = pd.Timestamp(model_package["last_train_date"])
traffic_values = model_package["traffic_values"]


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


def recursive_forecast(port_code, traffic):
    key = (int(port_code), traffic)

    if key not in history_store:
        return {"error": "No history for this port_code and traffic"}

    history = history_store[key].copy()

    if len(history) < 3:
        return {"error": "Not enough history"}

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
            port_code,
            traffic,
            future_date,
            lag_1,
            lag_2,
            lag_3
        )

        pred = pipeline.predict(X)[0]
        pred = int(round(max(0, pred)))

        predictions.append(pred)
        history.append(pred)

    return {
        "port_code": int(port_code),
        "traffic": traffic,
        "prediction": " ".join(map(str, predictions))
    }


@app.route("/")
def home():
    return "API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    if "port_code" not in data or "traffic" not in data:
        return jsonify({"error": "port_code and traffic required"}), 400

    try:
        port_code = int(data["port_code"])
    except:
        return jsonify({"error": "port_code must be int"}), 400

    traffic = data["traffic"]

    if traffic not in traffic_values:
        return jsonify({"error": f"traffic must be one of {traffic_values}"}), 400

    result = recursive_forecast(port_code, traffic)

    return jsonify(result)


@app.route("/update", methods=["POST"])
def update():
    global last_train_date

    data = request.get_json()

    required = ["date", "port_code", "traffic", "true_value"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    date = data["date"]
    date_parsed = pd.to_datetime(date, format="%b %Y")

    port_code = int(data["port_code"])
    traffic = data["traffic"]
    value = float(data["true_value"])

    key = (port_code, traffic)

    if key not in history_store:
        history_store[key] = []

    history_store[key].append(value)

    if date_parsed > last_train_date:
        last_train_date = date_parsed

    return jsonify({
        "date": date,
        "port_code": port_code,
        "traffic": traffic,
        "true_value": value
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)