import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model,
    TextField,
    FloatField,
    IntegrityError,
)
from playhouse.db_url import connect


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts"

PIPELINE_PATH = MODEL_DIR / "deploy_xgb_pipeline.pkl"
COLUMNS_PATH = MODEL_DIR / "deploy_raw_columns.json"
DTYPES_PATH = MODEL_DIR / "deploy_dtypes.json"


# -----------------------------
# Load model artifacts
# -----------------------------

pipeline = joblib.load(PIPELINE_PATH)

with open(COLUMNS_PATH) as f:
    raw_columns = json.load(f)

with open(DTYPES_PATH) as f:
    raw_dtypes = json.load(f)


# -----------------------------
# Database
# -----------------------------

DB = connect(os.environ.get("DATABASE_URL") or "sqlite:///predictions.db")


class ResponsePrediction(Model):
    unit_id = TextField()
    received_dttm = TextField()
    observation = TextField()
    predicted_response_time_seconds = FloatField()
    on_scene_dttm = TextField(null=True)
    actual_response_time_seconds = FloatField(null=True)

    class Meta:
        database = DB
        indexes = (
            (("unit_id", "received_dttm"), True),
        )


DB.connect(reuse_if_open=True)
DB.create_tables([ResponsePrediction], safe=True)


# -----------------------------
# Flask app
# -----------------------------

app = Flask(__name__)


def make_error(message, status_code=422):
    return jsonify({"error": message}), status_code


def validate_required_fields(payload, required_fields):
    if not isinstance(payload, dict):
        return False

    for field in required_fields:
        if field not in payload:
            return False

    return True


def build_observation(payload):
    obs = pd.DataFrame([payload], columns=raw_columns)

    for col, dtype in raw_dtypes.items():
        if col in obs.columns:
            try:
                obs[col] = obs[col].astype(dtype)
            except Exception:
                pass

    return obs


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict_response/", methods=["POST"])
def predict_response():
    payload = request.get_json(silent=True)

    if not validate_required_fields(payload, raw_columns):
        return make_error("Incorrectly formatted input data.")

    try:
        received_dt = pd.to_datetime(payload["received_dttm"], errors="raise")
        received_dttm = received_dt.isoformat()
    except Exception:
        return make_error("received_dttm must be a valid ISO 8601 datetime.")

    try:
        obs = build_observation(payload)
        prediction = float(pipeline.predict(obs)[0])
    except Exception as e:
        return make_error(f"Prediction failed: {str(e)}")

    record = ResponsePrediction(
        unit_id=str(payload["unit_id"]),
        received_dttm=received_dttm,
        observation=json.dumps(payload),
        predicted_response_time_seconds=prediction,
    )

    try:
        record.save()
    except IntegrityError:
        DB.rollback()
        return make_error("This unit_id and received_dttm pair already exists.")

    return jsonify({
        "unit_id": str(payload["unit_id"]),
        "received_dttm": received_dttm,
        "predicted_response_time_seconds": prediction,
    })


@app.route("/actual_response/", methods=["POST"])
def actual_response():
    payload = request.get_json(silent=True)

    required_fields = ["unit_id", "received_dttm", "on_scene_dttm"]

    if not validate_required_fields(payload, required_fields):
        return make_error("Incorrectly formatted input data.")

    try:
        received_dt = pd.to_datetime(payload["received_dttm"], errors="raise")
        on_scene_dt = pd.to_datetime(payload["on_scene_dttm"], errors="raise")
    except Exception:
        return make_error("received_dttm and on_scene_dttm must be valid ISO 8601 datetimes.")

    actual_seconds = (on_scene_dt - received_dt).total_seconds()

    if actual_seconds < 0:
        return make_error("Actual response time cannot be negative.")

    received_dttm = received_dt.isoformat()
    on_scene_dttm = on_scene_dt.isoformat()

    try:
        record = ResponsePrediction.get(
            (ResponsePrediction.unit_id == str(payload["unit_id"])) &
            (ResponsePrediction.received_dttm == received_dttm)
        )
    except ResponsePrediction.DoesNotExist:
        return make_error("This unit_id and received_dttm pair was not found.")

    record.on_scene_dttm = on_scene_dttm
    record.actual_response_time_seconds = float(actual_seconds)
    record.save()

    return jsonify({
        "unit_id": str(payload["unit_id"]),
        "received_dttm": received_dttm,
        "on_scene_dttm": on_scene_dttm,
        "actual_response_time_seconds": float(actual_seconds),
        "predicted_response_time_seconds": float(record.predicted_response_time_seconds),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

