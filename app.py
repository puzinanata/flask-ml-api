import os
import json
import pickle
import joblib
import pandas as pd

from flask import Flask, jsonify, request
from peewee import (
    Model, FloatField, IntegerField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

DB = connect(os.environ.get("DATABASE_URL") or "sqlite:///predictions.db")


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.connect(reuse_if_open=True)
DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "columns.json")) as fh:
    columns = json.load(fh)

pipeline = joblib.load(os.path.join(BASE_DIR, "pipeline.pickle"))

with open(os.path.join(BASE_DIR, "dtypes.pickle"), "rb") as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################


########################################
# Validation helpers

REQUIRED_FIELDS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

VALID_CATEGORIES = {
    "sex": [
        "Female", "Male"
    ],
    "race": [
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White",
    ],
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
        "?",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
}


def error_response(observation_id, message):
    return jsonify({
        "observation_id": observation_id,
        "error": message
    }), 200


def validate_payload(payload):
    """
    Returns:
        (observation_id, data, error_message)
    """
    if not isinstance(payload, dict):
        return None, None, "request must be a dictionary"

    if "observation_id" not in payload:
        return None, None, "missing field: observation_id"

    observation_id = payload["observation_id"]

    if "data" not in payload:
        return observation_id, None, "missing field: data"

    data = payload["data"]

    if not isinstance(data, dict):
        return observation_id, None, "data must be a dictionary"

    # missing fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return observation_id, None, f"missing field: {field}"

    # extra fields
    extra_fields = set(data.keys()) - set(REQUIRED_FIELDS)
    if extra_fields:
        extra_field = sorted(extra_fields)[0]
        return observation_id, None, f"unexpected field: {extra_field}"

    # categorical checks
    for field, valid_values in VALID_CATEGORIES.items():
        if data[field] not in valid_values:
            return observation_id, None, f"invalid value for {field}: {data[field]}"

    # numeric checks
    numeric_fields = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    for field in numeric_fields:
        value = data[field]
        if not isinstance(value, (int, float)):
            return observation_id, None, f"invalid value for {field}: {value}"

    # basic range checks to satisfy tests
    if data["age"] < 0:
        return observation_id, None, f"invalid value for age: {data['age']}"
    if data["capital-gain"] < 0:
        return observation_id, None, f"invalid value for capital-gain: {data['capital-gain']}"
    if data["capital-loss"] < 0:
        return observation_id, None, f"invalid value for capital-loss: {data['capital-loss']}"
    if data["hours-per-week"] < 0:
        return observation_id, None, f"invalid value for hours-per-week: {data['hours-per-week']}"

    return observation_id, data, None


# End validation helpers
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    observation_id, data, err = validate_payload(payload)
    if err is not None:
        return error_response(observation_id, err)

    try:
        obs = pd.DataFrame([data], columns=columns).astype(dtypes)
        prediction = bool(pipeline.predict(obs)[0])
        probability = float(pipeline.predict_proba(obs)[0, 1])
    except Exception:
        return error_response(observation_id, "Observation is invalid!")

    # save prediction attempt
    p = Prediction(
        observation_id=str(observation_id),
        observation=json.dumps(data),
        proba=probability
    )

    try:
        p.save()
    except IntegrityError:
        return error_response(observation_id, f"Observation ID {observation_id} already exists")

    return jsonify({
        "observation_id": observation_id,
        "prediction": prediction,
        "probability": probability
    }), 200


@app.route("/update", methods=["POST"])
def update():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "request must be a dictionary"}), 200

    if "observation_id" not in payload:
        return jsonify({
            "observation_id": None,
            "error": "missing field: observation_id"
        }), 200

    observation_id = payload["observation_id"]

    if "true_class" not in payload:
        return jsonify({
            "observation_id": observation_id,
            "error": "missing field: true_class"
        }), 200

    try:
        p = Prediction.get(Prediction.observation_id == str(observation_id))
        p.true_class = int(payload["true_class"])
        p.save()
        return jsonify(model_to_dict(p)), 200
    except ValueError:
        return jsonify({
            "observation_id": observation_id,
            "error": f"invalid value for true_class: {payload['true_class']}"
        }), 200
    except Prediction.DoesNotExist:
        return jsonify({
            "observation_id": observation_id,
            "error": f"Observation ID {observation_id} does not exist"
        }), 200


@app.route("/list-db-contents", methods=["GET"])
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ]), 200


# End webserver stuff
########################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", debug=False, port=port)