import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
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

with open("columns.json") as fh:
    columns = json.load(fh)

pipeline = joblib.load("pipeline.pickle")

with open("dtypes.pickle", "rb") as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    obs_dict = request.get_json(silent=True)

    if not isinstance(obs_dict, dict):
        return jsonify({
            "observation_id": None,
            "error": "Invalid JSON payload"
        }), 400

    if "observation_id" not in obs_dict:
        return jsonify({
            "observation_id": None,
            "error": "Missing field: observation_id"
        }), 400

    observation_id = obs_dict["observation_id"]

    if "data" not in obs_dict:
        return jsonify({
            "observation_id": observation_id,
            "error": "Missing field: data"
        }), 400

    observation = obs_dict["data"]

    if not isinstance(observation, dict):
        return jsonify({
            "observation_id": observation_id,
            "error": "Field 'data' must be a dictionary"
        }), 400

    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
        prediction = bool(pipeline.predict(obs)[0])
        probability = float(pipeline.predict_proba(obs)[0, 1])
    except Exception:
        return jsonify({
            "observation_id": observation_id,
            "error": "Observation is invalid!"
        }), 400

    p = Prediction(
        observation_id=str(observation_id),
        observation=json.dumps(observation),
        proba=probability
        )

    try:
        p.save()
    except IntegrityError:
        return jsonify({
            "observation_id": observation_id,
            "error": f"Observation ID {observation_id} already exists"
        }), 409

    response = {
        "observation_id": observation_id,
        "prediction": prediction,
        "probability": probability
    }

    return jsonify(response), 200


@app.route("/update", methods=["POST"])
def update():
    obs = request.get_json(silent=True)

    if not isinstance(obs, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    if "observation_id" not in obs:
        return jsonify({"error": "Missing field: observation_id"}), 400

    if "true_class" not in obs:
        return jsonify({
            "observation_id": obs.get("observation_id"),
            "error": "Missing field: true_class"
        }), 400

    observation_id = str(obs["observation_id"])

    try:
        p = Prediction.get(Prediction.observation_id == observation_id)
        p.true_class = int(obs["true_class"])
        p.save()
        return jsonify(model_to_dict(p)), 200
    except ValueError:
        return jsonify({
            "observation_id": observation_id,
            "error": "true_class must be numeric"
        }), 400
    except Prediction.DoesNotExist:
        return jsonify({
            "observation_id": observation_id,
            "error": f"Observation ID {observation_id} does not exist"
        }), 404


@app.route("/list-db-contents", methods=["GET"])
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ]), 200


# End webserver stuff
########################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, port=port)