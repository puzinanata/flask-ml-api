import os
import json
import pickle
import joblib
import pandas as pd

print(os.getcwd())

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

print(columns)
print(dtypes)

observation = {
    "age": 45,
    "education": "Bachelors",
    "hours-per-week": 45,
    "native-country": "United-States"
}

obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
print(obs)
print(pipeline.predict_proba(obs))

