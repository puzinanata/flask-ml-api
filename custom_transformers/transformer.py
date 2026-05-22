import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class DeploymentFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.required_columns = [
            "call_type",
            "call_type_group",
            "original_priority",
            "unit_id",
            "unit_type",
            "station_area",
            "battalion",
            "neighborhood_district",
            "zipcode_of_incident",
            "received_dttm",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Keep only raw API/deployment columns
        df = df[self.required_columns].copy()

        # Convert received datetime
        df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")

        # Clean priority
        df["priority_clean"] = np.where(
            df["original_priority"].astype(str).isin(["2", "3"]),
            df["original_priority"].astype(str),
            "Other"
        )

        # Time features
        df["hour"] = df["received_dttm"].dt.hour
        df["day_of_week"] = df["received_dttm"].dt.day_name()
        df["month"] = df["received_dttm"].dt.month_name()
        df["year"] = df["received_dttm"].dt.year

        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
        df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18])
        df["night_shift"] = (df["hour"] >= 22) | (df["hour"] < 6)

        # Treat identifiers as categorical strings
        # Station area
        df["station_area"] = (
            pd.to_numeric(df["station_area"], errors="coerce")
            .round()
            .astype(str)
        )

        # Zipcode
        df["zipcode_of_incident"] = (
            pd.to_numeric(df["zipcode_of_incident"], errors="coerce")
            .round()
            .astype(str)
        )

        # Final model features only
        return df[
            [
                "call_type",
                "call_type_group",
                "priority_clean",
                "unit_type",
                "station_area",
                "battalion",
                "neighborhood_district",
                "zipcode_of_incident",
                "hour",
                "day_of_week",
                "month",
                "year",
                "is_weekend",
                "rush_hour",
                "night_shift",
            ]
        ]
