# app/prediction_pipeline.py

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# ==============================================================
# FUNCTIONAL INTERFACE
# ==============================================================

def load_model(model_path=None):
    """Load saved fraud detection model and threshold safely."""
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../models/fraud_model.pkl")
    model_path = os.path.normpath(model_path)

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["model"], model_data["threshold"]


def make_predictions(df, model, threshold=0.5):
    """Quick functional API for prediction on new data."""
    pipeline = FraudModelPipeline()  # reuse preprocessing logic
    pipeline.model = model
    pipeline.threshold = threshold
    result, preds = pipeline.predict(df)
    return result, preds


# ==============================================================
# OOP CLASS (used internally or in advanced workflows)
# ==============================================================

class FraudModelPipeline:
    def __init__(self, model_path=None):
        """Initialize and load saved model + threshold."""
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../models/fraud_model.pkl")
        self.model_path = os.path.normpath(model_path)

        self.model = None
        self.threshold = 0.5
        self.feature_cols = [
            'amt', 'merchant', 'category', 'gender', 'city', 'state', 'job',
            'age', 'hour', 'day', 'weekday', 'distance'
        ]
        self.label_encoders = {}
        self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.threshold = model_data["threshold"]

    def _feature_engineering(self, df):
        """Add derived features."""
        df = df.copy()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df['weekday'] = df['trans_date_trans_time'].dt.weekday

        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = ((datetime.now() - df['dob']).dt.days / 365.25).astype(int)

        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2
        )
        return df

    def _encode_categorical(self, df):
        cat_cols = df.select_dtypes("object").columns
        for col in cat_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return df

    def preprocess(self, df):
        df = self._feature_engineering(df)
        df = df[self.feature_cols].copy()

        cat_cols = df.select_dtypes("object").columns
        num_cols = df.select_dtypes(exclude="object").columns

        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        df = self._encode_categorical(df)
        return df

    def predict(self, df):
        clean_df = self.preprocess(df)
        y_proba = self.model.predict_proba(clean_df)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        result = df.copy()
        result["fraud_probability"] = y_proba
        result["fraud_prediction"] = y_pred
        return result, y_pred


# ==============================================================
# Example standalone test
# ==============================================================

if __name__ == "__main__":
    df = pd.read_csv("../data/sample_new_data.csv")
    model, threshold = load_model()
    result_df, preds = make_predictions(df, model, threshold)
    print(result_df.head())
