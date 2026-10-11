
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heatwave_dnn.keras"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "dnn_preprocessor.joblib"


@st.cache_resource
def load_dnn_model():
    """Load the trained DNN and its saved preprocessing bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained DNN model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"DNN preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    bundle = joblib.load(PREPROCESSOR_PATH)

    return model, bundle


def predict_dnn(weather_data):
    """
    Classify same-day heatwave conditions from weather inputs.

    Parameters
    ----------
    weather_data : dict or pandas.DataFrame
        Must contain all 20 features expected by the saved model.
        Temperature inputs must be in Celsius.

    Returns
    -------
    dict
        Probability, predicted class, and threshold.
    """
    model, bundle = load_dnn_model()
    features = bundle["features"]
    threshold = float(bundle["threshold"])

    if isinstance(weather_data, dict):
        input_df = pd.DataFrame([weather_data])
    elif isinstance(weather_data, pd.DataFrame):
        input_df = weather_data.copy()
    else:
        raise TypeError(
            "weather_data must be a dictionary or pandas DataFrame."
        )

    if input_df.empty:
        raise ValueError("At least one weather record is required.")

    missing = [feature for feature in features if feature not in input_df.columns]
    if missing:
        raise ValueError(
            f"Missing required DNN features: {', '.join(missing)}"
        )

    input_df = input_df[features].apply(
        pd.to_numeric, errors="raise"
    )

    if input_df.empty:
        raise ValueError("At least one weather record is required.")

    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        raise ValueError("Weather inputs must contain finite numeric values.")

    processed = bundle["preprocessor"].transform(input_df)
    probabilities = model.predict(processed, verbose=0).reshape(-1)

    results = pd.DataFrame({
        "HEATWAVE_PROBABILITY": probabilities,
        "HEATWAVE_PREDICTED": (
            probabilities >= threshold
        ).astype(int),
    })

    return {
        "probability": float(results.iloc[0]["HEATWAVE_PROBABILITY"]),
        "prediction": int(results.iloc[0]["HEATWAVE_PREDICTED"]),
        "threshold": threshold,
    }
