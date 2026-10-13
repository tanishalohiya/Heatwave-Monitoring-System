from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

FEATURES = [
    "MONTH", "DAY", "WIND_U10", "WIND_V10", "MSLP", "BLH", "GEOP",
    "TEMP2M", "TMAX", "TMIN", "DEW2M", "CLOUD", "RAIN", "SRAD",
    "EVAP", "SOILT1", "SOILM1", "LAI", "LAT", "LON",
]


def load_weather_data(path=None):
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "data" / "Rajasthan_Heatwave_2006_2025.csv"
        )

    df = pd.read_csv(path)
    required = FEATURES + ["YEAR", "HEATWAVE"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].copy()
    df["HEATWAVE"] = pd.to_numeric(df["HEATWAVE"], errors="coerce")
    df = df.dropna(subset=["HEATWAVE"])
    df["HEATWAVE"] = df["HEATWAVE"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values("YEAR").reset_index(drop=True)


def build_models():
    return {
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=10, class_weight="balanced", random_state=42
            )),
        ]),
        "Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf", class_weight="balanced", random_state=42
            )),
        ]),
    }


def evaluate_models(df=None):
    if df is None:
        df = load_weather_data()

    years = sorted(df["YEAR"].dropna().unique())
    if len(years) < 2:
        raise ValueError("At least two distinct years are required.")

    split_index = max(1, int(len(years) * 0.8))
    split_index = min(split_index, len(years) - 1)
    test_years = years[split_index:]

    train = df[~df["YEAR"].isin(test_years)]
    test = df[df["YEAR"].isin(test_years)]

    if train.empty or test.empty or train["HEATWAVE"].nunique() < 2:
        raise ValueError("Insufficient data for a chronological train/test split.")

    X_train, y_train = train[FEATURES], train["HEATWAVE"]
    X_test, y_test = test[FEATURES], test["HEATWAVE"]

    results = []
    fitted_models = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Test Samples": len(y_test),
        })
        fitted_models[name] = model

    return pd.DataFrame(results), fitted_models, y_test, test_years
