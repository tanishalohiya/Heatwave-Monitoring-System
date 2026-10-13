
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from services import ml_models


def evaluate_comparison(district=None, test_year=None):
    original_features = ml_models.FEATURES.copy()

    try:
        # Exclude TMAX to reduce feature leakage
        ml_models.FEATURES = [
            feature for feature in original_features
            if feature != "TMAX"
        ]

        # Load dataset
        if district is not None:
            source = pd.read_csv(
                "/content/Heatwave-Monitoring-System/data/"
                "Rajasthan_Heatwave_2006_2025.csv"
            )
            source["YEAR"] = pd.to_numeric(
                source["YEAR"], errors="coerce"
            )
            source["HEATWAVE"] = pd.to_numeric(
                source["HEATWAVE"], errors="coerce"
            )

            source = source[
                (source["DISTRICT"] == district)
                & source["YEAR"].notna()
                & source["HEATWAVE"].notna()
            ].copy()

            source["YEAR"] = source["YEAR"].astype(int)
            source["HEATWAVE"] = source["HEATWAVE"].astype(int)

            df = source[
                ml_models.FEATURES + ["YEAR", "HEATWAVE"]
            ].copy()
        else:
            df = ml_models.load_weather_data()

        # Evaluate a specific year using only earlier years for training
        if test_year is not None:
            test_year = int(test_year)
            train = df[df["YEAR"] < test_year].copy()
            test = df[df["YEAR"] == test_year].copy()

            if train.empty or test.empty:
                raise ValueError(
                    f"No sufficient data for {district}, {test_year}."
                )

            if train["HEATWAVE"].nunique() < 2:
                raise ValueError(
                    "Training data must contain both heatwave classes."
                )

            if test["HEATWAVE"].nunique() < 2:
                raise ValueError(
                    "The selected test year contains only one class; "
                    "precision, recall and F1 may not be informative."
                )

            X_train = train[ml_models.FEATURES]
            y_train = train["HEATWAVE"]
            X_test = test[ml_models.FEATURES]
            y_test = test["HEATWAVE"]

            results = []
            fitted_models = {}

            for name, model in ml_models.build_models().items():
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

            return (
                pd.DataFrame(results),
                fitted_models,
                y_test,
                [test_year],
            )

        return ml_models.evaluate_models(df)

    finally:
        ml_models.FEATURES = original_features
