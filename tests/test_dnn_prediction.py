
import numpy as np
import pandas as pd
import pytest

from services import dnn_prediction


@pytest.fixture
def valid_weather_data():
    return {
        "MONTH": 5,
        "DAY": 15,
        "WIND_U10": 2.0,
        "WIND_V10": 1.0,
        "MSLP": 1005.0,
        "BLH": 1200.0,
        "GEOP": 5800.0,
        "TEMP2M": 35.0,
        "TMAX": 42.0,
        "TMIN": 28.0,
        "DEW2M": 20.0,
        "CLOUD": 0.2,
        "RAIN": 0.0,
        "SRAD": 250.0,
        "EVAP": 5.0,
        "SOILT1": 32.0,
        "SOILM1": 0.15,
        "LAI": 1.2,
        "LAT": 26.9,
        "LON": 70.9,
    }


def test_predict_dnn_with_valid_dictionary(valid_weather_data):
    class FakePreprocessor:
        def transform(self, data):
            return data.to_numpy(dtype=float)

    class FakeModel:
        def predict(self, data, verbose=0):
            return np.array([[0.8]])

    dnn_prediction.load_dnn_model = lambda: (
        FakeModel(),
        {
            "features": list(valid_weather_data.keys()),
            "threshold": 0.5,
            "preprocessor": FakePreprocessor(),
        },
    )

    result = dnn_prediction.predict_dnn(valid_weather_data)

    assert result["probability"] == pytest.approx(0.8)
    assert result["prediction"] == 1
    assert result["threshold"] == 0.5


def test_predict_dnn_with_dataframe(valid_weather_data):
    class FakePreprocessor:
        def transform(self, data):
            return data.to_numpy(dtype=float)

    class FakeModel:
        def predict(self, data, verbose=0):
            return np.array([[0.2]])

    dnn_prediction.load_dnn_model = lambda: (
        FakeModel(),
        {
            "features": list(valid_weather_data.keys()),
            "threshold": 0.5,
            "preprocessor": FakePreprocessor(),
        },
    )

    result = dnn_prediction.predict_dnn(pd.DataFrame([valid_weather_data]))

    assert result["probability"] == pytest.approx(0.2)
    assert result["prediction"] == 0


def test_predict_dnn_rejects_missing_features(valid_weather_data):
    incomplete_data = valid_weather_data.copy()
    incomplete_data.pop("TMAX")

    with pytest.raises(ValueError, match="Missing required DNN features"):
        dnn_prediction.predict_dnn(incomplete_data)


def test_predict_dnn_rejects_non_finite_values(valid_weather_data):
    invalid_data = valid_weather_data.copy()
    invalid_data["TMAX"] = float("nan")

    with pytest.raises(ValueError, match="finite numeric values"):
        dnn_prediction.predict_dnn(invalid_data)


def test_predict_dnn_rejects_unsupported_input():
    with pytest.raises(TypeError, match="dictionary or pandas DataFrame"):
        dnn_prediction.predict_dnn([1, 2, 3])


def test_predict_dnn_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="At least one weather record"):
        dnn_prediction.predict_dnn(pd.DataFrame())
