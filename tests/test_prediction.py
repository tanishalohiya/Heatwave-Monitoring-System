import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from services.prediction import predict_heatwave, classify_day, detect_heatwave_days


def test_heatwave_prediction_no_normal():
    assert predict_heatwave(47) == 1


def test_normal_prediction_no_normal():
    assert predict_heatwave(38) == 0


def test_boundary_absolute_threshold():
    assert predict_heatwave(45) == 1
    assert predict_heatwave(44.9) == 0


def test_classify_day_with_departure():
    # 40 is below the absolute threshold, but 5 degrees above normal
    # of 35 -> should still classify as heatwave (class 1)
    assert classify_day(40, normal_temperature=35) == 1


def test_classify_day_severe_with_departure():
    # 6.5+ departure -> severe, even though absolute temp is below 47
    assert classify_day(41.5, normal_temperature=35) == 2


def test_classify_day_normal_with_departure():
    assert classify_day(37, normal_temperature=35) == 0


def test_detect_heatwave_requires_two_consecutive_days():
    df = pd.DataFrame({
        "DISTRICT": ["Jaipur"] * 5,
        "DATE": pd.date_range("2024-05-01", periods=5),
        "TMAX": [40, 46, 40, 46, 46],  # single hot day (idx1) should NOT count,
                                       # two consecutive hot days (idx3,4) SHOULD
    })

    result = detect_heatwave_days(df)

    assert result.iloc[1]["HEATWAVE_PREDICTED"] == 0
    assert result.iloc[3]["HEATWAVE_PREDICTED"] == 1
    assert result.iloc[4]["HEATWAVE_PREDICTED"] == 1


def test_detect_heatwave_separates_districts():
    df = pd.DataFrame({
        "DISTRICT": ["Jaipur", "Jaipur", "Jodhpur", "Jodhpur"],
        "DATE": pd.to_datetime(
            ["2024-05-01", "2024-05-02", "2024-05-01", "2024-05-02"]
        ),
        "TMAX": [46, 46, 30, 30],
    })

    result = detect_heatwave_days(df)

    jaipur = result[result["DISTRICT"] == "Jaipur"]
    jodhpur = result[result["DISTRICT"] == "Jodhpur"]

    assert (jaipur["HEATWAVE_PREDICTED"] == 1).all()
    assert (jodhpur["HEATWAVE_PREDICTED"] == 0).all()