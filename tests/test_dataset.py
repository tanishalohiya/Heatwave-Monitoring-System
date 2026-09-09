import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from services.preprocessing import (
    load_dataset,
    clean_dataset,
    compute_climatology,
    add_normal_temperature,
)


def test_dataset_loading():
    df = load_dataset("data/Rajasthan_Heatwave_2006_2025.csv")
    assert df is not None
    assert len(df) > 0


def test_cleaning_removes_duplicates():
    df = pd.DataFrame({
        "YEAR": [2024, 2024, 2024],
        "MONTH": [5, 5, 5],
        "DAY": [1, 1, 2],
        "DISTRICT": ["Jaipur", "Jaipur", "Jaipur"],
        "TMAX": [40, 40, 41],
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2


def test_cleaning_interpolates_missing_values_per_district():
    df = pd.DataFrame({
        "YEAR": [2024] * 6,
        "MONTH": [5] * 6,
        "DAY": [1, 2, 3, 1, 2, 3],
        "DISTRICT": ["Jaipur", "Jaipur", "Jaipur", "Jodhpur", "Jodhpur", "Jodhpur"],
        "TMAX": [40.0, np.nan, 44.0, 30.0, np.nan, 34.0],
    })
    cleaned = clean_dataset(df)

    jaipur_day2 = cleaned[(cleaned["DISTRICT"] == "Jaipur") & (cleaned["DAY"] == 2)]["TMAX"].iloc[0]
    jodhpur_day2 = cleaned[(cleaned["DISTRICT"] == "Jodhpur") & (cleaned["DAY"] == 2)]["TMAX"].iloc[0]

    # Interpolated between district-specific neighbours, not a global mean
    assert jaipur_day2 == 42.0
    assert jodhpur_day2 == 32.0


def test_climatology_and_merge_shapes():
    df = load_dataset("data/Rajasthan_Heatwave_2006_2025.csv")
    df = clean_dataset(df)

    climatology = compute_climatology(df)
    assert set(["DISTRICT", "DAY_OF_YEAR", "NORMAL_TMAX"]).issubset(climatology.columns)
    assert climatology["DAY_OF_YEAR"].max() <= 366

    merged = add_normal_temperature(df, climatology)
    assert "NORMAL_TMAX" in merged.columns
    assert len(merged) == len(df)