import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.preprocessing import load_dataset, clean_dataset


def test_dataset_loading():
    df = load_dataset("data/Rajasthan_Heatwave_2006_2025.csv")

    assert df is not None
    assert len(df) > 0


def test_dataset_cleaning():
    df = load_dataset("data/Rajasthan_Heatwave_2006_2025.csv")
    cleaned_df = clean_dataset(df)

    # Cleaned dataset should still contain data
    assert len(cleaned_df) > 0

    # Dataset should not be None
    assert cleaned_df is not None