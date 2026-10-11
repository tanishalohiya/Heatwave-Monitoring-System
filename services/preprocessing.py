import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load the dataset from CSV and convert
    temperature values from Kelvin to Celsius.
    """
    df = pd.read_csv(file_path)

    temperature_cols = [
        "TEMP2M",
        "TMAX",
        "TMIN",
        "DEW2M"
    ]

    for col in temperature_cols:
        if col in df.columns:
            df[col] = df[col] - 273.15

    return df


def clean_dataset(df):
    """
    Clean the dataset:
    - Remove duplicate rows
    - Interpolate missing numeric values along the date axis, per district
      (keeps real trends instead of flattening a missing hot day to the
      column-wide mean)
    - Fall back to the overall column mean only for values interpolation
      can't reach (e.g. leading/trailing gaps in a district's series)
    """
    df = df.drop_duplicates().copy()

    if not {"YEAR", "MONTH", "DAY", "DISTRICT"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain YEAR, MONTH, DAY, DISTRICT columns."
        )

    if "DATE" not in df.columns:
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])

    df = df.sort_values(["DISTRICT", "DATE"])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("YEAR", "MONTH", "DAY")]

    df[numeric_cols] = (
        df.groupby("DISTRICT")[numeric_cols]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )

    remaining_na = df[numeric_cols].isna().any()
    for col in remaining_na[remaining_na].index:
        df[col] = df[col].fillna(df[col].mean())

    return df


def compute_climatology(df, window_days=7):
    """
    Compute a day-of-year "normal" TMAX per district, smoothed with a
    +/- window_days rolling window (wrapped across the year boundary) so
    that a single unusually hot or cool year doesn't distort the baseline.

    Returns a DataFrame with columns: DISTRICT, DAY_OF_YEAR, NORMAL_TMAX
    """
    data = df.copy()
    data["DAY_OF_YEAR"] = data["DATE"].dt.dayofyear

    daily_avg = (
        data.groupby(["DISTRICT", "DAY_OF_YEAR"])["TMAX"]
        .mean()
        .reset_index()
    )

    smoothed_frames = []
    for district, group in daily_avg.groupby("DISTRICT"):
        group = group.set_index("DAY_OF_YEAR").reindex(range(1, 367))
        padded = pd.concat(
            [
                group["TMAX"].tail(window_days),
                group["TMAX"],
                group["TMAX"].head(window_days),
            ]
        )
        smoothed = padded.rolling(
            window=window_days * 2 + 1, center=True, min_periods=1
        ).mean()
        smoothed = smoothed.iloc[window_days: window_days + 366]

        smoothed_frames.append(
            pd.DataFrame(
                {
                    "DISTRICT": district,
                    "DAY_OF_YEAR": range(1, 367),
                    "NORMAL_TMAX": smoothed.values,
                }
            )
        )

    return pd.concat(smoothed_frames, ignore_index=True)


def add_normal_temperature(df, climatology):
    """
    Merge the day-of-year climatological normal onto each row.
    """
    data = df.copy()
    data["DAY_OF_YEAR"] = data["DATE"].dt.dayofyear
    data = data.merge(climatology, on=["DISTRICT", "DAY_OF_YEAR"], how="left")
    return data