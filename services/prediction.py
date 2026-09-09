import pandas as pd


# IMD-style thresholds (plains criteria)
ABSOLUTE_HEATWAVE_TMAX = 45.0          # qualifies only combined with departure check
ABSOLUTE_SEVERE_HEATWAVE_TMAX = 47.0   # heatwave regardless of departure
DEPARTURE_HEATWAVE = 4.5               # deg C above normal
DEPARTURE_SEVERE_HEATWAVE = 6.5        # deg C above normal
MIN_CONSECUTIVE_DAYS = 2


def classify_day(max_temperature, normal_temperature=None):
    """
    Classify a single day using IMD-style criteria.

    Returns:
        0 -> Normal
        1 -> Hot day (heatwave-qualifying, before the consecutive-day check)
        2 -> Severe hot day
    """
    if normal_temperature is not None and not pd.isna(normal_temperature):
        departure = max_temperature - normal_temperature
        if departure >= DEPARTURE_SEVERE_HEATWAVE or max_temperature >= ABSOLUTE_SEVERE_HEATWAVE_TMAX:
            return 2
        if departure >= DEPARTURE_HEATWAVE or max_temperature >= ABSOLUTE_HEATWAVE_TMAX:
            return 1
        return 0

    # Fallback when no climatological normal is available for this day/district
    if max_temperature >= ABSOLUTE_SEVERE_HEATWAVE_TMAX:
        return 2
    if max_temperature >= ABSOLUTE_HEATWAVE_TMAX:
        return 1
    return 0


def predict_heatwave(max_temperature, normal_temperature=None):
    """
    Backwards-compatible single-day prediction (used by the Prediction page,
    where a user enters one temperature with no time series context).
    Returns 1 if the day qualifies as hot/severe, else 0.

    NOTE: this does NOT enforce the 2-consecutive-day IMD rule - that
    requires a time series, so use detect_heatwave_days() for real
    historical/monitoring classification.
    """
    return 1 if classify_day(max_temperature, normal_temperature) >= 1 else 0


def detect_heatwave_days(df, min_consecutive_days=MIN_CONSECUTIVE_DAYS):
    """
    Given a DataFrame sorted by DATE with columns:
        DISTRICT, DATE, TMAX, and optionally NORMAL_TMAX
    adds:
        DAY_CLASS                 -> 0/1/2 per-day classification
        HEATWAVE_PREDICTED        -> 1 only if part of a run of
                                      >= min_consecutive_days qualifying
                                      (class >= 1) days in that district
        SEVERE_HEATWAVE_PREDICTED -> 1 if part of a run of severe
                                      (class == 2) days
    """
    data = df.sort_values(["DISTRICT", "DATE"]).copy()

    has_normal = "NORMAL_TMAX" in data.columns
    normal_values = data["NORMAL_TMAX"].to_numpy() if has_normal else None
    tmax_values = data["TMAX"].to_numpy()

    data["DAY_CLASS"] = [
        classify_day(tmax_values[i], normal_values[i] if has_normal else None)
        for i in range(len(data))
    ]

    def flag_runs(series, class_threshold):
        qualifies = series >= class_threshold
        run_id = (qualifies != qualifies.shift()).cumsum()
        run_length = qualifies.groupby(run_id).transform("size")
        return (qualifies & (run_length >= min_consecutive_days)).astype(int)

    data["HEATWAVE_PREDICTED"] = (
        data.groupby("DISTRICT")["DAY_CLASS"]
        .transform(lambda s: flag_runs(s, class_threshold=1))
    )

    data["SEVERE_HEATWAVE_PREDICTED"] = (
        data.groupby("DISTRICT")["DAY_CLASS"]
        .transform(lambda s: flag_runs(s, class_threshold=2))
    )

    return data