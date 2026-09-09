def generate_alert(day_class, district=None, date=None):
    """
    Generate an alert message based on day classification.

    day_class:
        0 -> Normal
        1 -> Heatwave
        2 -> Severe Heatwave

    Backwards compatible: old code passing 0/1 (from predict_heatwave)
    still works correctly, since 1 maps to the Heatwave branch.
    """
    location = f" in {district}" if district else ""
    when = f" on {date}" if date else ""

    if day_class == 2:
        return (
            f"🔴 Severe Heatwave Alert{location}{when}! "
            "Avoid outdoor activity, stay hydrated, and check on "
            "vulnerable individuals (elderly, children, outdoor workers)."
        )
    if day_class == 1:
        return (
            f"⚠ Heatwave Alert{location}{when}! "
            "Stay hydrated and avoid outdoor activities during peak hours "
            "(12 PM - 4 PM)."
        )
    return f"✅ Normal Weather{location}{when}. No heatwave detected."