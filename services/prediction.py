def predict_heatwave(max_temperature):
    """
    Simple rule-based prediction.
    Returns:
        1 -> Heatwave
        0 -> Normal
    """
    if max_temperature >= 45:
        return 1

    return 0