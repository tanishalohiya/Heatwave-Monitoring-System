def generate_alert(prediction):
    """
    Generate an alert message based on the prediction.
    """
    if prediction == 1:
        return "⚠ Heatwave Alert! Stay hydrated and avoid outdoor activities."

    return "✅ Normal Weather. No Heatwave Detected."