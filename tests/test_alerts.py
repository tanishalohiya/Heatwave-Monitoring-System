import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.alerts import generate_alert


def test_normal_weather():
    message = generate_alert(0)
    assert "Normal" in message


def test_heatwave_alert():
    message = generate_alert(1)
    assert "Heatwave" in message
    assert "Severe" not in message


def test_severe_heatwave_alert():
    message = generate_alert(2)
    assert "Severe Heatwave" in message


def test_alert_includes_district_and_date():
    message = generate_alert(1, district="Jaipur", date="10 May 2024")
    assert "Jaipur" in message
    assert "10 May 2024" in message