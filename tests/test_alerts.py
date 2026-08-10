import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.alerts import generate_alert


def test_heatwave_alert():
    message = generate_alert(1)

    assert "Heatwave" in message


def test_normal_weather():
    message = generate_alert(0)

    assert "Normal" in message