import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.prediction import predict_heatwave


def test_heatwave_prediction():
    assert predict_heatwave(47) == 1


def test_normal_prediction():
    assert predict_heatwave(38) == 0