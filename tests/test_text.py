import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import TextPreprocessingUtils
from backend.services import text_service

def test_text_preprocessing():
    raw = "He's totally lost it, don't you think?"
    clean = TextPreprocessingUtils.clean_and_normalize(raw)
    assert "he is" in clean.lower()
    assert "do not" in clean.lower()

def test_text_inference_valid():
    res_joy = text_service.predict_from_text("I am so incredibly happy and thrilled with our success!")
    assert res_joy["status"] == "success"
    assert res_joy["predicted_class"] == "joy"
    assert res_joy["confidence"] > 0.40
    assert len(res_joy["probabilities"]) == 7

    res_anger = text_service.predict_from_text("I am furious, stop talking to me right now!")
    assert res_anger["status"] == "success"
    assert res_anger["predicted_class"] == "anger"

def test_text_inference_empty_error():
    try:
        text_service.predict_from_text("    ")
        raise AssertionError("Expected ValueError on empty text.")
    except ValueError:
        pass

if __name__ == "__main__":
    test_text_preprocessing()
    test_text_inference_valid()
    test_text_inference_empty_error()
    print("All text unit tests passed!")
