from typing import Dict, Any, List, Optional
import numpy as np

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

class TextPreprocessingUtils:
    """
    Standard text normalization and contraction expansion for conversational dialogue.
    """
    CONTRACTIONS = {
        "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
        "what's": "what is", "where's": "where is", "there's": "there is",
        "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have"
    }

    @classmethod
    def clean_and_normalize(cls, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        normalized = cleaned.replace("\x92", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        for cont, exp in cls.CONTRACTIONS.items():
            normalized = normalized.replace(cont, exp).replace(cont.capitalize(), exp.capitalize())
        return normalized

