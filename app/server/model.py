"""
model.py

This module defines a SentimentModelLoader class that loads a pre-trained
scikit-learn pipeline from disk and provides methods to predict sentiment
labels for input text. The pipeline is assumed to be saved as a 'model.pkl'
file in a sibling 'model/' directory.
"""

import os
import joblib
import pandas as pd  # pandas is imported in case future methods require DataFrame inputs

# -----------------------------------------------------------------------------
# 1. File and Model Path Configuration
# -----------------------------------------------------------------------------
# Determine the directory containing this script
BASE_DIR = os.path.dirname(__file__)                

# Path to the folder where the model artifact lives
MODEL_DIR = os.path.join(BASE_DIR, "model")         

# Full path to the serialized sklearn Pipeline (.pkl file)
MODEL_PKL = os.path.join(MODEL_DIR, "model.pkl")    

# -----------------------------------------------------------------------------
# 2. SentimentModelLoader Class
# -----------------------------------------------------------------------------
class SentimentModelLoader:
    """
    Loader for a pre-trained sentiment analysis pipeline.

    On initialization, this class loads a scikit-learn Pipeline object
    from disk using joblib. It then provides:
      - predict(text): returns the integer sentiment label (0/1/2).
      - predict_label(text): returns a human-readable label string.
    """

    def __init__(self):
        """
        Load the serialized sklearn Pipeline from MODEL_PKL.

        Raises:
            FileNotFoundError: if MODEL_PKL does not exist.
            Any joblib loading errors if the file is corrupted.
        """
        # Load the Pipeline into memory once at startup
        self.pipeline = joblib.load(MODEL_PKL)

    def predict(self, text: str) -> int:
        """
        Predict the numeric sentiment label for a single text string.

        Args:
            text: A single input sentence to classify.

        Returns:
            An integer label:
                0 => Negative
                1 => Neutral
                2 => Positive

        Raises:
            ValueError: if the pipeline returns a non-integer or empty result.
        """
        # The pipeline expects an iterable of texts, so we wrap text in a list
        prediction_array = self.pipeline.predict([text])
        # Extract the first element and cast to int
        return int(prediction_array[0])

    def predict_label(self, text: str) -> str:
        """
        Predict a human-readable sentiment label for a single text string.

        Args:
            text: A single input sentence to classify.

        Returns:
            A string label:
                "Negative", "Neutral", or "Positive".
            If the numeric label is out of the expected range, returns "Unknown".
        """
        # Map numeric prediction to descriptive string
        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        numeric_label = self.predict(text)
        return label_map.get(numeric_label, "Unknown")
