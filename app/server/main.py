"""
FastAPI application exposing a sentiment analysis endpoint.
Loads a pre-trained sentiment model and provides a `/predict` route
that accepts raw text and returns the predicted sentiment label.
"""

from ast import List
from fastapi import FastAPI
from pydantic import BaseModel
from pyparsing import Optional
from model import SentimentModelLoader
import uvicorn

# -----------------------------------------------------------------------------
# 1. Application and Model Initialization
# -----------------------------------------------------------------------------
# Create the FastAPI app instance
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for classifying text into Negative, Neutral, or Positive sentiment.",
    version="1.0.0"
)

# Instantiate and load the sentiment model once at startup
model_loader = SentimentModelLoader()

# -----------------------------------------------------------------------------
# 2. Request and Response Schemas
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """
    Defines the JSON payload expected by the `/predict` endpoint.
    Attributes:
        text (str): The input text string to be classified.
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Defines the JSON response returned by the `/predict` endpoint.
    Attributes:
        sentiment (str): The predicted sentiment label.
    """
    sentiment: str

# -----------------------------------------------------------------------------
# 3. Prediction Endpoint
# -----------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sentiment",
    response_description="The sentiment label for the provided text"
)

def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Handle POST requests to /predict.
    1. Extract the 'text' field from the request body.
    2. Use the loaded model to predict the sentiment label.
    3. Wrap the result in a PredictionResponse object.
    """
    # Use the model loader's helper to get human-readable label
    sentiment_label = model_loader.predict_label(request.text)

    # Return the response model (automatically serialized to JSON)
    return PredictionResponse(sentiment=sentiment_label)

# -----------------------------------------------------------------------------
# 4. Application Entry Point with Auto-Reload
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run the app with Uvicorn:
    # - `reload=True` watches source files and reloads on changes (development mode)
    # - Refer to app as "main:app" (module_name:app_variable)
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
