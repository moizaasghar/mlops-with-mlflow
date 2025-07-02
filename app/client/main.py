"""
This script defines a Gradio-based frontend for a sentiment classification model.
It sends user input to a FastAPI backend endpoint and displays the predicted sentiment.
"""

import gradio as gr
import requests

# URL of the FastAPI backend prediction endpoint
BACKEND_URL = "http://127.0.0.1:8000/predict"

def query_backend(text: str) -> str:
    """
    Send a POST request to the backend with the user-provided text
    and return the predicted sentiment label.

    Args:
        text: The input text string to classify.

    Returns:
        The sentiment label returned by the backend (e.g., "Positive"),
        or an error message if the request fails.
    """
    try:
        # Prepare the JSON payload for the request
        payload = {"text": text}

        # Execute the HTTP POST request
        response = requests.post(BACKEND_URL, json=payload)

        # If the backend returns HTTP 200, parse and return the sentiment
        if response.status_code == 200:
            result = response.json()
            return result.get("sentiment", "Unknown")

        # Otherwise, return an error with the status code
        return f"Error: Received status code {response.status_code}"
    except requests.RequestException as e:
        # Catch network/connection errors and return a descriptive message
        return f"Error: {e}"

# Define the Gradio interface
iface = gr.Interface(
    fn=query_backend,                        # Function to call on user input
    inputs=gr.Textbox(
        lines=2,
        placeholder="Type a sentence here...",
        label="Enter Text"
    ),
    outputs=gr.Label(label="Predicted Sentiment"),
    title="Sentiment Classifier",
    description=(
        "Frontend for a sentiment analysis model served via FastAPI. "
        "Enter your text and click Submit to see the predicted sentiment."
    ),
    allow_flagging="never"                   # Disable Gradio's flagging feature
)

if __name__ == "__main__":
    # Launch the Gradio app on port 7860; set share=True to generate a public link
    iface.launch(server_port=7860, share=False)
