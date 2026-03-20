import os

import gradio as gr
import requests

# Configurable backend URL — defaults to Docker service name,
# override with API_URL env var for local development.
# API_URL = os.getenv("API_URL", "http://backend:8000/predict")
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

EXAMPLES = [
    "I absolutely love this product, it changed my life!",
    "The service was terrible and I want a refund.",
    "It was okay, nothing special.",
    "This is the best movie I have ever seen!",
    "I'm not sure how I feel about this update.",
]


def predict(text: str) -> str:
    if not text or not text.strip():
        return "Please enter some text to analyze."

    try:
        response = requests.post(API_URL, json={"text": text}, timeout=30)
        response.raise_for_status()
    except requests.ConnectionError:
        return "Error: Could not connect to the backend. Is the API server running?"
    except requests.Timeout:
        return "Error: The request timed out. Please try again."
    except requests.HTTPError as e:
        return f"Error: Backend returned status {e.response.status_code}."

    result = response.json()
    sentiment = result.get("sentiment", "unknown")
    confidence = result.get("confidence", {})

    # Format output with confidence scores
    lines = [f"Sentiment: {sentiment.upper()}"]
    if confidence:
        lines.append("")
        lines.append("Confidence scores:")
        for label, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            lines.append(f"  {label:<10} {bar} {score:.1%}")

    return "\n".join(lines)


app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter text",
        placeholder="Type a sentence to analyze its sentiment...",
        lines=3,
    ),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analyzer",
    description="Analyze the sentiment of any text using a fine-tuned BERT model. Enter a sentence and click Submit.",
    examples=EXAMPLES,
    allow_flagging="never",
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
