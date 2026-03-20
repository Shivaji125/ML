"""
Download the fine-tuned sentiment model from HuggingFace Hub and save locally.

Usage:
    1. Copy .env.example to .env and add your HuggingFace token
    2. Run:  python model.py

The model will be saved to ./model/
"""

import os
import sys

from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

MODEL_NAME = "Wolverine001/bert_finetuned_senti"
SAVE_DIR = "./model"


def main():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        print("ERROR: HUGGINGFACEHUB_API_TOKEN not found.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your HuggingFace token to the .env file")
        sys.exit(1)

    print(f"Downloading model '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=token)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
