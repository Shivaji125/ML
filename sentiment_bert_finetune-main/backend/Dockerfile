FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies — CPU-only torch from the official wheel index
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the pre-downloaded model (run `python model.py` first)
COPY ./model /app/model

# Copy application code
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
