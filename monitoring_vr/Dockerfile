# Dockerfile (for verify_predictions or retrain_model)
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 
# Copy your app.py, verify_predictions.py, retrain_model.py etc.

# Default command to run (can be overridden by Cloud Run Job)
CMD ["python", "verify_predictions.py"]