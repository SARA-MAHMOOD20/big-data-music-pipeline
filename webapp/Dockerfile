FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_extraction /app/feature_extraction
COPY recommender/recommend.py /app/recommender/recommend.py
COPY webapp /app/webapp

ENV PYTHONPATH=/app/feature_extraction:/app/recommender
ENV MODELS_DIR=/app/models

WORKDIR /app/webapp
EXPOSE 5000
CMD ["python", "app.py"]
