FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/artifacts/incentive_recommender.joblib

EXPOSE 8000

CMD ["uvicorn", "incentive_model.api:app", "--host", "0.0.0.0", "--port", "8000"]
