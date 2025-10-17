FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app

EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
