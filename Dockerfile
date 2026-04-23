# CPU-only image for development / environments without GPU
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY perceptra_inference/ ./perceptra_inference/
COPY service/ ./service/
COPY config.yaml ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir .[onnx,server]

RUN useradd -m -u 1000 inferencer && chown -R inferencer:inferencer /app
USER inferencer

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python -c "import requests; requests.get('http://localhost:8080/v1/healthz', timeout=5)"

EXPOSE 8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
