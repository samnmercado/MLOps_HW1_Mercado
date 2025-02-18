# MLFlow Dockerfile
FROM python:3.12-slim-bookworm

RUN useradd -m appuser
WORKDIR /home/appuser

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    # Install UV as appuser
    su - appuser -c "curl -sSL https://astral.sh/uv/install.sh | sh" && \
    # Remove build dependencies and clean up
    apt-get purge -y --auto-remove curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/home/appuser/.local/bin:$PATH"

USER appuser

RUN uv venv -p python3.12 venv
ENV PATH="/home/appuser/venv/bin:$PATH"

RUN uv pip install --no-cache \
    mlflow[extras]==2.20.2 \
    psycopg2-binary==2.9.10 \
    boto3==1.36.21 \
    cryptography==44.0.1 \
    pymysql==1.1.1