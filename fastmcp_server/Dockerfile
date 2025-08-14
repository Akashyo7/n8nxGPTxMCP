# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
COPY fastmcp_server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app
COPY . /app/fastmcp_server

EXPOSE 8000
CMD ["uvicorn", "fastmcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
