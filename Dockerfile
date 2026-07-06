FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN python3 -m pip install --index-url ${PYTORCH_INDEX_URL} torch torchvision torchaudio

COPY docker/requirements-server.txt /tmp/requirements-server.txt
RUN python3 -m pip install -r /tmp/requirements-server.txt

COPY . /app

RUN mkdir -p /app/data /app/recbole_results

CMD ["python3", "src/recbole_framework/tuning/run_server_full_experiments.py"]
