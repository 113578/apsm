FROM python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/src/app
