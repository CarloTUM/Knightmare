# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install -e .

# Drop privileges.
RUN useradd --create-home --uid 1000 knightmare
USER knightmare

ENTRYPOINT ["knightmare"]
CMD ["--help"]
