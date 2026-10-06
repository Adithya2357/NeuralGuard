# syntax=docker/dockerfile:1
#
# NeuralGuard application image (used by the "demo" compose profile).
#
#   docker build -t neuralguard .
#   docker run --rm neuralguard demo                 # zero-infrastructure end-to-end run
#   docker run --rm neuralguard detect --help
#
# A model is trained at build time, so the image is self-contained: the detector works
# without mounting anything. Mount your own model over /app/models (or point
# NEURALGUARD_MODEL_PATH elsewhere) to use a different one.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Unprivileged runtime user (fixed UID/GID so volume permissions are predictable).
RUN groupadd --system --gid 10001 neuralguard \
    && useradd --system --uid 10001 --gid neuralguard --create-home \
       --home-dir /home/neuralguard --shell /usr/sbin/nologin neuralguard

# Dependencies first, in their own layer: editing the code does not reinstall them.
# requirements.txt mirrors the runtime dependencies in pyproject.toml.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then the package itself (--no-deps: everything it needs is installed above, and
# 'pip check' fails the build if requirements.txt ever drifts from pyproject.toml).
COPY pyproject.toml README.md LICENSE ./
COPY neuralguard ./neuralguard
RUN pip install --no-cache-dir --no-deps . \
    && pip check \
    && rm -rf build ./*.egg-info

# Train the model as root, so the runtime user can read it but not replace it
# (the model is a joblib pickle; whoever can write it can run code in the detector).
RUN neuralguard train --samples 40000 --output /app/models/threat_model.joblib \
    && chmod 0755 /app/models \
    && chmod 0644 /app/models/*

ENV NEURALGUARD_MODEL_PATH=/app/models/threat_model.joblib

USER neuralguard:neuralguard

ENTRYPOINT ["neuralguard"]
CMD ["detect"]
