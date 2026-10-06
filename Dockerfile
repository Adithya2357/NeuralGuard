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
#
# Live capture needs root inside the container (the image's user has no capabilities,
# and --cap-add NET_RAW alone does not give it any) and the host's network:
#   docker run --rm --user root --network host neuralguard produce --source live --interface eth0

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
# Same settings as `neuralguard train`: a smaller training set misses attack variants.
RUN neuralguard train --output /app/models/threat_model.joblib \
    && chmod 0755 /app/models \
    && chmod 0644 /app/models/*

ENV NEURALGUARD_MODEL_PATH=/app/models/threat_model.joblib

USER neuralguard:neuralguard

# As PID 1 the CLI ignores signals it has no handler for, and it only installs its
# SIGTERM handler once it starts: Python always handles SIGINT, so docker stop works
# from the first moment (and is a graceful shutdown once the command runs).
STOPSIGNAL SIGINT

ENTRYPOINT ["neuralguard"]
CMD ["detect"]
