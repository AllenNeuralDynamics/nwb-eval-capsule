# hash:sha256:3d767a822e86517a502ac515bfc8758989e7a129ea7db08aec6cd822352697b8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    anthropic==0.86.0 \
    eval-type-backport==0.3.1 \
    lazynwb==0.2.84 \
    pydantic-settings==2.11.0
