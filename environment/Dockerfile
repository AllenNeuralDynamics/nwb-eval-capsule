# hash:sha256:7d65a2189215764e472421324533d7ec6b9a7f6201730b168bb4351ef922a669
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
    lazynwb==0.2.83 \
    pydantic-settings==2.11.0
