# ---- Full target: all models including 40B/20B (requires Hopper GPU for FP8) ----
FROM nvcr.io/nvidia/pytorch:25.04-py3 AS full

LABEL org.opencontainers.image.source="https://github.com/ylab-hi/EasyEvo2" \
      org.opencontainers.image.description="EasyEvo2 with Evo 2 — all models (FP8/Hopper)" \
      org.opencontainers.image.licenses="MIT"

ENV HF_HOME=/app/.cache/huggingface \
    PIP_ROOT_USER_ACTION=ignore

RUN pip install --no-cache-dir evo2

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

# Verify installation
RUN easyevo2 list-models

ENTRYPOINT ["easyevo2"]
CMD ["--help"]

# ---- Light target: 7B models only (works on any CUDA GPU, no FP8 needed) ----
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS light

LABEL org.opencontainers.image.source="https://github.com/ylab-hi/EasyEvo2" \
      org.opencontainers.image.description="EasyEvo2 with Evo 2 — 7B models (any CUDA GPU)" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch BEFORE flash-attn (flash-attn compiles against torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation
RUN pip install --no-cache-dir evo2

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

# Verify installation
RUN easyevo2 list-models

ENTRYPOINT ["easyevo2"]
CMD ["--help"]
