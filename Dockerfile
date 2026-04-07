# ---- Full target: all models including 40B/20B (requires Hopper GPU for FP8) ----
FROM nvcr.io/nvidia/pytorch:25.04-py3 AS full

LABEL org.opencontainers.image.source="https://github.com/ylab-hi/EasyEvo2" \
      org.opencontainers.image.description="EasyEvo2 with Evo 2 — all models (FP8/Hopper)" \
      org.opencontainers.image.licenses="MIT"

ENV HF_HOME=/app/.cache/huggingface \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install evo2

WORKDIR /app
COPY . .
RUN uv pip install .

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
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1

# Install minimal build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (static binary, no dependencies)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python 3.12 via uv and symlink into a stable PATH location
RUN uv python install 3.12 && \
    ln -s $(uv python find 3.12) /usr/local/bin/python3.12 && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python3

# Install torch first (flash-attn compiles against it)
RUN uv pip install --python 3.12 --system torch --index-url https://download.pytorch.org/whl/cu128

# Install flash-attn (long compile, separate layer for caching)
RUN uv pip install --python 3.12 --system flash-attn==2.8.0.post2 --no-build-isolation

# Install evo2
RUN uv pip install --python 3.12 --system evo2

# Install easyevo2
WORKDIR /app
COPY . .
RUN uv pip install --python 3.12 --system .

# Verify installation
RUN easyevo2 list-models

ENTRYPOINT ["easyevo2"]
CMD ["--help"]
