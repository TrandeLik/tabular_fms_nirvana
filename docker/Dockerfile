FROM python:3.12-slim-bookworm

LABEL description="tabular_fms_nirvana"
LABEL version="0.0.1"

# 📦
ARG _BOX="/box"
RUN mkdir -p ${_BOX}

# Common tools
RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    cmake \
    coreutils \
    g++ \
    gdb \
    git \
    libaio-dev \
    moreutils \
    nano \
    ncdu \
    software-properties-common \
    tree \
    vim \
    wget

# uv
COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR="${_BOX}/uv-python"
ENV UV_PROJECT_ENVIRONMENT="${_BOX}/.venv"

# Python environment
COPY uv.lock pyproject.toml ${_BOX}/
RUN uv sync --project ${_BOX} --no-install-project --frozen --managed-python --link-mode copy --no-cache

# Permissions
RUN chmod -R a+rwX ${_BOX}
