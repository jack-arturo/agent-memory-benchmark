# syntax=docker/dockerfile:1
# Runner image for the Agent Memory Benchmark (omb) harness.
#
# Bundles the harness + uv + a Docker CLI/Compose plugin. The AutoMem provider is
# self-spinning: from inside this container it runs `docker compose` against the
# mounted host daemon to bring up AutoMem + FalkorDB + Qdrant as sibling
# containers, then reaches them through the host gateway (AUTOMEM_HOST=
# host.docker.internal). A reproducer needs only Docker + a GEMINI_API_KEY — no
# host Python/uv toolchain.
#
# Pinned to -bookworm because Docker's apt repo publishes for it (the default
# slim base tracks trixie, which Docker's repo lags).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# Docker CLI + Compose v2 plugin (to drive the host daemon via the mounted
# socket), git for any VCS deps, and build-essential as insurance for any
# sdist-only transitive dep in the harness's wide provider tree.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git build-essential \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-compose-plugin \
 && rm -rf /var/lib/apt/lists/*

# uv for fast, lockfile-faithful installs.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency layer first (cached until pyproject/uv.lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project source + finalize the install.
COPY . .
RUN uv sync --frozen

# Default to help; the Makefile / compose override with the real command.
CMD ["omb", "--help"]
