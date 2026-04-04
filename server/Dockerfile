# Base image can be overridden during CI or Spaces builds if needed.
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# Use a dedicated work directory for the build stage.
WORKDIR /app

# Install the small set of system tools required during dependency setup.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy the full environment source into the builder image.
COPY . /app/env
WORKDIR /app/env

# Ensure `uv` is present even when the base image does not provide it.
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install project dependencies and the project itself into `.venv`.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Start the smaller runtime stage from the same base image family.
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the prepared virtual environment and application code from the builder.
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Make the runtime use the project virtual environment and source tree.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=32

# Health checks use the custom REST health route exposed by the app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn from the project directory so relative imports resolve cleanly.
# The container honors deployment-time environment variables for host, port,
# worker count, and session limits.
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
