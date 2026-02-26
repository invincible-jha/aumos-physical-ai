FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for open3d, numpy, and sensor processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM base AS builder

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e ".[dev]" --no-deps && \
    pip install --no-cache-dir -e "."

# ---------------------------------------------------------------------------
# Production stage
# ---------------------------------------------------------------------------
FROM base AS production

# Non-root user for security
RUN useradd --create-home --shell /bin/bash aumos
USER aumos

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --chown=aumos:aumos src/ src/

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "aumos_physical_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
