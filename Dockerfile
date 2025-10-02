# ============ Base ============
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps you might need (yaml, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============ Deps layer (better caching) ============
# If you have both pyproject.toml and requirements.txt, we'll install both.
COPY pyproject.toml* poetry.lock* requirements.txt* /app/

# Try pyproject first (PEP 517), then fallback to requirements.
# If you don't use extras like [dev], keep it simple.
RUN --mount=type=cache,target=/root/.cache/pip \
    (pip install . || true) && \
    ( [ -f requirements.txt ] && pip install -r requirements.txt || true )

# ============ App layer ============
COPY . /app

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default directories (reports gets mounted outside in examples)
RUN mkdir -p /app/reports

# Default command just shows help.
# We'll override with `docker run ... --config ...` in docs below.
ENTRYPOINT ["python", "-m", "pip", "show", "strategy-simulator"]
