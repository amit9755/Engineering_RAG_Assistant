# ============================================================
# docker/Dockerfile - Production Container Image
#
# Learning Note:
#   A Dockerfile defines how to package the application into
#   a container image. Key best practices used here:
#
#   1. MULTI-STAGE BUILD: Separate "builder" and "runtime" stages
#      so the final image only contains what's needed to run,
#      not the full build toolchain. Smaller = faster + more secure.
#
#   2. NON-ROOT USER: Running as a non-root user is a security
#      best practice. If the container is compromised, the attacker
#      gets limited privileges.
#
#   3. LAYER CACHING: Copy requirements.txt BEFORE source code
#      so Docker can cache the pip install step. Only if
#      requirements.txt changes will pip re-run.
#
#   4. .dockerignore: Excludes .env, __pycache__, .git, etc.
#      to keep the image small and secure.
# ============================================================

# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download spaCy English model required by Presidio guardrails
# This prevents runtime download failures in offline/restricted environments
RUN python -m spacy download en_core_web_sm

# HuggingFace embedding model is copied from the build context (pre-cached on host).
# The model files are placed in data/hf_cache/ and mounted into the image cache dir.
# This avoids SSL issues when downloading during build in corporate proxy environments.
COPY data/hf_cache/ /root/.cache/huggingface/

# ===== Stage 2: Runtime =====
FROM python:3.11-slim AS runtime

# Security: create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy pre-downloaded models from builder stage
# spaCy model is installed as a Python package (already in .local)
# HuggingFace model cache
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application source code
COPY src/ ./src/
COPY ui/ ./ui/
COPY .env.example ./.env.example

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma_db && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Add user-installed packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check (Cloud Run uses this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Start the FastAPI application
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
