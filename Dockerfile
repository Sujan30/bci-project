# Multi-stage Dockerfile for sleep-bci project
# Stage 1: Builder - Install dependencies and create wheels
FROM python:3.11 as builder

WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./
COPY src/ ./src/

# Build wheels for all dependencies
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for data and models
RUN mkdir -p /app/data/raw /app/data/processed /app/data/results /app/models

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command: start API server
CMD ["sleepbci-serve"]
