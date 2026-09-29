# ==============================================================================
# Precision Agriculture GCS - Production Dockerfile
# Base: Python 3.12 Slim Linux Container
# Runtime: stdlib http.server (zero third-party deps for the cloud dashboard)
# Port: 8000 (local/production); Vercel variant uses port 80 (see Dockerfile.vercel)
# ==============================================================================

FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DATA_DIR=/tmp/gcs

# Install system dependencies (curl for healthcheck probe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -r gcs && useradd -r -g gcs -u 1000 -d /app gcs

WORKDIR /app

# Copy dependency manifest (stdlib-only; pip install is a no-op but keeps the
# pattern consistent with all other Python projects in this workspace)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script and make executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application source and seeded agronomic data assets
COPY gcs_server.py db_init.py prescription_engine.py \
     crop_health_edge.db camera_config.json \
     field_prescription_log.csv field_prescription_map.geojson \
     plant_classes.json \
     /app/

# Create writable state directory and assign ownership
RUN mkdir -p /tmp/gcs && chown -R gcs:gcs /app /tmp/gcs

USER gcs

EXPOSE 8000

# Healthcheck probing the /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["sh", "-c", "exec python gcs_server.py"]
