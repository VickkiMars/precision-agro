# ==============================================================================
# Precision Agriculture GCS - Production Dockerfile
# Base: Python 3.12 Slim Linux Container
# Runtime: stdlib http.server (zero third-party deps for the cloud dashboard)
# Port: 8000 (local/production); dynamic $PORT support
# ==============================================================================

FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATA_DIR=/tmp/gcs

# Create dedicated non-root application user
RUN groupadd -r gcs && useradd -r -g gcs -u 1000 -d /app gcs

WORKDIR /app

# Copy entrypoint script and make executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application source and seeded agronomic data assets
COPY gcs_server.py db_init.py prescription_engine.py \
     crop_health_edge.db camera_config.json \
     field_prescription_log.csv field_prescription_map.geojson \
     plant_classes.json \
     requirements.txt \
     /app/

# Create writable state directory and assign ownership
RUN mkdir -p /tmp/gcs && chown -R gcs:gcs /app /tmp/gcs && chmod -R 1777 /tmp/gcs

USER gcs

EXPOSE 8000

# Zero-dependency healthcheck probe using Python stdlib (no curl install needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz', timeout=3).getcode()==200 else 1)" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["sh", "-c", "exec python gcs_server.py"]
