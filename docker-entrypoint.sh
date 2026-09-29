#!/bin/bash
set -e

echo "[GCS] Starting Precision Agriculture Ground Control Station on PORT ${PORT:-80}..."

# Seed agronomic data assets into writable DATA_DIR (no-op if already seeded)
echo "[GCS] Seeding state files into ${DATA_DIR:-/tmp/gcs}..."
if python -c "import gcs_server; gcs_server.seed_state()"; then
    echo "[GCS] State seeded successfully."
else
    echo "[GCS Warning] Seed step failed. Proceeding with available data..."
fi

echo "[GCS] Executing command: $@"
exec "$@"
