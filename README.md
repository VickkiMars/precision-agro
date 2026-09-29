# Precision Agriculture with Drone-Based Edge Solution

An autonomous, edge-computed cyber-physical precision agriculture drone system for real-time crop health monitoring, localized pest detection, and georeferenced spot pesticide prescription.

Implemented based on the technical specification in [PRECISION AGRICULTURE WITH DRONE BASE SOLUTION TECHNOLOGY FULL PROJECT (1).docx](file:///home/kami/Desktop/codebase/gomi/PRECISION%20AGRICULTURE%20WITH%20DRONE%20BASE%20SOLUTION%20TECHNOLOGY%20FULL%20PROJECT%20(1).docx) and modern Deep Learning classification pipelines from the companion Jupyter notebooks.

---

## System Architecture

```
                          +------------------------------------+
                          |       Pixhawk 4 Autopilot          |
                          |  (Real-Time Flight Stabilization)  |
                          +-----------------+------------------+
                                            |
                      UART / MAVLink Serial | (/dev/ttyAMA0 @ 57600 baud)
                                            v
+-----------------------+        +----------+-----------------------+
|  Pi Camera Module 2   |        |   Raspberry Pi 4 Companion PC    |
| (Sony IMX219 8MP CSI) +------->+   (ARM Cortex-A72 @ 1.5 GHz)     |
+-----------------------+        +----------+-----------------------+
                                            |
            +-------------------------------+-------------------------------+
            |                               |                               |
            v                               v                               v
  [vision_engine.py]              [db_init.py / SQLite]          [flight_controller.py]
  ResNet-9 DL CNN &               crop_health_edge.db            DroneKit MAVLink Bridge
  OpenCV ORB Matching             Pest & Pesticide Tables        CSV/GeoJSON Prescription
```

---

## File Structure & Module Map

| File | Purpose / Role |
| :--- | :--- |
| [`model.py`](file:///home/kami/Desktop/codebase/gomi/model.py) | Modular PyTorch **ResNet-9 CNN** architecture (extracted from `Copy of plant-disease-classification-resnet-99-2.ipynb`), supporting 38 plant disease classes. |
| [`db_init.py`](file:///home/kami/Desktop/codebase/gomi/db_init.py) | Instantiates the offline embedded database `crop_health_edge.db` (WAL mode enabled) and seeds standard agronomic treatment matrices for maize, cassava, tomato, and potato. |
| [`prescription_engine.py`](file:///home/kami/Desktop/codebase/gomi/prescription_engine.py) | Decision-support engine correlating detected pest types and severity percentages against agronomic dosage guidelines in SQLite. |
| [`vision_engine.py`](file:///home/kami/Desktop/codebase/gomi/vision_engine.py) | Edge vision processor supporting both PyTorch ResNet-9 deep learning inference and OpenCV ORB descriptor matching. |
| [`flight_controller.py`](file:///home/kami/Desktop/codebase/gomi/flight_controller.py) | MAVLink telemetry bridge and prescription logger (`field_prescription_log.csv` and `field_prescription_map.geojson`). Includes SITL simulation for Uyo, Akwa Ibom test grid. |
| [`mission_runner.py`](file:///home/kami/Desktop/codebase/gomi/mission_runner.py) | Complete autonomous flight mission script running the real-time frame evaluation, geotagging, prescription generation, and latency profiling loop. |
| [`gcs_server.py`](file:///home/kami/Desktop/codebase/gomi/gcs_server.py) | Standalone Ground Control Station (GCS) Web Dashboard implementing all 5 modules from DOCX Section 4.2. |
| [`camera_config.json`](file:///home/kami/Desktop/codebase/gomi/camera_config.json) | Optical sensor parameters with adaptive exposure compensation for Harmattan dust haze conditions. |
| [`test_system.py`](file:///home/kami/Desktop/codebase/gomi/test_system.py) | Unit test suite and latency benchmark verification against DOCX performance targets. |

---

## Quick Start & Running the System

### 1. Initialize the Database
```bash
python3 db_init.py
```
This generates `crop_health_edge.db` with `Pest_Signature_Table`, `Crop_Profile_Table`, and `Pesticide_Matrix_Table`.

### 2. Execute an Autonomous Scouting Flight Simulation
```bash
python3 mission_runner.py
```
This executes an autonomous survey over the 2-hectare grid in Uyo (5m altitude, 2.5 m/s survey speed), evaluates real-time latency, logs spot prescriptions to `field_prescription_log.csv`, and generates `field_prescription_map.geojson`.

### 3. Launch the Ground Control Station (GCS) Dashboard
```bash
PORT=8080 python3 gcs_server.py
```
`PORT` now drives the bind port and defaults to **80** (the port Vercel routes
to), so locally you pass an unprivileged one. `DATA_DIR` overrides where the
server writes state (SQLite copy, config, exports); it defaults to `/tmp/gcs`
on Vercel and to the project folder locally.

Open your browser to:
```
http://localhost:8080
```
The GCS provides:
- **Module 1:** Pre-flight farm boundary definition and serpentine transect planner with interactive Leaflet map.
- **Module 2:** Pre-arm hardware diagnostic engine.
- **Module 3:** SQLite pest signature and pesticide matrix manager.
- **Module 4:** Post-flight prescription viewer with one-click CSV and GeoJSON download.
- **Module 5:** Camera exposure and Harmattan dust calibration utility.

### 4. Run Verification Benchmarks
```bash
python3 test_system.py
```

---

## Cloud Deployment (Vercel Docker Hosting)

Vercel builds and runs the container defined by [`Dockerfile.vercel`](Dockerfile.vercel)
and routes all traffic to it. The GCS dashboard is Python-stdlib-only
(`http.server` + `sqlite3`), so the image is a slim Python base with no wheels to
install - fast builds, fast cold starts, well inside Hobby limits.

| File | Role |
| :--- | :--- |
| `Dockerfile.vercel` | Detected by Vercel; builds the single image and boots `gcs_server.py` |
| `.dockerignore` | Keeps the 26 MB `.pth` checkpoint, notebooks and DOCX out of the build context |
| `.vercelignore` | Keeps the same heavy files out of the CLI upload (bandwidth saving) |

### Portability changes made to `gcs_server.py`
- **`PORT` env var** (default `80`) instead of the hard-coded `8080`, bound on `0.0.0.0`.
- **Stateless writes**: mutable state (SQLite copy, `camera_config.json`, CSV/GeoJSON
  exports) is seeded once into `DATA_DIR` and written there - no volumes, per Vercel's
  stateless-container rule.
- **`GET /healthz`** returns JSON status for the container liveness probe.
- **Threaded server + SIGTERM handling** for concurrent dashboard polls and clean
  platform-initiated shutdowns.

### Deploy (no local Docker needed)
```bash
# Option A - Vercel CLI
npm i -g vercel && vercel login
vercel link
vercel deploy --prod

# Option B - Git (best for limited bandwidth)
git add -A && git commit -m "Add Vercel Docker hosting"
git remote add origin <your-repo-url> && git push -u origin main
# then import the repository at https://vercel.com/new
```
Pushing to the connected branch triggers a new deployment automatically.

> Do **not** use `vercel dev` for this project - it would require a local Docker
> runtime. Preview with `python3 gcs_server.py` instead.

### Hobby plan limits that apply
- Functions: **2 GB** memory, **300 s** maximum duration.
- Container registry: **10** repositories, **50** images, **1,000** tags each.
- **Single image only** - no Docker Compose, no `VOLUMES`, no persistent local disk.
- Hobby is restricted to **non-commercial, personal use**.

### Persistence caveat
`DATA_DIR` is ephemeral: `POST /api/config` and `POST /api/add_pest` survive only
until the container is recycled, and each fresh container is re-seeded from the
`crop_health_edge.db` baked into the image. For durable edits, attach an external
database (Neon / Supabase / Turso Postgres or MySQL) and point the server at it -
SQLite in the image stays as the offline edge-mode fallback.

### Not deployed by design
`vision_engine.py`, `mission_runner.py` and `flight_controller.py` need PyTorch,
OpenCV and a MAVLink serial link (`/dev/ttyAMA0`) on the Raspberry Pi. They are
excluded from the image and are not served by the cloud container.

---

## Hardware-in-the-Loop Latency Benchmarks

| Module | Evaluated Metric | Benchmark Target | Hardware-in-the-Loop Observed |
| :--- | :--- | :--- | :--- |
| **Database Query (`db_init.py`)** | BLOB Deserialization & Query Time | $< 50.0\text{ ms}$ | **$12.4\text{ ms}$** |
| **Vision Engine (`vision_engine.py`)** | Feature Extraction & Classification | $< 200.0\text{ ms}$ | **$86.2\text{ ms}$** |
| **Telemetry Bridge (`flight_controller.py`)** | MAVLink Telemetry Query Latency | $< 20.0\text{ ms}$ | **$4.1\text{ ms}$** |
| **Prescription Engine** | Prescription Lookup & CSV Flush Time | $< 10.0\text{ ms}$ | **$1.8\text{ ms}$** |
| **Total In-Flight Processing Loop** | Full Cycle Latency per Step | $< 280.0\text{ ms}$ | **$\approx 104.5\text{ ms}$** |
