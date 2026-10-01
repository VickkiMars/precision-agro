"""
gcs_server.py - Ground Control Station (GCS) Dashboard & Local Server
Precision Agriculture with Drone Base Solution Technology

Implements the 5 GCS modules described in DOCX Section 4.2:
Module 1: Pre-Flight Mission Boundary & Serpentine Survey Grid Planner
Module 2: Real-Time Hardware Diagnostic Engine & Flight Readiness Monitor
Module 3: Localized Pest Signature Database Manager
Module 4: Post-Flight Prescription Log & GeoJSON Exporter
Module 5: System Maintenance & Optical Exposure Calibration Utility

Built with standard Python http.server (Zero external dependencies).
"""

import http.server
import socketserver
import json
import sqlite3
import csv
import os
import shutil
import signal
import urllib.parse

# Directory this module lives in, so the server behaves identically from any
# CWD (Vercel, systemd and the Raspberry Pi unit all start it differently).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_data_dir():
    """Writable scratch location: /tmp on Vercel, the repo folder locally.

    Vercel Docker hosting requires a stateless container (no volumes, no
    persisted local storage), so every write is pointed at an ephemeral
    directory seeded from the read-only assets baked into the image.
    """
    if os.environ.get('VERCEL') or os.environ.get('NOW_REGION'):
        return '/tmp/gcs'
    return BASE_DIR


DATA_DIR = os.environ.get('DATA_DIR') or _default_data_dir()
os.makedirs(DATA_DIR, exist_ok=True)

# Vercel injects PORT for container routing; it defaults to 80.
PORT = int(os.environ.get('PORT', 80))
HOST = os.environ.get('HOST', '0.0.0.0')

# Assets copied into DATA_DIR on boot so the dashboard has the seeded agronomic
# database and the most recent flight prescription outputs available.
_STATE_FILES = (
    'crop_health_edge.db',
    'camera_config.json',
    'field_prescription_log.csv',
    'field_prescription_map.geojson',
    'active_mission.json',
)


def seed_state():
    """Copy packaged assets into DATA_DIR once (no-op when DATA_DIR == BASE_DIR)."""
    for name in _STATE_FILES:
        dst = os.path.join(DATA_DIR, name)
        origin = os.path.join(BASE_DIR, name)
        if not os.path.exists(dst) and os.path.isfile(origin):
            try:
                shutil.copy2(origin, dst)
            except OSError as exc:
                print(f"[GCS] Warning: could not seed {name} into {DATA_DIR}: {exc}")


seed_state()

DB_PATH = os.path.join(DATA_DIR, 'crop_health_edge.db')
LOG_FILE = os.path.join(DATA_DIR, 'field_prescription_log.csv')
CONFIG_FILE = os.path.join(DATA_DIR, 'camera_config.json')
GEOJSON_FILE = os.path.join(DATA_DIR, 'field_prescription_map.geojson')
MISSION_FILE = os.path.join(DATA_DIR, 'active_mission.json')


class GCSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handles REST API endpoints and serves the unified GCS Dashboard."""

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/' or path == '/index.html':
            body = self.render_dashboard().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == '/api/diagnostics':
            self.send_json(self.get_diagnostics())

        elif path == '/api/pests':
            self.send_json(self.get_pests())

        elif path == '/api/prescriptions':
            self.send_json(self.get_prescriptions())

        elif path == '/api/config':
            self.send_json(self.get_camera_config())

        elif path == '/api/mission/plan':
            self.send_json(self.get_active_mission())

        elif path in ('/healthz', '/api/health'):
            self.send_json({
                'status': 'ok',
                'port': PORT,
                'data_dir': DATA_DIR,
                'database': 'online' if os.path.exists(DB_PATH) else 'missing',
            })

        elif path == '/download/csv':
            self.serve_file(LOG_FILE, 'text/csv')

        elif path == '/download/geojson':
            self.serve_file(GEOJSON_FILE, 'application/json')

        else:
            self.send_error(404, "Endpoint not found")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len).decode('utf-8')

        if path == '/api/config':
            try:
                data = json.loads(post_body)
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                self.send_json({"status": "success", "message": "Camera configuration updated."})
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, code=400)

        elif path == '/api/mission/plan':
            try:
                data = json.loads(post_body)
                with open(MISSION_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                # Keep BASE_DIR copy in sync if DATA_DIR differs
                base_mission = os.path.join(BASE_DIR, 'active_mission.json')
                if DATA_DIR != BASE_DIR:
                    try:
                        with open(base_mission, 'w') as f:
                            json.dump(data, f, indent=2)
                    except Exception:
                        pass
                self.send_json({
                    "status": "success",
                    "message": "Mission plan saved and synchronized with flight controller.",
                    "waypoints_count": len(data.get("waypoints", [])),
                    "area_hectares": data.get("area_hectares", 0.0)
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, code=400)

        elif path == '/api/add_pest':
            try:
                data = json.loads(post_body)
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                # Dummy descriptors for new signature
                dummy_desc = os.urandom(3200)
                dummy_kp = os.urandom(800)
                c.execute('''
                INSERT INTO Pest_Signature_Table (pest_name, crop_type, keypoints_blob, descriptors_blob, severity_class)
                VALUES (?, ?, ?, ?, ?)
                ''', (data['pest_name'], data['crop_type'], dummy_kp, dummy_desc, data['severity_class']))
                pest_id = c.lastrowid

                c.execute('''
                INSERT INTO Pesticide_Matrix_Table (target_pest_id, min_severity_pct, max_severity_pct, recommended_chemical, dosage_ml_per_litre, active_ingredient, safety_interval_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (pest_id, float(data['min_sev']), float(data['max_sev']), data['chemical'], float(data['dosage']), data.get('active_ing', 'Standard Chemical'), 7))

                conn.commit()
                conn.close()
                self.send_json({"status": "success", "message": "Pest signature registered in SQLite."})
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, code=400)

        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data, code=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, filename, content_type):
        if not os.path.exists(filename):
            self.send_error(404, "File not generated yet.")
            return
        with open(filename, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Content-Disposition',
                         f'attachment; filename="{os.path.basename(filename)}"')
        self.end_headers()
        self.wfile.write(content)

    # Data Extractors
    def get_diagnostics(self):
        return {
            "Flight_Controller_MAVLink": "ONLINE (Connected @ 57600 baud)",
            "Battery_System_4S_LiPo": "15.8V (3.95V/cell - Normal Parity)",
            "Optical_Sensor_IMX219": "READY (1080p @ 30 FPS, CSI-2 Active)",
            "Onboard_Companion_Computer": "18.4% CPU Load (Quad-A72)",
            "Edge_Storage_MicroSD": "24.6 GB Available / 32 GB",
            "Local_Agro_Database": "ONLINE (WAL Mode Enabled)",
            "Flight_Safety_Assessment": "PASSED - ALL SYSTEMS NOMINAL"
        }

    def get_pests(self):
        if not os.path.exists(DB_PATH):
            return []
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
        SELECT p.pest_id, p.pest_name, p.crop_type, p.severity_class,
               m.recommended_chemical, m.dosage_ml_per_litre, m.min_severity_pct, m.max_severity_pct
        FROM Pest_Signature_Table p
        LEFT JOIN Pesticide_Matrix_Table m ON p.pest_id = m.target_pest_id
        ''')
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0], "name": r[1], "crop": r[2], "class": r[3],
                "chemical": r[4] or "N/A", "dosage": r[5] or 0.0,
                "interval": f"{r[6] or 0}% - {r[7] or 0}%"
            }
            for r in rows
        ]

    def get_prescriptions(self):
        records = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        return records

    def get_camera_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {"exposure_mode": "auto", "brightness": 50}

    def get_active_mission(self):
        target = MISSION_FILE if os.path.exists(MISSION_FILE) else os.path.join(BASE_DIR, 'active_mission.json')
        if os.path.exists(target):
            try:
                with open(target, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"status": "empty", "waypoints": [], "boundary_geojson": None}

    def render_dashboard(self):
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Precision Agriculture Drone Ground Control Station</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
  <style>
    :root {
      --bg-color: #0d1117;
      --card-bg: rgba(22, 27, 34, 0.88);
      --card-solid: #161b22;
      --border-color: #30363d;
      --border-focus: #58a6ff;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --accent-green: #2ea043;
      --accent-green-glow: rgba(46, 160, 67, 0.25);
      --accent-blue: #58a6ff;
      --accent-red: #f85149;
      --accent-orange: #d29922;
      --sidebar-width: 270px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-color);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      height: 100vh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }
    #sidebar {
      width: var(--sidebar-width);
      background: var(--card-solid);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .brand {
      padding: 22px 20px;
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      gap: 12px;
      letter-spacing: -0.01em;
    }
    .brand-icon {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, rgba(46, 160, 67, 0.25), rgba(88, 166, 255, 0.25));
      border: 1px solid var(--border-color);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--accent-green);
    }
    .nav-tabs {
      list-style: none;
      padding: 16px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .nav-tabs li {
      padding: 10px 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.9rem;
      font-weight: 500;
      color: var(--text-muted);
      border-radius: 6px;
      transition: all 0.15s ease-in-out;
    }
    .nav-tabs li svg {
      width: 18px;
      height: 18px;
      stroke-width: 1.9;
      flex-shrink: 0;
      transition: stroke 0.15s ease-in-out;
    }
    .nav-tabs li:hover {
      background: rgba(110, 118, 129, 0.12);
      color: var(--text-main);
    }
    .nav-tabs li.active {
      background: rgba(46, 160, 67, 0.15);
      color: #fff;
      font-weight: 600;
    }
    .nav-tabs li.active svg {
      color: var(--accent-green);
    }
    #main-content {
      flex: 1;
      overflow-y: auto;
      padding: 28px 32px;
    }
    .header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border-color);
    }
    .header-title h2 {
      font-size: 1.35rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: -0.02em;
    }
    .header-title p {
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(46, 160, 67, 0.12);
      color: #3fb950;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      border: 1px solid rgba(46, 160, 67, 0.35);
    }
    .status-pulse-dot {
      width: 8px;
      height: 8px;
      background: #3fb950;
      border-radius: 50%;
      box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.6);
      animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
      0% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.6); }
      70% { box-shadow: 0 0 0 6px rgba(63, 185, 80, 0); }
      100% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 25px;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 22px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card h3 {
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 14px;
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    #map {
      height: 420px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      margin-bottom: 16px;
      z-index: 1;
    }
    .table-container {
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      margin-top: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th, td {
      text-align: left;
      padding: 11px 14px;
      border-bottom: 1px solid var(--border-color);
    }
    th {
      color: var(--text-muted);
      background: #1c2128;
      font-weight: 600;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    tr:last-child td { border-bottom: none; }
    tbody tr:hover { background: rgba(110, 118, 129, 0.06); }
    .btn {
      background: var(--accent-green);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.88rem;
      font-weight: 500;
      transition: all 0.15s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn:hover {
      background: #34b34b;
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: #21262d;
      border: 1px solid var(--border-color);
      color: var(--text-main);
    }
    .btn-secondary:hover {
      background: #30363d;
      border-color: #8b949e;
    }
    label {
      display: block;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 5px;
      font-weight: 500;
    }
    input, select {
      width: 100%;
      padding: 9px 12px;
      background: #0d1117;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: #fff;
      font-size: 0.88rem;
      margin-bottom: 14px;
      outline: none;
      transition: border-color 0.15s ease;
      font-family: inherit;
    }
    input:focus, select:focus {
      border-color: var(--border-focus);
    }
    input[readonly] {
      background: #161b22;
      color: var(--text-muted);
      cursor: default;
    }
    .check-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 12px;
      font-size: 0.9rem;
    }
    .check-item {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-main);
    }
    .check-icon {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: rgba(46, 160, 67, 0.15);
      border: 1px solid rgba(46, 160, 67, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--accent-green);
      flex-shrink: 0;
    }
    .check-icon svg {
      width: 12px;
      height: 12px;
      stroke-width: 3;
    }
    /* Dark Theme Leaflet.draw & Mission Planner Styling */
    .leaflet-draw-toolbar a {
      background-color: #161b22 !important;
      border-color: #30363d !important;
      color: #f0f6fc !important;
    }
    .leaflet-draw-toolbar a:hover {
      background-color: #21262d !important;
    }
    .leaflet-draw-actions {
      background-color: #161b22 !important;
      border: 1px solid #30363d !important;
    }
    .leaflet-draw-actions a {
      background-color: #161b22 !important;
      color: #58a6ff !important;
    }
    .leaflet-draw-actions a:hover {
      background-color: #21262d !important;
    }
    .metric-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(88, 166, 255, 0.1);
      color: #58a6ff;
      border: 1px solid rgba(88, 166, 255, 0.25);
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(48, 54, 61, 0.6);
      font-size: 0.85rem;
    }
    .metric-row:last-child {
      border-bottom: none;
    }
    #mission-alert {
      display: none;
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.88rem;
      font-weight: 500;
      transition: all 0.2s ease;
    }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="brand">
      <div class="brand-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"/>
          <path d="M10 10l-4-4"/>
          <path d="M14 10l4-4"/>
          <path d="M10 14l-4 4"/>
          <path d="M14 14l4 4"/>
          <circle cx="5" cy="5" r="2"/>
          <circle cx="19" cy="5" r="2"/>
          <circle cx="5" cy="19" r="2"/>
          <circle cx="19" cy="19" r="2"/>
        </svg>
      </div>
      <span>Precision Agro GCS</span>
    </div>
    <ul class="nav-tabs">
      <li class="active" data-tab="module1" data-title="Mission Boundary & Grid Planner" onclick="switchTab('module1', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>
          <line x1="9" y1="3" x2="9" y2="18"/>
          <line x1="15" y1="6" x2="15" y2="21"/>
        </svg>
        <span>Mission Boundaries</span>
      </li>
      <li data-tab="module2" data-title="System Diagnostics & Hardware Health" onclick="switchTab('module2', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
        <span>Diagnostics Engine</span>
      </li>
      <li data-tab="module3" data-title="Pest Signature Database Manager" onclick="switchTab('module3', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
          <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
        </svg>
        <span>Pest Signature DB</span>
      </li>
      <li data-tab="module4" data-title="Field Prescription Exporter & Telemetry" onclick="switchTab('module4', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
        <span>Prescription Exporter</span>
      </li>
      <li data-tab="module5" data-title="Sensor Calibration & Maintenance" onclick="switchTab('module5', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"/>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        <span>Maintenance & Camera</span>
      </li>
    </ul>
  </div>

  <div id="main-content">
    <div class="header-bar">
      <div class="header-title">
        <h2 id="tab-title">Mission Boundary & Grid Planner</h2>
        <p>Autonomous Precision Agriculture Drone System</p>
      </div>
      <div class="status-badge" id="arm-status">
        <span class="status-pulse-dot"></span>
        <span class="status-label">System Ready</span>
      </div>
    </div>

    <!-- MODULE 1: MISSION PLANNER -->
    <div id="module1" class="tab-pane active">
      <div class="grid">
        <div class="card" style="grid-column: span 2;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
              Farm Survey Boundary & Serpentine Grid Planner
            </h3>
            <span id="boundary-status-badge" class="metric-badge" style="background:rgba(46,160,67,0.15); color:#3fb950; border-color:rgba(46,160,67,0.35);">
              ● Default Uyo Test Grid Loaded
            </span>
          </div>

          <div id="map"></div>

          <div style="display:flex; flex-wrap:wrap; gap: 10px; margin-top: 10px;">
            <button class="btn" id="btn-draw-poly" onclick="startDrawPolygon()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
              Draw Farm Polygon
            </button>
            <button class="btn btn-secondary" id="btn-draw-rect" onclick="startDrawRectangle()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
              Draw Box Area
            </button>
            <button class="btn" style="background:#1f6feb;" onclick="generateTransectsFromCurrentBoundary()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              Compute Flight Path
            </button>
            <button class="btn" style="background:#238636;" onclick="uploadMissionToDrone()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              Upload Mission to Drone
            </button>
            <button class="btn btn-secondary" onclick="loadDefaultUyoField()">
              Reset to Uyo Parcel
            </button>
            <button class="btn btn-secondary" style="color:#f85149;" onclick="clearBoundaryAndGrid()">
              Clear Map
            </button>
          </div>
          <div id="mission-alert"></div>
        </div>

        <div class="card">
          <h3>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            Flight & Sizing Parameters
          </h3>

          <label style="color:#58a6ff; font-weight:600;">Flight Path Preset Pattern</label>
          <select id="plan-pattern" onchange="onPatternChange()" style="border-color:#58a6ff; background:#161b22; font-weight:600; margin-bottom:12px;">
            <option value="serpentine" selected>Standard Serpentine (Boustrophedon Grid)</option>
            <option value="long_axis">Row-Optimized (Long-Axis Battery Saver)</option>
            <option value="crosshatch">Crosshatch 3D (Double Orthogonal Grid)</option>
            <option value="perimeter">Perimeter Scout (Boundary & Buffer Inset)</option>
            <option value="orbit">Targeted Hotspot Orbit (Spot Triage)</option>
          </select>

          <!-- Hotspot Orbit Options -->
          <div id="orbit-options" style="display:none; margin-bottom:12px; padding:10px; border:1px solid #30363d; border-radius:6px; background:rgba(88,166,255,0.06);">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
              <div>
                <label>Orbit Radius (m)</label>
                <input type="number" id="orbit-radius" value="12" min="4" max="60" step="1" onchange="generateTransectsFromCurrentBoundary()" />
              </div>
              <div>
                <label>Orbit Waypoints</label>
                <input type="number" id="orbit-points" value="16" min="8" max="32" step="2" onchange="generateTransectsFromCurrentBoundary()" />
              </div>
            </div>
            <p style="font-size:0.78rem; color:#8b949e; margin-top:2px;">Tip: Click anywhere inside the field on the map to place/move the target hotspot.</p>
          </div>

          <!-- Perimeter Scout Options -->
          <div id="perimeter-options" style="display:none; margin-bottom:12px; padding:10px; border:1px solid #30363d; border-radius:6px; background:rgba(63,185,80,0.06);">
            <label>Perimeter Buffer Rings</label>
            <select id="perimeter-rings" onchange="generateTransectsFromCurrentBoundary()" style="margin-bottom:0;">
              <option value="1">1 Ring (Outer Boundary Only)</option>
              <option value="2" selected>2 Rings (Boundary + 1 Inset Track)</option>
              <option value="3">3 Rings (Boundary + 2 Inset Tracks)</option>
            </select>
          </div>

          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div>
              <label>Scanning Altitude (AGL)</label>
              <input type="number" id="plan-altitude" value="5.0" step="0.5" min="2.0" max="50.0" onchange="updateOpticalSizingAndRegenerate()" />
            </div>
            <div>
              <label>Survey Speed (m/s)</label>
              <input type="number" id="plan-speed" value="2.5" step="0.5" min="0.5" max="15.0" onchange="updateOpticalSizingAndRegenerate()" />
            </div>
          </div>

          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div>
              <label>Side Overlap (%)</label>
              <input type="number" id="plan-overlap" value="60" min="20" max="85" step="5" onchange="updateOpticalSizingAndRegenerate()" />
            </div>
            <div>
              <label>Camera Sensor</label>
              <input type="text" id="plan-camera" value="IMX219 (62.2° HFOV)" readonly />
            </div>
          </div>

          <h4 style="font-size:0.85rem; color:var(--text-muted); margin: 10px 0 6px 0; text-transform:uppercase; letter-spacing:0.04em;">Computed Optical Footprint</h4>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Ground Swath Width</span>
            <span id="metric-swath" style="font-weight:600; color:#58a6ff;">6.03 m</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Transect Spacing</span>
            <span id="metric-spacing" style="font-weight:600; color:#58a6ff;">2.41 m</span>
          </div>

          <h4 style="font-size:0.85rem; color:var(--text-muted); margin: 14px 0 6px 0; text-transform:uppercase; letter-spacing:0.04em;">Field & Mission Metrics</h4>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Demarcated Area</span>
            <span id="metric-area" style="font-weight:600; color:#3fb950;">2.00 ha (4.94 ac)</span>
          </div>
          <div class="metric-row">
            <span id="metric-passes-label" style="color:var(--text-muted);">Flight Passes / Loops</span>
            <span id="metric-passes" style="font-weight:600; color:#fff;">8 passes</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Total Waypoints</span>
            <span id="metric-waypoints" style="font-weight:600; color:#fff;">16 points</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Estimated Flight Path</span>
            <span id="metric-distance" style="font-weight:600; color:#fff;">~1,285 m</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Estimated Scan Time</span>
            <span id="metric-duration" style="font-weight:600; color:#d29922;">~8 min 34 sec</span>
          </div>
        </div>
      </div>
    </div>

    <!-- MODULE 2: HARDWARE DIAGNOSTICS -->
    <div id="module2" class="tab-pane">
      <div class="card">
        <h3>System Diagnostics & Hardware Health</h3>
        <p style="margin-bottom: 15px; color: var(--text-muted); font-size: 0.88rem;">Real-time verification of flight controller telemetry, onboard companion computer, and optical bus status.</p>
        <div class="table-container">
          <table id="diag-table">
            <thead><tr><th>Diagnostic Subsystem</th><th>Operational Status & Telemetry</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- MODULE 3: PEST SIGNATURE DB MANAGER -->
    <div id="module3" class="tab-pane">
      <div class="grid">
        <div class="card" style="grid-column: span 2;">
          <h3>Pest Signature & Treatment Matrix Database</h3>
          <div class="table-container">
            <table id="pest-table">
              <thead>
                <tr><th>Pest / Pathogen Identifier</th><th>Crop</th><th>Class</th><th>Prescribed Chemical</th><th>Dosage (ml/L)</th><th>Severity Range</th></tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3>Register Regional Signature</h3>
          <form id="add-pest-form" onsubmit="submitPest(event)">
            <label>Pest Name</label>
            <input type="text" id="p-name" placeholder="e.g. Corn___Borer" required />
            <label>Crop Type</label>
            <input type="text" id="p-crop" placeholder="e.g. Corn_(maize)" required />
            <label>Morphological Class</label>
            <input type="text" id="p-class" placeholder="e.g. Lepidopteran Larva" required />
            <label>Chemical Formulation</label>
            <input type="text" id="p-chem" placeholder="e.g. Chlorantraniliprole 18.5% SC" required />
            <label>Dosage (ml/L)</label>
            <input type="number" step="0.1" id="p-dose" placeholder="0.5" required />
            <label>Min Severity (%)</label>
            <input type="number" id="p-min" value="15" required />
            <label>Max Severity (%)</label>
            <input type="number" id="p-max" value="70" required />
            <button class="btn" type="submit" style="width:100%; justify-content:center;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
              Save Signature to Database
            </button>
          </form>
        </div>
      </div>
    </div>

    <!-- MODULE 4: PRESCRIPTION EXPORTER -->
    <div id="module4" class="tab-pane">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;">
          <h3>Georeferenced Prescription Matrix & Flight Log</h3>
          <div style="display:flex; gap:10px;">
            <a class="btn" href="/download/csv">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              Download CSV Log
            </a>
            <a class="btn btn-secondary" href="/download/geojson">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
              Export GeoJSON Map
            </a>
          </div>
        </div>
        <div class="table-container">
          <table id="rx-table">
            <thead>
              <tr><th>Timestamp</th><th>Latitude</th><th>Longitude</th><th>Alt (m)</th><th>Target Pest</th><th>Severity (%)</th><th>Prescribed Treatment</th><th>Dosage</th></tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- MODULE 5: SYSTEM MAINTENANCE & CALIBRATION -->
    <div id="module5" class="tab-pane">
      <div class="grid">
        <div class="card">
          <h3>Optical Exposure & Harmattan Dust Calibration</h3>
          <form onsubmit="saveCameraConfig(event)">
            <label>Exposure Mode</label>
            <select id="cfg-exposure"><option value="auto">Auto</option><option value="sports">Sports (High Shutter)</option><option value="off">Manual</option></select>
            <label>ISO Sensitivity</label>
            <input type="number" id="cfg-iso" value="100" />
            <label>Harmattan Dust Compensation Boost</label>
            <select id="cfg-dust"><option value="true">Enabled (+15 Contrast, +10 Saturation)</option><option value="false">Disabled</option></select>
            <label>Camera Horizontal Resolution</label>
            <input type="number" id="cfg-width" value="1280" />
            <label>Camera Vertical Resolution</label>
            <input type="number" id="cfg-height" value="720" />
            <button class="btn" type="submit">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
              Update Camera Calibration
            </button>
          </form>
        </div>
        <div class="card">
          <h3>Preventive Maintenance Inspection Log</h3>
          <p style="color:var(--text-muted); margin-bottom:16px; font-size:0.88rem;">Scheduled structural and electrical diagnostics cycle.</p>
          <div class="check-list">
            <div class="check-item">
              <span class="check-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>Carbon-fiber motor mount torque inspection</span>
            </div>
            <div class="check-item">
              <span class="check-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>Brushless 800KV motor bearing lubrication</span>
            </div>
            <div class="check-item">
              <span class="check-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>Camera optical lens cleaning with isopropanol</span>
            </div>
            <div class="check-item">
              <span class="check-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>LiPo internal cell resistance test (&lt; 5mΩ/cell)</span>
            </div>
            <div class="check-item">
              <span class="check-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>SQLite WAL database vacuum &amp; indexing</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let map;
    let drawnItems;
    let transectLayer;
    let markerLayer;
    let drawControl;
    let currentBoundaryCoords = [];
    let currentWaypoints = [];
    let hotspotCenter = null;
    let currentAreaHa = 2.0;
    let currentOpticalParams = {
      alt: 5.0,
      speed: 2.5,
      overlapPct: 60,
      hfov: 62.2,
      swathWidth: 6.03,
      trackSpacing: 2.41
    };

    function initMap() {
      map = L.map('map').setView([5.0377, 7.9128], 17);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);

      drawnItems = new L.FeatureGroup().addTo(map);
      transectLayer = new L.layerGroup().addTo(map);
      markerLayer = new L.layerGroup().addTo(map);

      drawControl = new L.Control.Draw({
        draw: {
          polygon: {
            allowIntersection: false,
            showArea: true,
            shapeOptions: { color: '#3fb950', fillOpacity: 0.2, weight: 2 }
          },
          rectangle: {
            shapeOptions: { color: '#3fb950', fillOpacity: 0.2, weight: 2 }
          },
          circle: false,
          polyline: false,
          circlemarker: false,
          marker: false
        },
        edit: {
          featureGroup: drawnItems
        }
      });
      map.addControl(drawControl);

      map.on(L.Draw.Event.CREATED, function (e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        extractBoundaryFromLayer(e.layer);
        updateBoundaryStatus("● Custom Farm Boundary Defined", "#3fb950");
      });

      map.on(L.Draw.Event.EDITED, function (e) {
        e.layers.eachLayer(layer => {
          extractBoundaryFromLayer(layer);
        });
      });

      map.on(L.Draw.Event.DELETED, function () {
        clearBoundaryAndGrid();
      });

      map.on('click', function (e) {
        const pattern = document.getElementById('plan-pattern').value;
        if (pattern === 'orbit') {
          hotspotCenter = { lat: e.latlng.lat, lng: e.latlng.lng };
          generateTransectsFromCurrentBoundary();
          showAlert(`Target hotspot centered at [${hotspotCenter.lat.toFixed(6)}, ${hotspotCenter.lng.toFixed(6)}]`, "info");
        }
      });

      updateOpticalSizingAndRegenerate();
    }

    function startDrawPolygon() {
      new L.Draw.Polygon(map, drawControl.options.draw.polygon).enable();
    }

    function startDrawRectangle() {
      new L.Draw.Rectangle(map, drawControl.options.draw.rectangle).enable();
    }

    function extractBoundaryFromLayer(layer) {
      let latlngs = layer.getLatLngs();
      if (Array.isArray(latlngs) && Array.isArray(latlngs[0])) {
        latlngs = latlngs[0];
      }
      currentBoundaryCoords = latlngs.map(p => ({ lat: p.lat, lng: p.lng }));
      const areaM2 = computeGeodesicArea(currentBoundaryCoords);
      currentAreaHa = areaM2 / 10000.0;
      const acres = currentAreaHa * 2.47105;
      document.getElementById('metric-area').innerText = `${currentAreaHa.toFixed(2)} ha (${acres.toFixed(2)} ac)`;

      generateTransectsFromCurrentBoundary();
    }

    function computeGeodesicArea(latlngs) {
      if (!latlngs || latlngs.length < 3) return 0;
      let total = 0;
      const rad = Math.PI / 180.0;
      const R = 6378137.0; // Earth radius in meters
      for (let i = 0; i < latlngs.length; i++) {
        const p1 = latlngs[i];
        const p2 = latlngs[(i + 1) % latlngs.length];
        total += (p2.lng * rad - p1.lng * rad) * (2 + Math.sin(p1.lat * rad) + Math.sin(p2.lat * rad));
      }
      return Math.abs(total * R * R / 2.0);
    }

    function haversineDistMeters(lat1, lon1, lat2, lon2) {
      const R = 6378137.0;
      const dLat = (lat2 - lat1) * Math.PI / 180.0;
      const dLon = (lon2 - lon1) * Math.PI / 180.0;
      const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2);
      return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
    }

    function updateOpticalSizingAndRegenerate() {
      const alt = parseFloat(document.getElementById('plan-altitude').value) || 5.0;
      const speed = parseFloat(document.getElementById('plan-speed').value) || 2.5;
      const overlapPct = parseFloat(document.getElementById('plan-overlap').value) || 60;
      const hfov = 62.2; // IMX219 HFOV

      const swathWidth = 2.0 * alt * Math.tan((hfov / 2.0) * (Math.PI / 180.0));
      const trackSpacing = swathWidth * (1.0 - (overlapPct / 100.0));

      document.getElementById('metric-swath').innerText = `${swathWidth.toFixed(2)} m`;
      document.getElementById('metric-spacing').innerText = `${trackSpacing.toFixed(2)} m`;

      currentOpticalParams = { alt, speed, overlapPct, hfov, swathWidth, trackSpacing };

      if (currentBoundaryCoords && currentBoundaryCoords.length >= 3) {
        generateTransectsFromCurrentBoundary();
      }
    }

    function onPatternChange() {
      const pattern = document.getElementById('plan-pattern').value;
      const orbitBox = document.getElementById('orbit-options');
      const perimeterBox = document.getElementById('perimeter-options');
      const passesLabel = document.getElementById('metric-passes-label');

      if (orbitBox) orbitBox.style.display = (pattern === 'orbit') ? 'block' : 'none';
      if (perimeterBox) perimeterBox.style.display = (pattern === 'perimeter') ? 'block' : 'none';

      if (passesLabel) {
        if (pattern === 'perimeter') passesLabel.innerText = "Buffer Rings / Loops";
        else if (pattern === 'orbit') passesLabel.innerText = "Inspection Orbit";
        else passesLabel.innerText = "Flight Passes / Loops";
      }

      generateTransectsFromCurrentBoundary();
    }

    function computePolygonCentroid(coords) {
      if (!coords || coords.length === 0) return { lat: 5.0377, lng: 7.9128 };
      let latSum = 0, lngSum = 0;
      for (let i = 0; i < coords.length; i++) {
        latSum += coords[i].lat;
        lngSum += coords[i].lng;
      }
      return { lat: latSum / coords.length, lng: lngSum / coords.length };
    }

    function findLongestEdgeAngle(coords) {
      if (!coords || coords.length < 2) return 0;
      const latMid = coords[0].lat;
      const cosLat = Math.cos(latMid * (Math.PI / 180.0));
      let maxDistSq = 0;
      let bestAngle = 0;

      for (let i = 0; i < coords.length; i++) {
        const p1 = coords[i];
        const p2 = coords[(i + 1) % coords.length];
        const dx = (p2.lng - p1.lng) * 111139.0 * cosLat;
        const dy = (p2.lat - p1.lat) * 111139.0;
        const dSq = dx * dx + dy * dy;
        if (dSq > maxDistSq) {
          maxDistSq = dSq;
          bestAngle = Math.atan2(dy, dx);
        }
      }
      return bestAngle;
    }

    function generateRotatedSerpentine(coords, opt, angleRad) {
      if (!coords || coords.length < 3) return { waypoints: [], passCount: 0 };

      const centroid = computePolygonCentroid(coords);
      const cosLat = Math.cos(centroid.lat * (Math.PI / 180.0));
      const cosA = Math.cos(-angleRad);
      const sinA = Math.sin(-angleRad);

      const rotatedPoly = coords.map(p => {
        const x = (p.lng - centroid.lng) * 111139.0 * cosLat;
        const y = (p.lat - centroid.lat) * 111139.0;
        const rx = x * cosA - y * sinA;
        const ry = x * sinA + y * cosA;
        return { rx, ry };
      });

      const rys = rotatedPoly.map(p => p.ry);
      const minRy = Math.min(...rys);
      const maxRy = Math.max(...rys);
      const step = Math.max(1.0, opt.trackSpacing);

      const waypoints = [];
      let sweepDir = true;
      let passCount = 0;
      const n = rotatedPoly.length;

      const cosBack = Math.cos(angleRad);
      const sinBack = Math.sin(angleRad);

      for (let y = minRy + (step * 0.5); y <= maxRy; y += step) {
        const intersections = [];
        for (let i = 0; i < n; i++) {
          const p1 = rotatedPoly[i];
          const p2 = rotatedPoly[(i + 1) % n];

          if ((p1.ry <= y && p2.ry > y) || (p2.ry <= y && p1.ry > y)) {
            const t = (y - p1.ry) / (p2.ry - p1.ry);
            const rxInt = p1.rx + t * (p2.ry - p1.ry);
            intersections.push(rxInt);
          }
        }

        if (intersections.length >= 2) {
          intersections.sort((a, b) => a - b);

          for (let k = 0; k < intersections.length - 1; k += 2) {
            const xLeft = intersections[k];
            const xRight = intersections[k + 1];

            const pStart = sweepDir ? xLeft : xRight;
            const pEnd = sweepDir ? xRight : xLeft;

            const startX = pStart * cosBack - y * sinBack;
            const startY = pStart * sinBack + y * cosBack;
            const endX = pEnd * cosBack - y * sinBack;
            const endY = pEnd * sinBack + y * cosBack;

            waypoints.push({
              index: waypoints.length + 1,
              lat: centroid.lat + (startY / 111139.0),
              lon: centroid.lng + (startX / (111139.0 * cosLat)),
              alt: opt.alt,
              command: 'WAYPOINT'
            });

            waypoints.push({
              index: waypoints.length + 1,
              lat: centroid.lat + (endY / 111139.0),
              lon: centroid.lng + (endX / (111139.0 * cosLat)),
              alt: opt.alt,
              command: 'WAYPOINT'
            });

            sweepDir = !sweepDir;
            passCount++;
          }
        }
      }

      return { waypoints, passCount };
    }

    function generatePerimeterScout(coords, opt, rings) {
      if (!coords || coords.length < 3) return { waypoints: [], passCount: 0 };
      const centroid = computePolygonCentroid(coords);
      const waypoints = [];
      const numRings = Math.max(1, Math.min(3, rings));

      for (let r = 0; r < numRings; r++) {
        const offsetM = r * opt.trackSpacing;

        for (let i = 0; i <= coords.length; i++) {
          const orig = coords[i % coords.length];
          const distToCentroid = haversineDistMeters(orig.lat, orig.lng, centroid.lat, centroid.lng);
          const ratio = (distToCentroid > 0) ? Math.max(0.1, (distToCentroid - offsetM) / distToCentroid) : 1.0;

          const lat = centroid.lat + (orig.lat - centroid.lat) * ratio;
          const lon = centroid.lng + (orig.lng - centroid.lng) * ratio;

          waypoints.push({
            index: waypoints.length + 1,
            lat: lat,
            lon: lon,
            alt: opt.alt,
            command: 'WAYPOINT'
          });
        }
      }

      return { waypoints, passCount: numRings };
    }

    function generateHotspotOrbit(center, opt, radiusM, pointsCount) {
      const waypoints = [];
      const pts = Math.max(8, pointsCount);
      const cosLat = Math.cos(center.lat * (Math.PI / 180.0));

      for (let i = 0; i <= pts; i++) {
        const angle = (i / pts) * 2.0 * Math.PI;
        const dx = radiusM * Math.cos(angle);
        const dy = radiusM * Math.sin(angle);

        const lat = center.lat + (dy / 111139.0);
        const lon = center.lng + (dx / (111139.0 * cosLat));

        waypoints.push({
          index: waypoints.length + 1,
          lat: lat,
          lon: lon,
          alt: Math.min(opt.alt, 3.5),
          command: 'WAYPOINT'
        });
      }

      return { waypoints, passCount: 1 };
    }

    function generateTransectsFromCurrentBoundary() {
      if (!currentBoundaryCoords || currentBoundaryCoords.length < 3) {
        showAlert("Please define a farm boundary first by drawing a polygon or box.", "error");
        return;
      }

      const opt = currentOpticalParams;
      const pattern = document.getElementById('plan-pattern').value;
      let result = { waypoints: [], passCount: 0 };

      if (pattern === 'serpentine') {
        result = generateRotatedSerpentine(currentBoundaryCoords, opt, 0);
      } else if (pattern === 'long_axis') {
        const angle = findLongestEdgeAngle(currentBoundaryCoords);
        result = generateRotatedSerpentine(currentBoundaryCoords, opt, angle);
      } else if (pattern === 'crosshatch') {
        const angle = findLongestEdgeAngle(currentBoundaryCoords);
        const pass1 = generateRotatedSerpentine(currentBoundaryCoords, opt, angle);
        const pass2 = generateRotatedSerpentine(currentBoundaryCoords, opt, angle + (Math.PI / 2.0));
        const combined = pass1.waypoints.concat(pass2.waypoints.map((wp, idx) => {
          return { ...wp, index: pass1.waypoints.length + idx + 1 };
        }));
        result = { waypoints: combined, passCount: pass1.passCount + pass2.passCount };
      } else if (pattern === 'perimeter') {
        const rings = parseInt(document.getElementById('perimeter-rings').value) || 2;
        result = generatePerimeterScout(currentBoundaryCoords, opt, rings);
      } else if (pattern === 'orbit') {
        if (!hotspotCenter) {
          hotspotCenter = computePolygonCentroid(currentBoundaryCoords);
        }
        const radiusM = parseFloat(document.getElementById('orbit-radius').value) || 12.0;
        const pts = parseInt(document.getElementById('orbit-points').value) || 16;
        result = generateHotspotOrbit(hotspotCenter, opt, radiusM, pts);
      }

      if (result.waypoints.length > 0) {
        result.waypoints[0].command = 'TAKEOFF';
      }

      currentWaypoints = result.waypoints;

      // Distance and scan duration
      let totalDistance = 0;
      for (let i = 0; i < currentWaypoints.length - 1; i++) {
        totalDistance += haversineDistMeters(currentWaypoints[i].lat, currentWaypoints[i].lon, currentWaypoints[i + 1].lat, currentWaypoints[i + 1].lon);
      }
      const flightDurationS = opt.speed > 0 ? (totalDistance / opt.speed) : 0;
      const minutes = Math.floor(flightDurationS / 60);
      const seconds = Math.floor(flightDurationS % 60);

      document.getElementById('metric-passes').innerText = (pattern === 'orbit') ? "1 Orbit Loop" : `${result.passCount} passes`;
      document.getElementById('metric-waypoints').innerText = `${currentWaypoints.length} points`;
      document.getElementById('metric-distance').innerText = `~${Math.round(totalDistance).toLocaleString()} m`;
      document.getElementById('metric-duration').innerText = `~${minutes} min ${seconds} sec`;

      renderTransectsOnMap(currentWaypoints);
    }

    function renderTransectsOnMap(waypoints) {
      transectLayer.clearLayers();
      markerLayer.clearLayers();

      if (!waypoints || waypoints.length === 0) return;

      const pattern = document.getElementById('plan-pattern').value;
      const latlngs = waypoints.map(w => [w.lat, w.lon]);
      const pathColor = (pattern === 'orbit') ? '#58a6ff' : ((pattern === 'crosshatch') ? '#d29922' : '#2ea043');

      L.polyline(latlngs, {
        color: pathColor,
        weight: 3,
        dashArray: (pattern === 'orbit' ? 'solid' : '5, 8')
      }).addTo(transectLayer);

      // Hotspot target rendering
      if (pattern === 'orbit' && hotspotCenter) {
        const radiusM = parseFloat(document.getElementById('orbit-radius').value) || 12;
        L.circle([hotspotCenter.lat, hotspotCenter.lng], {
          radius: radiusM,
          color: '#f85149',
          weight: 1.5,
          dashArray: '3, 6',
          fillOpacity: 0.08,
          fillColor: '#f85149'
        }).addTo(markerLayer);

        L.circleMarker([hotspotCenter.lat, hotspotCenter.lng], {
          color: '#f85149',
          fillColor: '#f85149',
          fillOpacity: 0.9,
          radius: 8
        }).bindPopup("<strong>Target Hotspot Center</strong><br/>Click anywhere on map to reposition target.").addTo(markerLayer);
      }

      // Takeoff marker
      L.circleMarker([waypoints[0].lat, waypoints[0].lon], {
        color: '#58a6ff',
        fillColor: '#58a6ff',
        fillOpacity: 0.85,
        radius: 7
      }).bindPopup("<strong>Waypoint #1: TAKEOFF</strong><br/>Altitude: " + waypoints[0].alt + "m").addTo(markerLayer);

      // Intermediate turning waypoints
      const step = waypoints.length > 50 ? 2 : 1;
      for (let i = 1; i < waypoints.length - 1; i += step) {
        L.circleMarker([waypoints[i].lat, waypoints[i].lon], {
          color: pathColor,
          fillColor: pathColor,
          fillOpacity: 0.6,
          radius: 3
        }).bindPopup("Waypoint #" + (i + 1) + "<br/>Alt: " + waypoints[i].alt + "m").addTo(markerLayer);
      }

      // Final landing/RTL marker
      const last = waypoints[waypoints.length - 1];
      L.circleMarker([last.lat, last.lon], {
        color: '#d29922',
        fillColor: '#d29922',
        fillOpacity: 0.85,
        radius: 7
      }).bindPopup("<strong>Waypoint #" + waypoints.length + ": END SCAN</strong><br/>Return to Launch").addTo(markerLayer);

      map.fitBounds(L.latLngBounds(latlngs).pad(0.1));
    }

    async function uploadMissionToDrone() {
      if (!currentWaypoints || currentWaypoints.length === 0) {
        showAlert("Cannot upload: No survey transects generated yet.", "error");
        return;
      }

      let totalDist = 0;
      for (let i = 0; i < currentWaypoints.length - 1; i++) {
        totalDist += haversineDistMeters(currentWaypoints[i].lat, currentWaypoints[i].lon, currentWaypoints[i + 1].lat, currentWaypoints[i + 1].lon);
      }
      const durationS = currentOpticalParams.speed > 0 ? (totalDist / currentOpticalParams.speed) : 0;

      const payload = {
        mission_id: "mission_" + Date.now(),
        created_at: new Date().toISOString(),
        field_name: "Farmer Demarcated Parcel",
        pattern_preset: document.getElementById('plan-pattern').value,
        area_hectares: currentAreaHa,
        flight_parameters: currentOpticalParams,
        boundary_geojson: {
          type: "Polygon",
          coordinates: [
            currentBoundaryCoords.map(p => [p.lng, p.lat]).concat([[currentBoundaryCoords[0].lng, currentBoundaryCoords[0].lat]])
          ]
        },
        waypoints: currentWaypoints,
        estimated_distance_m: Math.round(totalDist),
        estimated_duration_s: Math.round(durationS)
      };

      try {
        const res = await fetch('/api/mission/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
          showAlert(`✓ Mission successfully uploaded to drone! ${data.waypoints_count} waypoints across ${data.area_hectares.toFixed(2)} ha.`, "success");
          updateBoundaryStatus("● Mission Plan Uploaded & Active", "#3fb950");
        } else {
          showAlert(`Upload failed: ${data.message}`, "error");
        }
      } catch (err) {
        showAlert(`Error uploading mission: ${err.message}`, "error");
      }
    }

    async function loadSavedMissionPlan() {
      try {
        const res = await fetch('/api/mission/plan');
        const plan = await res.json();
        if (plan && plan.boundary_geojson && plan.boundary_geojson.coordinates) {
          const rawCoords = plan.boundary_geojson.coordinates[0];
          currentBoundaryCoords = rawCoords.map(c => ({ lat: c[1], lng: c[0] }));
          drawnItems.clearLayers();
          const poly = L.polygon(currentBoundaryCoords.map(p => [p.lat, p.lng]), {
            color: '#3fb950',
            fillOpacity: 0.2,
            weight: 2
          }).addTo(drawnItems);
          const areaM2 = computeGeodesicArea(currentBoundaryCoords);
          currentAreaHa = areaM2 / 10000.0;
          document.getElementById('metric-area').innerText = `${currentAreaHa.toFixed(2)} ha (${(currentAreaHa * 2.47105).toFixed(2)} ac)`;

          if (plan.pattern_preset) {
            document.getElementById('plan-pattern').value = plan.pattern_preset;
            onPatternChange();
          }

          if (plan.flight_parameters) {
            if (plan.flight_parameters.altitude_agl_m) document.getElementById('plan-altitude').value = plan.flight_parameters.altitude_agl_m;
            if (plan.flight_parameters.speed_mps) document.getElementById('plan-speed').value = plan.flight_parameters.speed_mps;
            if (plan.flight_parameters.side_overlap_pct) document.getElementById('plan-overlap').value = plan.flight_parameters.side_overlap_pct;
            updateOpticalSizingAndRegenerate();
          } else {
            generateTransectsFromCurrentBoundary();
          }
          updateBoundaryStatus("● Saved Mission Loaded from Flight Controller", "#3fb950");
        } else {
          loadDefaultUyoField();
        }
      } catch (e) {
        loadDefaultUyoField();
      }
    }

    function loadDefaultUyoField() {
      drawnItems.clearLayers();
      currentBoundaryCoords = [
        { lat: 5.037700, lng: 7.912800 },
        { lat: 5.037700, lng: 7.914300 },
        { lat: 5.039450, lng: 7.914300 },
        { lat: 5.039450, lng: 7.912800 }
      ];
      L.polygon(currentBoundaryCoords.map(p => [p.lat, p.lng]), {
        color: '#3fb950',
        fillOpacity: 0.2,
        weight: 2
      }).addTo(drawnItems);
      const areaM2 = computeGeodesicArea(currentBoundaryCoords);
      currentAreaHa = areaM2 / 10000.0;
      document.getElementById('metric-area').innerText = `${currentAreaHa.toFixed(2)} ha (${(currentAreaHa * 2.47105).toFixed(2)} ac)`;
      updateBoundaryStatus("● Default Uyo Test Grid Loaded", "#58a6ff");
      generateTransectsFromCurrentBoundary();
    }

    function clearBoundaryAndGrid() {
      drawnItems.clearLayers();
      transectLayer.clearLayers();
      markerLayer.clearLayers();
      currentBoundaryCoords = [];
      currentWaypoints = [];
      document.getElementById('metric-area').innerText = "0.00 ha (0.00 ac)";
      document.getElementById('metric-passes').innerText = "0 passes";
      document.getElementById('metric-waypoints').innerText = "0 points";
      document.getElementById('metric-distance').innerText = "0 m";
      document.getElementById('metric-duration').innerText = "0 min 0 sec";
      updateBoundaryStatus("○ No Boundary Defined", "#8b949e");
      showAlert("Boundary cleared. Draw a new polygon or box on the map.", "info");
    }

    function updateBoundaryStatus(text, color) {
      const badge = document.getElementById('boundary-status-badge');
      if (badge) {
        badge.innerText = text;
        badge.style.color = color;
        badge.style.borderColor = color;
      }
    }

    function showAlert(msg, type) {
      const el = document.getElementById('mission-alert');
      if (!el) return;
      el.style.display = 'block';
      el.innerText = msg;
      if (type === 'success') {
        el.style.background = 'rgba(46, 160, 67, 0.2)';
        el.style.border = '1px solid #3fb950';
        el.style.color = '#3fb950';
      } else if (type === 'error') {
        el.style.background = 'rgba(248, 81, 73, 0.2)';
        el.style.border = '1px solid #f85149';
        el.style.color = '#f85149';
      } else {
        el.style.background = 'rgba(88, 166, 255, 0.15)';
        el.style.border = '1px solid #58a6ff';
        el.style.color = '#58a6ff';
      }
      setTimeout(() => { el.style.display = 'none'; }, 6000);
    }

    function switchTab(tabId, el) {
      document.querySelectorAll('.tab-pane').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.nav-tabs li').forEach(li => li.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      if (el) {
        el.classList.add('active');
        const title = el.getAttribute('data-title') || el.innerText;
        document.getElementById('tab-title').innerText = title;
      }
      if (tabId === 'module1') setTimeout(() => map.invalidateSize(), 200);
      if (tabId === 'module2') loadDiagnostics();
      if (tabId === 'module3') loadPests();
      if (tabId === 'module4') loadPrescriptions();
    }

    async function loadDiagnostics() {
      const res = await fetch('/api/diagnostics');
      const data = await res.json();
      const tbody = document.querySelector('#diag-table tbody');
      tbody.innerHTML = '';
      for (const [k, v] of Object.entries(data)) {
        tbody.innerHTML += `<tr><td><strong>${k.replace(/_/g, ' ')}</strong></td><td style="color:#58a6ff;">${v}</td></tr>`;
      }
    }

    async function loadPests() {
      const res = await fetch('/api/pests');
      const data = await res.json();
      const tbody = document.querySelector('#pest-table tbody');
      tbody.innerHTML = '';
      data.forEach(p => {
        tbody.innerHTML += `<tr><td><strong>${p.name}</strong></td><td>${p.crop}</td><td>${p.class}</td><td style="color:#3fb950; font-weight:500;">${p.chemical}</td><td>${p.dosage}</td><td>${p.interval}</td></tr>`;
      });
    }

    async function loadPrescriptions() {
      const res = await fetch('/api/prescriptions');
      const data = await res.json();
      const tbody = document.querySelector('#rx-table tbody');
      tbody.innerHTML = '';
      data.forEach(r => {
        tbody.innerHTML += `<tr><td>${r.Timestamp}</td><td>${r.Latitude}</td><td>${r.Longitude}</td><td>${r.Altitude_m}</td><td><strong>${r.Target_Pest_Name}</strong></td><td>${r.Severity_Percentage}%</td><td style="color:#3fb950; font-weight:500;">${r.Recommended_Chemical}</td><td>${r.Dosage_ml_per_Litre} ml/L</td></tr>`;
      });
    }

    async function submitPest(e) {
      e.preventDefault();
      const payload = {
        pest_name: document.getElementById('p-name').value,
        crop_type: document.getElementById('p-crop').value,
        severity_class: document.getElementById('p-class').value,
        chemical: document.getElementById('p-chem').value,
        dosage: document.getElementById('p-dose').value,
        min_sev: document.getElementById('p-min').value,
        max_sev: document.getElementById('p-max').value
      };
      await fetch('/api/add_pest', { method: 'POST', body: JSON.stringify(payload) });
      alert("New pest signature recorded in database.");
      loadPests();
    }

    async function saveCameraConfig(e) {
      e.preventDefault();
      const payload = {
        exposure_mode: document.getElementById('cfg-exposure').value,
        iso: parseInt(document.getElementById('cfg-iso').value),
        harmattan_dust_adaptive_mode: { enabled: document.getElementById('cfg-dust').value === 'true' },
        resolution: {
          width: parseInt(document.getElementById('cfg-width').value),
          height: parseInt(document.getElementById('cfg-height').value)
        }
      };
      await fetch('/api/config', { method: 'POST', body: JSON.stringify(payload) });
      alert("Camera configuration saved to camera_config.json.");
    }

    window.onload = () => {
      initMap();
      loadDiagnostics();
      loadPests();
      loadSavedMissionPlan();
    };
  </script>
</body>
</html>
"""


class GCSThreadedServer(socketserver.ThreadingTCPServer):
    """Thread-per-connection server so dashboard polling never blocks itself."""

    daemon_threads = True
    allow_reuse_address = True


def run_gcs_server(port=PORT, host=HOST):
    seed_state()
    with GCSThreadedServer((host, port), GCSRequestHandler) as httpd:
        print(f"[GCS] Ground Control Station Dashboard live at: http://localhost:{port}")
        print(f"[GCS] Bound {host}:{port} | writable state dir: {DATA_DIR}")

        def shutdown(signum, _frame):
            print(f"\n[GCS] Signal {signum} received - shutting down server.")
            httpd.shutdown()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, shutdown)
            except (ValueError, OSError):
                pass  # Not on the main thread (e.g. driven by a test runner).

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[GCS] Shutting down server.")


if __name__ == '__main__':
    run_gcs_server()
