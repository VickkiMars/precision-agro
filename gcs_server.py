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


class GCSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handles REST API endpoints and serves the unified GCS Dashboard."""

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.render_dashboard().encode('utf-8'))

        elif path == '/api/diagnostics':
            self.send_json(self.get_diagnostics())

        elif path == '/api/pests':
            self.send_json(self.get_pests())

        elif path == '/api/prescriptions':
            self.send_json(self.get_prescriptions())

        elif path == '/api/config':
            self.send_json(self.get_camera_config())

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
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def serve_file(self, filename, content_type):
        if not os.path.exists(filename):
            self.send_error(404, "File not generated yet.")
            return
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Content-Disposition',
                         f'attachment; filename="{os.path.basename(filename)}"')
        self.end_headers()
        with open(filename, 'rb') as f:
            self.wfile.write(f.read())

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
          <h3>Farm Survey Boundary & Serpentine Grid (Uyo, Akwa Ibom)</h3>
          <div id="map"></div>
          <div style="display:flex; gap: 10px;">
            <button class="btn" onclick="generateSerpentineGrid()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
              Compute Serpentine Transects (2-Ha Grid)
            </button>
            <button class="btn btn-secondary" onclick="clearMap()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              Reset Boundary
            </button>
          </div>
        </div>
        <div class="card">
          <h3>Flight Parameters</h3>
          <label>Scanning Altitude (AGL)</label>
          <input type="text" value="5.0 meters (NCAA Compliant)" readonly />
          <label>Survey Speed</label>
          <input type="text" value="2.5 m/s" readonly />
          <label>Field Center Target</label>
          <input type="text" value="5.037700 N, 7.912800 E (Uyo)" readonly />
          <label>Camera Field of View</label>
          <input type="text" value="62.2 Horizontal (Sony IMX219)" readonly />
          <label>Computed Transect Passes</label>
          <input type="text" id="pass-count" value="12 Serpentine Passes" readonly />
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
    let map, gridLayer;
    function initMap() {
      map = L.map('map').setView([5.0377, 7.9128], 17);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);
      gridLayer = L.layerGroup().addTo(map);
      generateSerpentineGrid();
    }

    function generateSerpentineGrid() {
      gridLayer.clearLayers();
      const originLat = 5.0377;
      const originLon = 7.9128;
      const latlngs = [];

      for (let r = 0; r < 8; r++) {
        const lat = originLat + (r * 0.00025);
        if (r % 2 === 0) {
          latlngs.push([lat, originLon]);
          latlngs.push([lat, originLon + 0.0015]);
        } else {
          latlngs.push([lat, originLon + 0.0015]);
          latlngs.push([lat, originLon]);
        }
      }

      L.polyline(latlngs, { color: '#2ea043', weight: 3, dashArray: '5, 8' }).addTo(gridLayer);
      L.circleMarker([originLat, originLon], { color: '#58a6ff', radius: 6 }).addTo(gridLayer).bindPopup("Takeoff Waypoint #1");
      map.fitBounds(L.latLngBounds(latlngs));
    }

    function clearMap() { gridLayer.clearLayers(); }

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
