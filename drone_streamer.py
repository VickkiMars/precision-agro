"""
drone_streamer.py - Real-Time Optical Inflow Simulator & MAVLink Geotagging
Precision Agriculture with Drone Base Solution Technology

Simulates the real-time inflow of camera frames acquired by the drone's
Sony IMX219 CSI camera payload during low-altitude parcel scouting.
Iterates chronologically through captured field images, performs edge
vision evaluation (ResNet-9 / ORB matching), tags telemetry coordinates,
and queries the agronomic pesticide matrix.
"""

import os
import time
import glob
import threading
from typing import Dict, Any, Optional

from vision_engine import EdgeVisionEngine
from flight_controller import DroneMissionManager
from prescription_engine import PrescriptionEngine


class DroneCameraStreamer:
    """
    Orchestrates the continuous or stepped inflow of drone camera frames
    from the downscaled real-time capture dataset.
    """

    def __init__(self,
                 captures_dir: str = 'realtime_captures',
                 db_path: str = 'crop_health_edge.db',
                 log_file: str = 'field_prescription_log.csv',
                 geojson_file: str = 'field_prescription_map.geojson',
                 mode: str = 'hybrid'):
        self.captures_dir = captures_dir
        self.db_path = db_path
        self.log_file = log_file
        self.geojson_file = geojson_file

        # Load and sort captured images chronologically
        self.frame_paths = []
        if os.path.isdir(self.captures_dir):
            self.frame_paths = sorted(
                glob.glob(os.path.join(self.captures_dir, '*.jpg')) +
                glob.glob(os.path.join(self.captures_dir, '*.png')) +
                glob.glob(os.path.join(self.captures_dir, '*.jpeg'))
            )

        self.total_frames = len(self.frame_paths)
        self.current_index = 0
        self.is_streaming = False
        self.stream_interval_s = 1.0  # 1 FPS default simulation rate
        self._thread = None
        self._lock = threading.Lock()

        # Edge Subsystems
        self.vision = EdgeVisionEngine(db_path=db_path, mode=mode)
        self.drone = DroneMissionManager(simulation=True, log_file=log_file)
        self.prescription = PrescriptionEngine(db_path=db_path)

        # Last processed state cache
        self.last_state: Dict[str, Any] = {
            "status": "ready",
            "frame_index": 0,
            "total_frames": self.total_frames,
            "filename": os.path.basename(self.frame_paths[0]) if self.frame_paths else "none.jpg",
            "image_path": self.frame_paths[0] if self.frame_paths else None,
            "lat": self.drone.sim_lat,
            "lon": self.drone.sim_lon,
            "alt": self.drone.sim_alt,
            "speed_mps": 2.5,
            "pest_name": "Healthy Canopy",
            "severity_pct": 0.0,
            "latency_ms": 0.0,
            "chemical": "None",
            "dosage": 0.0,
            "action": "NO_ACTION",
            "is_streaming": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Warm up with initial frame if available
        if self.frame_paths:
            self._process_frame_at_index(0, log_to_csv=False)

    def _process_frame_at_index(self, index: int, log_to_csv: bool = True) -> Dict[str, Any]:
        """Process frame at given index and update telemetry + diagnosis."""
        if not self.frame_paths:
            return self.last_state

        idx = index % self.total_frames
        fpath = self.frame_paths[idx]
        fname = os.path.basename(fpath)

        # 1. Telemetry Geotag
        lat, lon, alt, tel_lat = self.drone.get_current_telemetry()

        # 2. Edge Vision Evaluation on real captured image
        pest_name, severity_pct, vis_lat = self.vision.evaluate_frame(fpath)

        # 3. Prescriptive Decision Query
        rx = self.prescription.get_prescription(pest_name, severity_pct)

        # 4. Optional CSV / GeoJSON logging
        if log_to_csv and rx["action"] != "NO_ACTION":
            self.drone.log_prescription(
                pest_name=rx["pest_name"],
                severity_pct=rx["severity_pct"],
                chemical=rx["chemical"],
                dosage=rx["dosage_ml_per_litre"]
            )
            self.drone.export_geojson(self.geojson_file)

        state = {
            "status": "inflow_active",
            "frame_index": idx + 1,
            "total_frames": self.total_frames,
            "filename": fname,
            "image_path": fpath,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "alt": round(alt, 1),
            "speed_mps": 2.5,
            "pest_name": rx["pest_name"],
            "severity_pct": round(severity_pct, 1),
            "latency_ms": round(vis_lat, 2),
            "chemical": rx["chemical"],
            "dosage": rx["dosage_ml_per_litre"],
            "action": rx["action"],
            "is_streaming": self.is_streaming,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with self._lock:
            self.current_index = idx
            self.last_state = state

        return state

    def step_next(self) -> Dict[str, Any]:
        """Manually advance to the next captured frame in sequence."""
        next_idx = (self.current_index + 1) % max(1, self.total_frames)
        return self._process_frame_at_index(next_idx, log_to_csv=True)

    def get_current_state(self) -> Dict[str, Any]:
        """Query current drone camera and diagnostic state."""
        with self._lock:
            state = dict(self.last_state)
            state["is_streaming"] = self.is_streaming
            return state

    def get_current_frame_bytes(self) -> Optional[bytes]:
        """Return raw JPEG bytes of current frame."""
        with self._lock:
            fpath = self.last_state.get("image_path")
        if fpath and os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                return f.read()
        return None

    def start_streaming(self, interval: float = 1.0):
        """Begin automated background streaming of incoming drone frames."""
        if self.is_streaming:
            return
        self.stream_interval_s = max(0.2, interval)
        self.is_streaming = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print(f"[DroneStreamer] Background image inflow active @ {1.0/self.stream_interval_s:.1f} FPS.")

    def pause_streaming(self):
        """Pause automated stream loop."""
        self.is_streaming = False
        print("[DroneStreamer] Background image inflow paused.")

    def _stream_loop(self):
        while self.is_streaming:
            self.step_next()
            time.sleep(self.stream_interval_s)


# Global singleton instance for GCS dashboard and mission runners
_streamer_instance: Optional[DroneCameraStreamer] = None


def get_drone_streamer(captures_dir='realtime_captures', db_path='crop_health_edge.db') -> DroneCameraStreamer:
    global _streamer_instance
    if _streamer_instance is None:
        _streamer_instance = DroneCameraStreamer(captures_dir=captures_dir, db_path=db_path)
    return _streamer_instance


if __name__ == '__main__':
    streamer = get_drone_streamer()
    print(f"Loaded {streamer.total_frames} drone capture frames.")
    for i in range(5):
        s = streamer.step_next()
        print(f"[{s['frame_index']:02d}/{s['total_frames']}] {s['filename']} | Lat: {s['lat']}, Lon: {s['lon']} | {s['pest_name']} ({s['severity_pct']}%) -> {s['action']}")
