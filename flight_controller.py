"""
flight_controller.py - MAVLink Telemetry Bridge & Field Prescription Logger
Precision Agriculture with Drone Base Solution Technology

Coordinates communication with Pixhawk 4 Autopilot via DroneKit/PyMAVLink.
Logs georeferenced spot treatment prescriptions to field_prescription_log.csv.
Includes SITL / simulation fallback for testing without hardware.
"""

import csv
import time
import os
import json

try:
    from dronekit import connect, VehicleMode
    DRONEKIT_AVAILABLE = True
except ImportError:
    DRONEKIT_AVAILABLE = False


class DroneMissionManager:
    """
    Manages telemetry synchronization between the Pixhawk Autopilot and the
    Raspberry Pi companion computer.
    """

    def __init__(self, connection_string='/dev/ttyAMA0', baud=57600, simulation=False, log_file='field_prescription_log.csv'):
        self.connection_string = connection_string
        self.baud = baud
        self.log_file = log_file
        self.simulation = simulation or (not DRONEKIT_AVAILABLE)
        self.vehicle = None

        # Simulation origin (Demarcated field in Uyo, Akwa Ibom State, Nigeria)
        self.sim_lat = 5.037700
        self.sim_lon = 7.912800
        self.sim_alt = 5.0  # 5 meters scanning altitude
        self.step_counter = 0

        # Initialize CSV header if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Latitude',
                    'Longitude',
                    'Altitude_m',
                    'Target_Pest_Name',
                    'Severity_Percentage',
                    'Recommended_Chemical',
                    'Dosage_ml_per_Litre'
                ])

        self._connect()

    def _connect(self):
        """Establish MAVLink connection or initialize simulator."""
        if not self.simulation and DRONEKIT_AVAILABLE:
            try:
                print(f"[FCU] Connecting to Pixhawk Autopilot on {self.connection_string} @ {self.baud} baud...")
                self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baud)
                print("[FCU] Connected to Pixhawk 4 successfully.")
            except Exception as e:
                print(f"[FCU] Hardware serial connection failed ({e}). Reverting to SITL simulation mode.")
                self.simulation = True
        else:
            print("[FCU] Operating in Autonomous Flight Simulation Mode (Uyo Field Coordinates).")

    def get_current_telemetry(self):
        """
        Query real-time WGS84 GPS coordinates and relative altitude.
        Returns:
            lat (float), lon (float), alt (float)
        """
        t0 = time.perf_counter()

        if not self.simulation and self.vehicle:
            location = self.vehicle.location.global_relative_frame
            lat, lon, alt = location.lat, location.lon, location.alt
        else:
            # Generate serpentine coverage transect coordinates (2-hectare grid)
            # 2.5 m/s flight speed at 1 Hz update rate
            self.step_counter += 1
            row = self.step_counter // 20
            col = self.step_counter % 20

            # Delta steps roughly ~2.5 meters in lat/lon
            lat_offset = row * 0.000025
            lon_offset = (col if row % 2 == 0 else (19 - col)) * 0.000025

            lat = self.sim_lat + lat_offset
            lon = self.sim_lon + lon_offset
            alt = self.sim_alt

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return lat, lon, alt, latency_ms

    def log_prescription(self, pest_name, severity_pct, chemical, dosage):
        """
        Append georeferenced spot prescription to the persistent CSV log.
        Returns write latency in milliseconds.
        """
        t0 = time.perf_counter()
        lat, lon, alt, _ = self.get_current_telemetry()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                f"{lat:.6f}",
                f"{lon:.6f}",
                f"{alt:.1f}",
                pest_name,
                f"{severity_pct:.2f}",
                chemical,
                f"{dosage:.2f}"
            ])

        write_latency_ms = (time.perf_counter() - t0) * 1000.0
        return write_latency_ms

    def export_geojson(self, geojson_path='field_prescription_map.geojson'):
        """Convert CSV prescription log into standardized GeoJSON for tractor sprayers."""
        features = []
        if not os.path.exists(self.log_file):
            return False

        with open(self.log_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "timestamp": row['Timestamp'],
                            "altitude": float(row['Altitude_m']),
                            "pest_name": row['Target_Pest_Name'],
                            "severity_pct": float(row['Severity_Percentage']),
                            "chemical": row['Recommended_Chemical'],
                            "dosage_ml_per_l": float(row['Dosage_ml_per_Litre'])
                        }
                    }
                    features.append(feature)
                except (ValueError, KeyError):
                    continue

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)

        print(f"[FCU] Exported GeoJSON prescription map to: {geojson_path}")
        return True

    def close(self):
        """Safely release MAVLink serial handle."""
        if self.vehicle:
            self.vehicle.close()
            print("[FCU] Closed MAVLink vehicle connection.")
