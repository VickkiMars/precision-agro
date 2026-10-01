"""
test_system.py - Unit Testing, Integration Verification & Latency Benchmarks
Precision Agriculture with Drone Base Solution Technology

Verifies all components against the target benchmarks established in Section 4.6.1:
1. Database Query: < 50.0 ms (Observed DOCX: 12.4 ms)
2. Vision Engine ORB / DL: < 200.0 ms (Observed DOCX: 86.2 ms)
3. Telemetry Bridge: < 20.0 ms (Observed DOCX: 4.1 ms)
4. Prescription Engine & CSV Flush: < 10.0 ms (Observed DOCX: 1.8 ms)
"""

import time
import os
import unittest
import numpy as np

from db_init import create_pest_database, seed_agronomic_records
from prescription_engine import PrescriptionEngine
from flight_controller import DroneMissionManager
from vision_engine import EdgeVisionEngine

try:
    import torch
    from model import ResNet9, PLANT_CLASSES
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPrecisionAgroSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_test = "test_crop_health.db"
        cls.log_test = "test_prescription_log.csv"
        cls.geojson_test = "test_prescription_map.geojson"

        # Initialize test DB
        create_pest_database(cls.db_test)
        seed_agronomic_records(cls.db_test)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test files
        for f in [cls.db_test, cls.log_test, cls.geojson_test, f"{cls.db_test}-wal", f"{cls.db_test}-shm"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_01_database_query_latency(self):
        """Benchmark 1: Database Query & BLOB Deserialization (< 50.0 ms)."""
        pe = PrescriptionEngine(db_path=self.db_test)
        latencies = []

        for _ in range(10):
            t0 = time.perf_counter()
            rx = pe.get_prescription("Corn_(maize)___Common_rust_", 55.0)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

        avg_lat = np.mean(latencies)
        print(f"\n[Test 1] DB Query Latency: {avg_lat:.2f} ms (Target: < 50.0 ms)")
        self.assertLess(avg_lat, 50.0, "Database query latency exceeded benchmark.")
        self.assertEqual(rx["action"], "SPOT_SPRAY")
        self.assertIn("Azoxystrobin", rx["chemical"])

    def test_02_resnet9_model_architecture(self):
        """Verify PyTorch ResNet-9 output dimensions and forward pass."""
        if not TORCH_AVAILABLE:
            print("\n[Test 2] PyTorch not installed. Skipping direct tensor test.")
            return

        model = ResNet9(in_channels=3, num_diseases=len(PLANT_CLASSES))
        model.eval()

        dummy_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_tensor)

        self.assertEqual(output.shape, (1, 38), "Output tensor shape must be (1, 38).")
        probs = torch.softmax(output, dim=1)
        self.assertAlmostEqual(torch.sum(probs).item(), 1.0, places=4)
        print("\n[Test 2] ResNet-9 Architecture Output Verified: (1, 38) Softmax Validated.")

    def test_03_vision_engine_latency(self):
        """Benchmark 2: Vision Engine Evaluation (< 200.0 ms)."""
        ve = EdgeVisionEngine(db_path=self.db_test, mode='hybrid')
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_frame[:, :, 1] = 150  # Green canopy

        latencies = []
        for _ in range(5):
            pest, sev, lat = ve.evaluate_frame(dummy_frame)
            latencies.append(lat)

        avg_lat = np.mean(latencies)
        print(f"[Test 3] Vision Engine Latency: {avg_lat:.2f} ms (Target: < 200.0 ms)")
        self.assertLess(avg_lat, 200.0, "Vision Engine latency exceeded benchmark.")

    def test_04_telemetry_bridge_latency(self):
        """Benchmark 3: MAVLink Telemetry Query (< 20.0 ms)."""
        dm = DroneMissionManager(simulation=True, log_file=self.log_test)
        latencies = []

        for _ in range(10):
            lat, lon, alt, latency_ms = dm.get_current_telemetry()
            latencies.append(latency_ms)

        avg_lat = np.mean(latencies)
        print(f"[Test 4] Telemetry Query Latency: {avg_lat:.2f} ms (Target: < 20.0 ms)")
        self.assertLess(avg_lat, 20.0, "Telemetry query latency exceeded benchmark.")
        self.assertAlmostEqual(alt, 5.0, places=1)
        dm.close()

    def test_05_prescription_write_latency_and_export(self):
        """Benchmark 4: Prescription CSV Write (< 10.0 ms) & GeoJSON Generation."""
        dm = DroneMissionManager(simulation=True, log_file=self.log_test)

        write_lat = dm.log_prescription(
            pest_name="Corn_(maize)___Fall_Armyworm_lesion",
            severity_pct=62.5,
            chemical="Chlorantraniliprole 18.5% SC",
            dosage=0.5
        )

        print(f"[Test 5] CSV Prescription Write Latency: {write_lat:.2f} ms (Target: < 10.0 ms)")
        self.assertLess(write_lat, 10.0, "Prescription write latency exceeded benchmark.")

        # Test GeoJSON Export
        success = dm.export_geojson(self.geojson_test)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.geojson_test))
        dm.close()

    def test_06_mission_planner_persistence(self):
        """Benchmark 5: Interactive Mission Plan GeoJSON & Waypoint Persistence."""
        import json
        from gcs_server import GCSRequestHandler

        handler = GCSRequestHandler.__new__(GCSRequestHandler)
        plan = handler.get_active_mission()

        self.assertIn("waypoints", plan)
        self.assertGreater(len(plan["waypoints"]), 0, "Active mission must contain waypoints.")
        self.assertIn("flight_parameters", plan)
        self.assertIn("boundary_geojson", plan)

        params = plan["flight_parameters"]
        self.assertAlmostEqual(params["altitude_agl_m"], 5.0)
        self.assertAlmostEqual(params["speed_mps"], 2.5)
        print(f"[Test 6] Active Mission Plan Verified: {len(plan['waypoints'])} waypoints across {plan['area_hectares']} ha.")


if __name__ == '__main__':
    unittest.main()
