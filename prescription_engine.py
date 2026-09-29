"""
prescription_engine.py - Agronomic Prescription Lookup Engine
Precision Agriculture with Drone Base Solution Technology

Maps target pest detection and quantified infestation severity to site-specific
chemical recommendations and localized dosages (ml/L or g/L).
"""

import sqlite3
import os


class PrescriptionEngine:
    """
    Evaluates detected biological pest anomalies and severity percentages
    against the offline SQLite Pesticide Matrix.
    """

    def __init__(self, db_path='crop_health_edge.db'):
        self.db_path = db_path

    def get_prescription(self, pest_name, severity_pct):
        """
        Query pesticide matrix based on pest name and severity percentage.
        Returns:
            dict containing recommendation details or None if under action threshold.
        """
        if not os.path.exists(self.db_path):
            return {
                "pest_name": pest_name,
                "pest": pest_name,
                "severity_pct": round(severity_pct, 2),
                "severity": round(severity_pct, 2),
                "chemical": "Database Offline - Manual Scouting Required",
                "recommended_chemical": "Database Offline - Manual Scouting Required",
                "dosage_ml_per_litre": 0.0,
                "dosage_ml_per_l": 0.0,
                "dosage": 0.0,
                "action": "MONITOR"
            }

        # Healthy vegetation check
        if "healthy" in pest_name.lower() or severity_pct < 10.0:
            return {
                "pest_name": pest_name,
                "pest": pest_name,
                "severity_pct": round(severity_pct, 2),
                "severity": round(severity_pct, 2),
                "chemical": "None (Healthy Crop Canopy)",
                "recommended_chemical": "None (Healthy Crop Canopy)",
                "dosage_ml_per_litre": 0.0,
                "dosage_ml_per_l": 0.0,
                "dosage": 0.0,
                "action": "NO_ACTION",
                "active_ingredient": "N/A",
                "safety_interval_days": 0
            }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query linked pesticide matrix
        query = '''
        SELECT 
            p.pest_name,
            p.crop_type,
            m.recommended_chemical,
            m.dosage_ml_per_litre,
            m.active_ingredient,
            m.safety_interval_days,
            m.min_severity_pct,
            m.max_severity_pct
        FROM Pest_Signature_Table p
        JOIN Pesticide_Matrix_Table m ON p.pest_id = m.target_pest_id
        WHERE p.pest_name = ?
          AND ? >= m.min_severity_pct
          AND ? <= m.max_severity_pct
        ORDER BY m.dosage_ml_per_litre DESC
        LIMIT 1;
        '''

        cursor.execute(query, (pest_name, severity_pct, severity_pct))
        row = cursor.fetchone()

        if row:
            result = {
                "pest_name": row[0],
                "pest": row[0],
                "crop_type": row[1],
                "chemical": row[2],
                "recommended_chemical": row[2],
                "dosage_ml_per_litre": row[3],
                "dosage_ml_per_l": row[3],
                "dosage": row[3],
                "active_ingredient": row[4],
                "safety_interval_days": row[5],
                "severity_pct": round(severity_pct, 2),
                "severity": round(severity_pct, 2),
                "action": "SPOT_SPRAY"
            }
        else:
            # Fallback for severe infestation above max interval or unseeded pest
            result = {
                "pest_name": pest_name,
                "pest": pest_name,
                "crop_type": "Field Crop",
                "chemical": "Standard Broad-Spectrum Protection (E.g. Mancozeb/Lambda-Cyhalothrin)",
                "recommended_chemical": "Standard Broad-Spectrum Protection (E.g. Mancozeb/Lambda-Cyhalothrin)",
                "dosage_ml_per_litre": 1.5,
                "dosage_ml_per_l": 1.5,
                "dosage": 1.5,
                "active_ingredient": "Standard IPM Formulation",
                "safety_interval_days": 14,
                "severity_pct": round(severity_pct, 2),
                "severity": round(severity_pct, 2),
                "action": "INSPECT_AND_SPOT_TREAT"
            }

        conn.close()
        return result
