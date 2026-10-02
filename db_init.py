"""
db_init.py - Offline Database Instantiation & Agronomic Seeding
Precision Agriculture with Drone Base Solution Technology

Creates and seeds:
1. Pest_Signature_Table (keypoints/descriptors BLOBs for edge matching)
2. Crop_Profile_Table (baseline healthy profiles)
3. Pesticide_Matrix_Table (severity thresholds, chemicals, dosages)
"""

import sqlite3
import numpy as np
import os


def create_pest_database(db_path='crop_health_edge.db'):
    """Instantiate offline relational database schema with WAL mode enabled."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable Write-Ahead Logging for high-concurrency, low-latency edge operations
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    # 1. Create Pest Signature Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Pest_Signature_Table (
        pest_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pest_name TEXT NOT NULL UNIQUE,
        crop_type TEXT NOT NULL,
        keypoints_blob BLOB NOT NULL,
        descriptors_blob BLOB NOT NULL,
        severity_class TEXT NOT NULL
    );
    ''')

    # 2. Create Crop Profile Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Crop_Profile_Table (
        crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop_type TEXT NOT NULL UNIQUE,
        baseline_healthy_color TEXT NOT NULL,
        growth_stage TEXT NOT NULL
    );
    ''')

    # 3. Create Pesticide Matrix Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Pesticide_Matrix_Table (
        matrix_id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_pest_id INTEGER NOT NULL,
        min_severity_pct REAL NOT NULL,
        max_severity_pct REAL NOT NULL,
        recommended_chemical TEXT NOT NULL,
        dosage_ml_per_litre REAL NOT NULL,
        active_ingredient TEXT,
        safety_interval_days INTEGER DEFAULT 7,
        FOREIGN KEY (target_pest_id) REFERENCES Pest_Signature_Table (pest_id)
    );
    ''')

    conn.commit()
    conn.close()
    print(f"[DB] Initialized database schema at: {db_path}")


def generate_synthetic_descriptors(seed=42, n_features=100):
    """
    Generate realistic 32-byte binary ORB descriptors.
    OpenCV ORB descriptors are shape (N, 32) uint8 arrays.
    """
    rng = np.random.default_rng(seed)
    descriptors = rng.integers(0, 256, size=(n_features, 32), dtype=np.uint8)
    # Dummy keypoints coordinates (N, 2)
    keypoints = rng.uniform(0, 640, size=(n_features, 2)).astype(np.float32)
    return keypoints.tobytes(), descriptors.tobytes()


def extract_real_or_synthetic_descriptors(pest_name, pest_dir='pest_database', seed=42, n_features=300):
    """
    Extract real OpenCV ORB keypoints and descriptors from pest database images if available.
    Falls back to synthetic descriptors if images are not present.
    """
    if os.path.exists(pest_dir):
        class_folder = os.path.join(pest_dir, pest_name)
        if os.path.isdir(class_folder):
            try:
                import cv2
                orb = cv2.ORB_create(nfeatures=n_features)
                all_descs = []
                all_kps = []
                # Scan up to 10 representative images from the class
                image_files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:10]
                for fname in image_files:
                    fpath = os.path.join(class_folder, fname)
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        kp, des = orb.detectAndCompute(img, None)
                        if des is not None and len(des) > 0:
                            all_descs.append(des)
                            pts = np.array([p.pt for p in kp], dtype=np.float32)
                            all_kps.append(pts)
                            if sum(len(d) for d in all_descs) >= n_features:
                                break
                if all_descs:
                    stacked_des = np.vstack(all_descs)[:n_features].astype(np.uint8)
                    stacked_kps = np.vstack(all_kps)[:n_features].astype(np.float32)
                    return stacked_kps.tobytes(), stacked_des.tobytes()
            except Exception as e:
                print(f"[DB] Notice: Could not extract OpenCV descriptors for {pest_name} ({e}). Using synthetic fallback.")

    return generate_synthetic_descriptors(seed=seed, n_features=n_features)


def seed_agronomic_records(db_path='crop_health_edge.db', pest_dir='pest_database'):
    """
    Populate standard agronomic prescription matrices for:
    - Maize / Corn (Fall Armyworm, Stem Borer, Ear Rot, Common Rust, Leaf Blight)
    - Cassava (Cassava Mosaic Disease, Bacterial Blight)
    - Tomato & Potato (Early/Late Blight, Yellow Leaf Curl)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Seed Crop Profiles
    crops = [
        ('Corn_(maize)', '#3B8B2E', 'Vegetative / Tasseling'),
        ('Cassava', '#2E7D32', 'Tuber Expansion / Maturation'),
        ('Tomato', '#43A047', 'Flowering / Fruiting'),
        ('Potato', '#388E3C', 'Tuber Initiation'),
        ('Apple', '#2E7D32', 'Fruit Development')
    ]
    for crop_type, color, stage in crops:
        cursor.execute('''
        INSERT OR IGNORE INTO Crop_Profile_Table (crop_type, baseline_healthy_color, growth_stage)
        VALUES (?, ?, ?)
        ''', (crop_type, color, stage))

    # Seed Pest Signatures & Prescriptions from the Archive Database
    pests_data = [
        # 1. Fall Armyworm in Maize (from archive)
        {
            "name": "Corn_(maize)___Fall_Armyworm_lesion",
            "crop": "Corn_(maize)",
            "class": "Lepidopteran Defoliator",
            "seed": 101,
            "prescriptions": [
                (10.0, 35.0, "Emamectin Benzoate 5% SG", 0.4, "Emamectin Benzoate", 7),
                (35.1, 70.0, "Chlorantraniliprole 18.5% SC", 0.5, "Anthranilic Diamide", 14),
                (70.1, 100.0, "Spinetoram 11.7% SC (Spot Over-Dose)", 0.8, "Spinosyn", 14),
            ]
        },
        # 2. Stem Borer in Maize (from archive)
        {
            "name": "Corn_(maize)___Stem_Borer",
            "crop": "Corn_(maize)",
            "class": "Lepidopteran Borer (Busseola fusca)",
            "seed": 109,
            "prescriptions": [
                (10.0, 40.0, "Deltamethrin 2.8% EC", 0.5, "Pyrethroid", 10),
                (40.1, 75.0, "Chlorantraniliprole 18.5% SC", 0.6, "Anthranilic Diamide", 14),
                (75.1, 100.0, "Carbosulfan 25% EC Spot Barrier", 1.0, "Carbamate", 21),
            ]
        },
        # 3. Ear Rot in Maize (from archive)
        {
            "name": "Corn_(maize)___Ear_Rot",
            "crop": "Corn_(maize)",
            "class": "Fungal Pathology (Fusarium / Aspergillus)",
            "seed": 110,
            "prescriptions": [
                (10.0, 40.0, "Azoxystrobin 23% SC", 0.8, "Strobilurin", 14),
                (40.1, 75.0, "Tebuconazole 250 EC", 1.0, "Triazole", 21),
                (75.1, 100.0, "Flutriafol + Azoxystrobin Dual Action", 1.2, "Dual Action Fungicide", 28),
            ]
        },
        # 4. Maize Common Rust
        {
            "name": "Corn_(maize)___Common_rust_",
            "crop": "Corn_(maize)",
            "class": "Fungal Basidiomycete",
            "seed": 102,
            "prescriptions": [
                (15.0, 45.0, "Mancozeb 75% WP", 2.5, "Dithiocarbamate", 10),
                (45.1, 100.0, "Azoxystrobin 23% SC", 1.0, "Strobilurin", 14),
            ]
        },
        # 5. Maize Northern Leaf Blight
        {
            "name": "Corn_(maize)___Northern_Leaf_Blight",
            "crop": "Corn_(maize)",
            "class": "Fungal Ascomycete",
            "seed": 103,
            "prescriptions": [
                (15.0, 50.0, "Propiconazole 25% EC", 1.0, "Triazole", 14),
                (50.1, 100.0, "Pyraclostrobin + Fluxapyroxad", 1.2, "Dual Action Fungicide", 21),
            ]
        },
        # 6. Cassava Mosaic Disease
        {
            "name": "Cassava___Cassava_Mosaic_Disease",
            "crop": "Cassava",
            "class": "Geminiviridae (Whitefly Vector)",
            "seed": 104,
            "prescriptions": [
                (10.0, 40.0, "Imidacloprid 17.8% SL (Vector Control)", 0.5, "Neonicotinoid", 21),
                (40.1, 75.0, "Acetamiprid 20% SP + Roguing Rogue Nodes", 0.6, "Neonicotinoid", 14),
                (75.1, 100.0, "Uprooting & Destruction Notice + Imidacloprid Perimeter", 1.0, "Total Quarantine", 0),
            ]
        },
        # 7. Tomato Early Blight
        {
            "name": "Tomato___Early_blight",
            "crop": "Tomato",
            "class": "Fungal Foliar Leaf Spot",
            "seed": 105,
            "prescriptions": [
                (15.0, 40.0, "Chlorothalonil 75% WP", 2.0, "Chloronitrile", 7),
                (40.1, 75.0, "Difenoconazole 25% EC", 0.8, "Triazole", 7),
                (75.1, 100.0, "Azoxystrobin 20% + Difenoconazole 12.5% SC", 1.2, "Strobilurin + Triazole", 14),
            ]
        },
        # 8. Tomato Yellow Leaf Curl Virus
        {
            "name": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "crop": "Tomato",
            "class": "Begomovirus (Whitefly Vector)",
            "seed": 106,
            "prescriptions": [
                (15.0, 50.0, "Thiamethoxam 25% WG", 0.4, "Neonicotinoid", 14),
                (50.1, 100.0, "Pyriproxyfen 10% EC (Insect Growth Regulator)", 1.5, "IGR", 14),
            ]
        },
        # 9. Potato Early Blight
        {
            "name": "Potato___Early_blight",
            "crop": "Potato",
            "class": "Fungal Foliar Alternaria",
            "seed": 107,
            "prescriptions": [
                (15.0, 50.0, "Mancozeb 75% WP", 2.5, "Dithiocarbamate", 7),
                (50.1, 100.0, "Famoxadone + Cymoxanil", 1.0, "Oxazolidinedione", 14),
            ]
        },
        # 10. Potato Late Blight
        {
            "name": "Potato___Late_blight",
            "crop": "Potato",
            "class": "Oomycete Phytophthora",
            "seed": 108,
            "prescriptions": [
                (10.0, 45.0, "Metalaxyl-M + Mancozeb (Ridomil Gold)", 2.5, "Phenylamide", 10),
                (45.1, 100.0, "Dimethomorph 50% WP", 1.5, "Cinnamic Acid Derivative", 14),
            ]
        }
    ]

    for item in pests_data:
        kp_blob, desc_blob = extract_real_or_synthetic_descriptors(
            pest_name=item["name"],
            pest_dir=pest_dir,
            seed=item["seed"],
            n_features=300
        )
        cursor.execute('''
        INSERT INTO Pest_Signature_Table 
        (pest_name, crop_type, keypoints_blob, descriptors_blob, severity_class)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(pest_name) DO UPDATE SET
            keypoints_blob=excluded.keypoints_blob,
            descriptors_blob=excluded.descriptors_blob,
            severity_class=excluded.severity_class
        ''', (item["name"], item["crop"], kp_blob, desc_blob, item["class"]))

        # Retrieve pest_id
        cursor.execute("SELECT pest_id FROM Pest_Signature_Table WHERE pest_name = ?", (item["name"],))
        pest_id = cursor.fetchone()[0]

        # Insert Prescriptions
        for min_s, max_s, chem, dosage, ai, phi in item["prescriptions"]:
            cursor.execute('''
            INSERT OR IGNORE INTO Pesticide_Matrix_Table 
            (target_pest_id, min_severity_pct, max_severity_pct, recommended_chemical, dosage_ml_per_litre, active_ingredient, safety_interval_days)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pest_id, min_s, max_s, chem, dosage, ai, phi))

    conn.commit()
    conn.close()
    print(f"[DB] Successfully seeded agronomic records into: {db_path}")


if __name__ == "__main__":
    db_file = "crop_health_edge.db"
    create_pest_database(db_file)
    seed_agronomic_records(db_file)

