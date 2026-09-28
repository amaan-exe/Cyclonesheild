import os
import json
import uuid
import datetime
from urllib.parse import urlparse
import bcrypt
import psycopg2

def _load_env_file():
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.path.isfile(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                            v = v[1:-1]
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    parsed = urlparse(DATABASE_URL)
    DB_CONFIG = {
        "dbname": parsed.path.lstrip("/") or "postgres",
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 5432
    }
    if parsed.hostname not in ("127.0.0.1", "localhost") or "DB_SSLMODE" in os.environ:
        DB_CONFIG["sslmode"] = os.environ.get("DB_SSLMODE", "require")
else:
    DB_CONFIG = {
        "dbname": os.environ.get("DB_NAME", "postgres"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASS", ""),
        "host": os.environ.get("DB_HOST", "127.0.0.1"),
        "port": int(os.environ.get("DB_PORT", 5432))
    }
    if DB_CONFIG["host"] not in ("127.0.0.1", "localhost") or "DB_SSLMODE" in os.environ:
        DB_CONFIG["sslmode"] = os.environ.get("DB_SSLMODE", "require")

def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def init_and_seed():
    if DATABASE_URL:
        dsn = DATABASE_URL.replace("postgres://", "postgresql://", 1) if DATABASE_URL.startswith("postgres://") else DATABASE_URL
        conn = psycopg2.connect(dsn)
    else:
        conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Execute schema.sql
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    cur.execute(sql)
    print("Executed schema.sql successfully.")

    # Enable auth service mode for initial seeding
    cur.execute("SET app.is_auth_service = 'true';")

    # 2. Seed citizen: Amanullah
    citizen_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO citizens (id, aadhaar_number, name, district, mobile_masked, risk_zone)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (citizen_id, "123412341234", "Amanullah", "Puri", "XXXX-XXXX-9876", "High"))

    # Also seed a second citizen for cross-citizen RLS isolation testing
    citizen2_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO citizens (id, aadhaar_number, name, district, mobile_masked, risk_zone)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (citizen2_id, "987654321098", "Ramesh Kumar", "Balasore", "XXXX-XXXX-1234", "Moderate"))

    print(f"Seeded citizens (Amanullah: {citizen_id}, Ramesh: {citizen2_id})")

    # 3. Seed authorities
    authorities = [
        (str(uuid.uuid4()), "forecaster_hq", hash_pw("imd@2026"), "imd_forecaster", None),
        (str(uuid.uuid4()), "admin_puri", hash_pw("admin@puri"), "district_admin", "Puri"),
        (str(uuid.uuid4()), "ngo_puri", hash_pw("ngo@puri"), "ngo", "Puri"),
        (str(uuid.uuid4()), "admin_balasore", hash_pw("admin@balasore"), "district_admin", "Balasore")
    ]
    for auth in authorities:
        cur.execute("""
            INSERT INTO authorities (id, user_id, password_hash, role, district_scope)
            VALUES (%s, %s, %s, %s, %s);
        """, auth)
    print("Seeded authorities.")

    # 4. Seed active storm: Cyclone Dana
    storm_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO storms (id, name, status)
        VALUES (%s, %s, %s);
    """, (storm_id, "Cyclone Dana", "active"))
    print(f"Seeded active storm (ID: {storm_id})")

    # 5. Seed realistic ML Advisory payload
    now = datetime.datetime.now(datetime.timezone.utc)
    advisory_payload = {
        "storm_id": storm_id,
        "storm_name": "Cyclone Dana",
        "bulletin_number": "BOB/06/2026/14",
        "bulletin_time": now.isoformat(),
        "warning_status": "RED ALERT — SEVERE CYCLONIC STORM",
        "advisory_level": "RED",
        "requires_human_confirmation": False,
        "current_state": {
            "lat": 18.42,
            "lon": 86.85,
            "max_wind_kt": 65.0,
            "max_wind_kph": 120.4,
            "min_pressure_hpa": 982.0,
            "imd_category": "Very Severe Cyclonic Storm",
            "imd_code": "VSCS",
            "movement_speed_kph": 16.5,
            "movement_direction_deg": 325.0,
            "estimated_landfall_time": (now + datetime.timedelta(hours=14)).strftime("%Y-%m-%d %H:00 UTC"),
            "landfall_location": "Between Puri and Dhamra (Odisha Coast)",
            "dvorak_t_number": "T4.0",
            "system_phase": "Intensifying"
        },
        "model_metadata": {
            "pipeline": "CycloneInferencePipeline (5-Model Multi-Head)",
            "vortex_detector": "U-Net IR Center Localizer (98.2% acc)",
            "pattern_classifier": "Dvorak Multi-Task CNN + Grad-CAM",
            "intensity_classifier": "Hybrid Thermodynamic Stacking Classifier",
            "track_predictor": "Physics-Informed Beta-Advection GRU Seq2Seq",
            "rl_correction": "PPO / CQL Offline Policy Correction (Active)",
            "satellite_feed": "ISRO INSAT-3DR / MOSDAC Calibrated Tb (Channel IR1)",
            "sst_feed": "NOAA OISST High-Resolution (29.4°C Warm Core)",
            "shear_feed": "GFS 850-200 hPa Deep Vertical Shear (12.4 kt, favorable)"
        },
        "track_points": [
            {"lead_h": -18, "lat": 16.20, "lon": 89.10, "wind_kt": 40.0, "wind_kph": 74.1, "pressure_hpa": 996.0, "imd_category": "Cyclonic Storm", "imd_code": "CS", "is_past": True},
            {"lead_h": -12, "lat": 16.90, "lon": 88.35, "wind_kt": 48.0, "wind_kph": 88.9, "pressure_hpa": 992.0, "imd_category": "Severe Cyclonic Storm", "imd_code": "SCS", "is_past": True},
            {"lead_h": -6, "lat": 17.65, "lon": 87.60, "wind_kt": 58.0, "wind_kph": 107.4, "pressure_hpa": 986.0, "imd_category": "Severe Cyclonic Storm", "imd_code": "SCS", "is_past": True},
            {"lead_h": 0, "lat": 18.42, "lon": 86.85, "wind_kt": 65.0, "wind_kph": 120.4, "pressure_hpa": 982.0, "imd_category": "Very Severe Cyclonic Storm", "imd_code": "VSCS", "is_current": True}
        ],
        "predictions": [
            {
                "lead_h": 6,
                "timestamp": (now + datetime.timedelta(hours=6)).isoformat(),
                "lat": 19.18,
                "lon": 86.15,
                "wind_kt": 72.0,
                "wind_kph": 133.3,
                "pressure_hpa": 976.0,
                "imd_category": "Very Severe Cyclonic Storm",
                "imd_code": "VSCS",
                "lat_uncertainty_km": 28.0,
                "lon_uncertainty_km": 26.5,
                "confidence": 0.94,
                "rl_nudge": {"dlat_deg": -0.04, "dlon_deg": -0.03, "dwind_kt": 2.1}
            },
            {
                "lead_h": 12,
                "timestamp": (now + datetime.timedelta(hours=12)).isoformat(),
                "lat": 19.92,
                "lon": 85.50,
                "wind_kt": 78.0,
                "wind_kph": 144.5,
                "pressure_hpa": 970.0,
                "imd_category": "Very Severe Cyclonic Storm",
                "imd_code": "VSCS",
                "lat_uncertainty_km": 42.0,
                "lon_uncertainty_km": 39.0,
                "confidence": 0.91,
                "rl_nudge": {"dlat_deg": -0.07, "dlon_deg": -0.05, "dwind_kt": 3.4}
            },
            {
                "lead_h": 24,
                "timestamp": (now + datetime.timedelta(hours=24)).isoformat(),
                "lat": 20.75,
                "lon": 84.80,
                "wind_kt": 55.0,
                "wind_kph": 101.9,
                "pressure_hpa": 988.0,
                "imd_category": "Severe Cyclonic Storm",
                "imd_code": "SCS",
                "lat_uncertainty_km": 68.0,
                "lon_uncertainty_km": 62.0,
                "confidence": 0.84,
                "rl_nudge": {"dlat_deg": -0.09, "dlon_deg": -0.08, "dwind_kt": -2.0}
            },
            {
                "lead_h": 48,
                "timestamp": (now + datetime.timedelta(hours=48)).isoformat(),
                "lat": 22.10,
                "lon": 84.10,
                "wind_kt": 35.0,
                "wind_kph": 64.8,
                "pressure_hpa": 1002.0,
                "imd_category": "Cyclonic Storm",
                "imd_code": "CS",
                "lat_uncertainty_km": 110.0,
                "lon_uncertainty_km": 98.0,
                "confidence": 0.72,
                "rl_nudge": {"dlat_deg": -0.12, "dlon_deg": -0.10, "dwind_kt": -1.5}
            },
            {
                "lead_h": 72,
                "timestamp": (now + datetime.timedelta(hours=72)).isoformat(),
                "lat": 23.40,
                "lon": 83.70,
                "wind_kt": 22.0,
                "wind_kph": 40.7,
                "pressure_hpa": 1008.0,
                "imd_category": "Depression",
                "imd_code": "D",
                "lat_uncertainty_km": 165.0,
                "lon_uncertainty_km": 145.0,
                "confidence": 0.61,
                "rl_nudge": {"dlat_deg": -0.15, "dlon_deg": -0.12, "dwind_kt": 0.0}
            }
        ],
        "wind_radii_nm": {
            "quadrant_ne": {"r34_kt": 120, "r50_kt": 60, "r64_kt": 30},
            "quadrant_se": {"r34_kt": 110, "r50_kt": 55, "r64_kt": 25},
            "quadrant_sw": {"r34_kt": 80, "r50_kt": 40, "r64_kt": 20},
            "quadrant_nw": {"r34_kt": 95, "r50_kt": 45, "r64_kt": 22}
        },
        "district_risk_matrix": [
            {"district": "Puri", "state": "Odisha", "risk_level": "Severe (Red)", "wind_forecast_kph": "120-145", "surge_m": "2.5-3.2", "rainfall_mm": "220-300", "evacuation_status": "Mandatory in progress"},
            {"district": "Jagatsinghpur", "state": "Odisha", "risk_level": "Severe (Red)", "wind_forecast_kph": "110-135", "surge_m": "2.0-2.8", "rainfall_mm": "190-260", "evacuation_status": "Mandatory in progress"},
            {"district": "Kendrapara", "state": "Odisha", "risk_level": "Severe (Red)", "wind_forecast_kph": "110-130", "surge_m": "1.8-2.5", "rainfall_mm": "180-240", "evacuation_status": "Mandatory in progress"},
            {"district": "Bhadrak", "state": "Odisha", "risk_level": "High (Orange)", "wind_forecast_kph": "90-115", "surge_m": "1.2-1.8", "rainfall_mm": "150-200", "evacuation_status": "Vulnerable zones advised"},
            {"district": "Balasore", "state": "Odisha", "risk_level": "High (Orange)", "wind_forecast_kph": "85-110", "surge_m": "1.0-1.5", "rainfall_mm": "130-180", "evacuation_status": "Vulnerable zones advised"},
            {"district": "Ganjam", "state": "Odisha", "risk_level": "Moderate (Yellow)", "wind_forecast_kph": "60-80", "surge_m": "0.5-1.0", "rainfall_mm": "80-120", "evacuation_status": "Preparedness alert"}
        ],
        "official_bulletin_text": (
            "THE VERY SEVERE CYCLONIC STORM 'DANA' OVER NORTHWEST & ADJOINING WESTCENTRAL BAY OF BENGAL MOVED NORTH-NORTHWESTWARDS "
            "WITH A SPEED OF 16.5 KMPH DURING PAST 6 HOURS AND LAY CENTRED AT 14:00 IST OF TODAY, THE 7TH SEPTEMBER 2026 OVER NORTHWEST BAY OF BENGAL "
            "NEAR LATITUDE 18.42°N AND LONGITUDE 86.85°E, ABOUT 140 KM SOUTHEAST OF PURI (ODISHA) AND 210 KM SOUTH-SOUTHEAST OF DHAMRA PORT. "
            "IT IS VERY LIKELY TO CONTINUE TO MOVE NORTH-NORTHWESTWARDS AND CROSS NORTH ODISHA COAST BETWEEN PURI AND DHAMRA DURING EARLY MORNING HOURS "
            "AS A VERY SEVERE CYCLONIC STORM WITH A WIND SPEED OF 120-130 KMPH GUSTING TO 145 KMPH. STORM SURGE OF ABOUT 2.0 TO 3.0 METRES ABOVE "
            "ASTRONOMICAL TIDE IS VERY LIKELY TO INUNDATE LOW LYING COASTAL AREAS OF PURI, JAGATSINGHPUR AND KENDRAPARA DISTRICTS AT THE TIME OF LANDFALL."
        ),
        "impact_advisory": {
            "fishermen_warning": "Total suspension of fishing operations over North & Central Bay of Bengal.",
            "ports_warning": "Great Danger Signal No. GD-10 hoisted at Paradeep and Gopalpur ports. Signal No. 9 at Dhamra.",
            "infrastructure": "Extensive damage to thatched houses, uprooting of large trees, disruption of power and communication lines.",
            "crop_damage": "Flooding of standing paddy and banana plantations in coastal districts."
        }
    }

    cur.execute("""
        INSERT INTO advisories (id, storm_id, created_at, payload)
        VALUES (%s, %s, %s, %s);
    """, (str(uuid.uuid4()), storm_id, now, json.dumps(advisory_payload)))
    print("Seeded active storm advisory payload.")

    # 6. Seed sample SOS requests
    # Amanullah's SOS in Puri
    sos1_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO sos_requests (id, citizen_id, district, location_lat, location_lon, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (sos1_id, citizen_id, "Puri", 19.8135, 85.8312, "pending", now - datetime.timedelta(minutes=25)))

    # Ramesh's SOS in Balasore
    sos2_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO sos_requests (id, citizen_id, district, location_lat, location_lon, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (sos2_id, citizen2_id, "Balasore", 21.4934, 86.9135, "pending", now - datetime.timedelta(minutes=10)))

    print("Seeded sample SOS requests.")

    # 7. Seed shelters (capacity-aware)
    shelters = [
        (str(uuid.uuid4()), "Puri Zilla School Cyclone Shelter", "Puri", 19.8080, 85.8240, 1200, 480, "06752-222100", ["Medical Aid", "Water", "Food Supply", "Generator Power"]),
        (str(uuid.uuid4()), "Brahmagiri Multi-Purpose Cyclone Shelter", "Puri", 19.8010, 85.6540, 850, 310, "06752-228340", ["Water", "Food Supply", "Sanitation"]),
        (str(uuid.uuid4()), "Konark Community Relief Center", "Puri", 19.8876, 86.0945, 1000, 720, "06752-235122", ["Medical Aid", "Water", "Dry Rations", "Solar Backup"]),
        (str(uuid.uuid4()), "Astaranga Coastal High School Shelter", "Puri", 19.9820, 86.2710, 600, 150, "06752-241090", ["Water", "First Aid", "Sleeping Mats"]),
        (str(uuid.uuid4()), "Balasore Town Hall Relief Camp", "Balasore", 21.4950, 86.9310, 1500, 620, "06782-262015", ["Medical Aid", "Hot Meals", "Power Backup"]),
        (str(uuid.uuid4()), "Chandipur Coastal Evacuation Shelter", "Balasore", 21.4420, 87.0120, 900, 540, "06782-273110", ["Water", "First Aid", "Rescue Boat Point"]),
        (str(uuid.uuid4()), "Paradeep Port Trust Shelter", "Jagatsinghpur", 20.3160, 86.6110, 2000, 1450, "06722-222044", ["Emergency Hospital", "Water", "Power", "SAT Phone"]),
        (str(uuid.uuid4()), "Erasama Community Cyclone Shelter", "Jagatsinghpur", 20.1980, 86.4250, 800, 390, "06722-231190", ["Water", "Food", "Medical Unit"])
    ]
    for s in shelters:
        cur.execute("""
            INSERT INTO shelters (id, name, district, location_lat, location_lon, total_capacity, current_occupancy, contact_number, facilities)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, s)
    print("Seeded capacity-aware shelters.")

    cur.close()
    conn.close()
    print("Database initialization and seeding completed successfully!")

if __name__ == "__main__":
    init_and_seed()
