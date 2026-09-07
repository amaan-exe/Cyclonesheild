# Cyclone Shield AI — National Tropical Cyclone AI Identification & Prediction System
### Government of India | India Meteorological Department (IMD) & National Disaster Management Authority (NDMA)
**Smart India Hackathon (SIH 2026) | Problem Statement: 26070 | Team: Hexaminds**

---

## 📑 Table of Contents
1. [Executive Summary & Mission](#1-executive-summary--mission)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The Staged Prediction Funnel (T-18d to T-3d)](#3-the-staged-prediction-funnel-t-18d-to-t-3d)
4. [Backend API Reference & Endpoint Documentation](#4-backend-api-reference--endpoint-documentation)
5. [Core Services & Logic Breakdown](#5-core-services--logic-breakdown)
   - [5.1 Cyclone ML Inference & Replay Service (`backend/ml_service.py`)](#51-cyclone-ml-inference--replay-service)
   - [5.2 Staged Prediction Funnel Service (`backend/prediction_funnel.py`)](#52-staged-prediction-funnel-service)
   - [5.3 Authentication & Session Security (`backend/auth.py`)](#53-authentication--session-security)
   - [5.4 Database Connection Pool & RLS Manager (`backend/database/db.py`)](#54-database-connection-pool--rls-manager)
   - [5.5 Database Schema & Row-Level Security (`backend/database/schema.sql`)](#55-database-schema--row-level-security)
6. [Frontend UI Architecture & Function Documentation](#6-frontend-ui-architecture--function-documentation)
   - [6.1 Executive Disaster Command Layout (`frontend/index.html`)](#61-executive-disaster-command-layout)
   - [6.2 Client Application Controller (`frontend/app.js`)](#62-client-application-controller)
   - [6.3 Government of India Design System (`frontend/style.css`)](#63-government-of-india-design-system)
7. [PostgreSQL Row-Level Security (RLS) Defense In Depth](#7-postgresql-row-level-security-rls-defense-in-depth)
8. [Automated Test Suite & Verification Matrix](#8-automated-test-suite--verification-matrix)
9. [Installation, Setup & Deployment Guide](#9-installation-setup--deployment-guide)

---

## 1. Executive Summary & Mission

**Cyclone Shield AI** is an operational-grade tropical cyclone surveillance, predictive modeling, and early disaster response coordination platform designed for India's vulnerable coastal seaboards (Bay of Bengal and Arabian Sea). 

Instead of treating cyclone identification as a naive, binary *"is there a cyclone?"* computer vision task, the platform implements a **Defense-in-Depth Staged Prediction Funnel** spanning four calibrated forecasting horizons:
- **T-18 Days:** Regional Basin Risk Flagging (SST anomalies & MJO convective enhancement)
- **T-14 Days:** Cyclogenesis Probability Calculation (Genesis Potential Index ensemble)
- **T-7 Days:** Developing Vortex System Identification (Nascent BOB-06 center localization & Dvorak T1.5 tracking)
- **T-3 Days (72h):** High-Resolution Track, Intensity, Landfall Corridor & Wind Hazard Swaths

The platform integrates directly with **INSAT-3D/3DR satellite radiometry**, official **IMD Best Track archives**, **ESRI/NASA cartographic chart overlays**, **PostgreSQL Row-Level Security (RLS)** for multi-tenant district disaster isolation, **capacity-aware cyclone shelter allocation**, and **OASIS Common Alerting Protocol (CAP v1.2)** automated alert feeds.

---

## 2. End-to-End System Architecture

```
                                  [ SATELLITE & SENSOR INGESTION ]
                                INSAT-3D/3DR IR1 (10.8µm) & Best Track
                                                  │
                                                  ▼
                                    [ CYCLONE ML INFERENCE ]
                               Deep Sequence CNN-LSTM + U-Net Localizer
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
    [ 4-STAGE PREDICTION FUNNEL ]                                       [ OFFICIAL ADVISORY ENGINE ]
    T-18d: Basin SST Risk (Amber)                                      IMD Category (VSCS Dana), Waypoints
    T-14d: Cyclogenesis GPI 8.4 (Gold)                                 72h Cone & 3-Tier Wind Swaths (34/50/64 KT)
    T-7d:  Vortex ID BOB-06 (Orange)                                   District Vulnerability Risk Matrix
    T-3d:  72h Track & Landfall (Red)                                  OASIS CAP v1.2 XML / JSON Alerts
                │                                                                   │
                └─────────────────────────────────┬─────────────────────────────────┘
                                                  ▼
                                     [ FASTAPI APPLICATION CORE ]
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ▼                         ▼                         ▼
             [ POSTGRESQL WITH RLS ]    [ LEAFLET GIS ENGINE ]    [ SCOPED EMERGENCY AI ]
             Isolated District Data     IMD Cartographic Chart    Groq AI / Local Rule Fallback
             Citizen SOS Broadcasts     ESRI Optical Satellite    Life-Saving Evacuation Protocols
             Capacity-Aware Shelters    Dynamic Wind Capsules     Toll-Free Helpline Directives
```

### Key Workflow Loops:
1. **Synoptic Prediction Loop:** The backend ML worker evaluates atmospheric frames, persists advisory records with confidence bounds, and dynamically serves real-time telemetry to the dashboard.
2. **Citizen Safety Loop:** Citizens authenticate via masked Aadhaar, view hyper-localized district severity ratings, locate nearest capacity-ready shelters, and broadcast GPS-tagged SOS requests.
3. **Authority Incident Command Loop:** District Administrators (e.g. Puri Admin) and State/National Forecasters log into their respective scoped consoles. PostgreSQL Row-Level Security guarantees zero cross-district data leakage while allowing nationwide command oversight.
4. **Inter-Agency Dissemination:** Automatic generation of OASIS CAP v1.2 XML feeds enables turnkey integration with NDMA SACHET, state cell broadcasts, and early-warning sirens.

---

## 3. The Staged Prediction Funnel (T-18d to T-3d)

Traditional cyclone classifiers only output a static *"cyclone detected"* label when a fully developed system is already threatening land, giving emergency authorities virtually no lead time. Cyclone Shield AI organizes forecasting into four operational horizons:

| Horizon | Stage Name | Physical Atmospheric Triggers | Operational Command Directives | Map Visualization |
| :--- | :--- | :--- | :--- | :--- |
| **T-18 Days** | **Regional Basin Risk** | SST: 29.8°C (+1.4°C anomaly), TCHP: 92 kJ/cm², MJO Phase 3/4 active convection, 850 hPa vorticity | Pre-season advisory issued to NDMA & OSDMA/APSDMA; audit shelter readiness and generator fuel | Basin-wide thermal anomaly polygon (dashed amber) & hotspot core |
| **T-14 Days** | **Cyclogenesis Probability** | Genesis Potential Index (GPI = 8.4), 20-member dynamical ensemble gives 68.5% consensus, vertical wind shear < 12 kt | Cautionary Signal No. 1 hoisted at Paradeep & Vizag; recall deep-sea fishing trawlers | 68.5% cyclogenesis zone polygon (dashed gold) & nascent centroid |
| **T-7 Days** | **System Identification** | Developing vortex center BOB-06 (14.2°N, 89.5°E), Dvorak T1.5 curved band pattern, central pressure drop -6 hPa | Mandatory recall of all fishermen; District Emergency Operations Centers (DEOCs) in 24x7 readiness | Depression vortex center (orange circle) & projected NNW trajectory line |
| **T-3 Days** | **Track & Intensity Cone** | Deep sequence model predicts 72h waypoints, sustained winds 120 km/h (gusts 145), landfall in Puri-Dhamra corridor | Great Danger Signal GD-10; mandatory evacuation within 5 km of coast; pre-position NDRF/ODRAF teams | 72h IMD Uncertainty Cone, 34/50/64 KT wind hazard swaths & track waypoints |

---

## 4. Backend API Reference & Endpoint Documentation

The FastAPI application (`backend/main.py`) exposes 16 production endpoints across authentication, meteorology, staged prediction, incident dispatch, and multi-agency alerting:

### 4.1 Authentication Endpoints
- **`POST /api/auth/citizen`**
  - **Description:** Authenticates citizen via 12-digit Aadhaar number under privileged auth service context. Returns a scoped JWT bearer token.
  - **Request:** `{"aadhaar_number": "123412341234"}`
  - **Response:** `{"access_token": "...", "token_type": "bearer", "citizen": {"id": "...", "name": "Amanullah", "district": "Puri", "risk_zone": "High"}}`
- **`POST /api/auth/authority`**
  - **Description:** Authenticates operational authorities (IMD forecaster, District Admin, NGO). Validates bcrypt password hashes and injects `role` and `district_scope` into the session token.
  - **Request:** `{"user_id": "admin_puri", "password": "..."}`
  - **Response:** `{"access_token": "...", "token_type": "bearer", "authority": {"user_id": "admin_puri", "role": "district_admin", "district_scope": "Puri"}}`
- **`GET /api/auth/me`**
  - **Description:** Returns profile metadata for the authenticated token, verifying active session validity.

### 4.2 Storm Advisory & Synoptic Meteorology
- **`GET /api/storms/active`**
  - **Description:** Returns the active storm's latest synoptic state, historical track coordinates, 72h forecast waypoints, 3-tier wind radii, and district risk vulnerability table.
  - **Response:** `{"storm_id": "...", "storm_name": "Cyclone Dana", "advisory": {"warning_status": "RED ALERT...", "current_state": {...}, "track_points": [...], "predictions": [...]}}`
- **`GET /api/storms/{storm_id}/advisories`**
  - **Description:** Retrieves paginated historical synoptic advisories for hindcast replay.

### 4.3 Staged Prediction Funnel
- **`GET /api/ml/prediction-funnel`**
  - **Description:** Returns the complete 4-tier operational prediction funnel data (T-18d, T-14d, T-7d, T-3d) with physics triggers, lead times, and comparison against naive binary classifiers.
- **`GET /api/ml/prediction-funnel/stage/{stage_key}`**
  - **Description:** Returns deep physical parameters, spatial geometries, and command actions for a specific horizon key (`t18_regional_risk`, `t14_cyclogenesis`, `t7_system_id`, `t3_track_intensity`).

### 4.4 Citizen SOS & Emergency Dispatch (RLS Enforced)
- **`POST /api/sos`** *(Requires Citizen Token)*
  - **Description:** Broadcasts an emergency distress call with GPS coordinates. Bound strictly to the citizen's own account via database RLS.
  - **Request:** `{"district": "Puri", "location_lat": 19.81, "location_lon": 85.83}`
- **`GET /api/sos/my`** *(Requires Citizen Token)*
  - **Description:** Retrieves the authenticated citizen's personal distress calls and rescue status lifecycle (`pending` &rarr; `in_progress` &rarr; `resolved`).
- **`GET /api/sos/queue`** *(Requires Authority Token)*
  - **Description:** Returns the live emergency incident queue. Filtered at the PostgreSQL database engine level using Row-Level Security based on the authority's `district_scope`.
- **`PATCH /api/sos/{sos_id}`** *(Requires Authority Token)*
  - **Description:** Updates the lifecycle status of an emergency request (`in_progress` / `resolved`). Forbidden across district boundaries.

### 4.5 Capacity-Aware Shelter Network
- **`GET /api/shelters`**
  - **Description:** Lists all designated multi-purpose cyclone shelters, filtered optionally by `district`. Includes total capacity, occupied beds, remaining available beds, contact numbers, and emergency amenities (Water, Medical, Generators).

### 4.6 Scoped Emergency AI Assistant
- **`POST /api/chat`**
  - **Description:** AI emergency assistant powered by **Groq AI (Llama-3.3 / Qwen-2.5)** with zero-downtime deterministic local fallback for disaster shelter locations, dos & don'ts, landfall ETA, and helpline routing.

### 4.7 Multi-Agency Standards & Metrics
- **`GET /api/alerts/cap`** & **`GET /api/alerts/cap.xml`**
  - **Description:** Exports official **OASIS Common Alerting Protocol v1.2** disaster feeds (JSON & XML formats) for automated integration into NDMA SACHET and mobile alert sirens.
- **`GET /api/system/pipeline-status`**
  - **Description:** Technical architectural breakdown of the 6-layer AI/ML pipeline, Grad-CAM explainability, and Monte Carlo Dropout uncertainty quantification.
- **`GET /api/stakeholders/metrics`**
  - **Description:** Operational intelligence metrics across Citizens, Emergency Services (NDRF/ODRAF), NGOs/Shelters, and Volunteer networks.
- **`GET /api/odisha/osdma-metrics`**
  - **Description:** Historical benchmarks from the Odisha Zero Casualty Model (1999 Super Cyclone vs 2013 Phailin vs 2019 Fani vs 2026 Dana) and UN Early Warnings for All (EW4All) pillars.
- **`POST /api/ml/replay`** *(Requires Authority Token)*
  - **Description:** Steps the ML inference engine forward to ingest the next synoptic observation frame and publish an updated official bulletin.

### 4.8 Surge-Safe Evacuation Corridor & Shelter Guidance
- **`GET /api/shelters/evacuation-route`**
  - **Description:** Computes a surge-safe inland evacuation corridor from citizen GPS coordinates to the nearest capacity-ready shelter. Bypasses vulnerable coastal strips (<2 km from shore) and marine drives onto elevated roadways (Grand Road / Bada Danda high ground).
  - **Parameters:** `origin_lat` (float), `origin_lon` (float), `shelter_id` (optional string), `district` (optional string).
  - **Response:**
    ```json
    {
      "target_shelter": { "id": "...", "name": "Puri Zilla School Cyclone Shelter", "available_beds": 720, "occupancy_rate": "40.0%" },
      "distance_km": 1.8,
      "eta_walking_mins": 21,
      "eta_vehicle_mins": 5,
      "surge_safety_status": "CERTIFIED SAFE — ELEVATED INLAND CORRIDOR",
      "route_coordinates": [[19.805, 85.830], [19.809, 85.827], [19.814, 85.831]],
      "turn_by_turn": [
        { "step": 1, "instruction": "Evacuate inland away from coastline.", "distance": "0.6 km", "safe_marker": "Inland Road" },
        { "step": 2, "instruction": "Merge onto Grand Road high ground corridor.", "distance": "1.2 km", "safe_marker": "Flood-Free Zone" }
      ]
    }
    ```

### 4.9 Offline Disaster Pack & Emergency Data Bundle
- **`GET /api/offline/emergency-pack`**
  - **Description:** Returns a self-contained, standalone offline emergency safety bundle cached directly into device `localStorage`. Enables complete offline shelter discovery, survival instructions, and helpline routing when cellular towers fail.
  - **Contents:** Pre-cached shelter list with bed counts, emergency helplines (112, 1070, 1077, 1078), 140-char 2G GSM SMS formatting templates, USSD menu dial codes (`*112*1#`), and multilingual survival guides (English, Odia, Hindi).

### 4.10 Multilingual Coastal Localization Engine
- **`GET /api/i18n/translations`**
  - **Description:** Delivers complete localization dictionaries across the 5 primary coastal languages of the Indian subcontinent: **English (`en`)**, **ଓଡ଼ିଆ / Odia (`or`)**, **हिन्दी / Hindi (`hi`)**, **বাংলা / Bengali (`bn`)**, and **తెలుగు / Telugu (`te`)**.
  - **Features:** Real-time DOM reactive translation of bulletins, advisory levels, interactive tabs, and emergency action directives. Coupled with `SpeechSynthesisUtterance` for local-language audible sirens.

---

## 5. Core Services & Logic Breakdown

### 5.1 Cyclone ML Inference & Replay Service (`backend/ml_service.py`)
- **`CycloneMLService.__init__()`**: Initializes the ML pipeline by scanning for trained deep learning checkpoints in `ml_model_repo/outputs/checkpoints`.
- **`run_inference_cycle(storm_key)`**: Ingests multi-spectral brightness temperatures, computes eye coordinates, estimates central pressure deficit, categorizes the system according to IMD classification criteria, and computes 72h future waypoints.
- **`create_imd_advisory_payload(...)`**: Generates the complete official meteorological payload including the 3-tier wind hazard swaths:
  - Gale winds ($\ge 34\text{ KT} / 62\text{ km/h}$)
  - Storm winds ($50\text{ to }63\text{ KT} / 90-117\text{ km/h}$)
  - Hurricane winds ($\ge 64\text{ KT} / 120\text{ km/h}$)
- **`step_replay()`**: Simulates a live operational pass by advancing the storm along its track, recalculating physics properties, and persisting the resulting advisory into PostgreSQL.

### 5.2 Staged Prediction Funnel Service (`backend/prediction_funnel.py`)
- **`PredictionFunnelService.get_all_stages()`**: Returns the full dictionary containing all 4 early-warning horizons along with physical atmospheric parameters (SST anomalies, Madden-Julian Oscillation phase, Genesis Potential Index, vertical wind shear, Dvorak T-numbers).
- **`get_stage_by_key(stage_key)`**: Provides deep physical and geospatial data (polygons, circle radii, heading lines) for the selected horizon.
- **`comparison_matrix`**: Compares each stage against legacy binary classifiers across False Alarm Ratio, Lead Time, Resource Mobilization Efficiency, and Evacuation Window.

### 5.3 Authentication & Session Security (`backend/auth.py`)
- **`create_access_token(data, expires_delta)`**: Encodes JWT access tokens containing user identifiers, role scopes, and expiration timestamps.
- **`verify_password(plain, hashed)`** & **`get_password_hash(password)`**: Secures authority passwords using bcrypt hashing.
- **`get_current_citizen(credentials)`** & **`get_current_authority(credentials)`**: FastAPI security dependencies that validate bearer tokens and extract role claims before handing execution to endpoints.

### 5.4 Database Connection Pool & RLS Manager (`backend/database/db.py`)
- **`get_db_connection()`**: Manages a resilient connection pool to PostgreSQL.
- **`get_db_cursor(context_dict)`**: Context manager that opens a transaction and immediately executes PostgreSQL session variable injection:
  ```sql
  SET LOCAL app.current_citizen_id = '...';
  SET LOCAL app.current_authority_id = '...';
  SET LOCAL app.current_role = '...';
  SET LOCAL app.current_district_scope = '...';
  SET LOCAL app.is_auth_service = '...';
  ```
  This ensures that every query running inside the transaction is strictly evaluated by PostgreSQL's native Row-Level Security engine.

### 5.5 Database Schema & Row-Level Security (`backend/database/schema.sql`)
- **Tables:**
  - `citizens`: Aadhaar token, name, district, masked mobile, risk zone.
  - `authorities`: User ID, password hash, role (`imd_forecaster`, `district_admin`, `ngo`), district scope.
  - `storms`: Storm ID, name, status (`active` / `dissipated`).
  - `advisories`: Advisory UUID, storm FK, created timestamp, JSONB payload.
  - `sos_requests`: Request UUID, citizen FK, district, coordinates, status (`pending`, `in_progress`, `resolved`), handler FK.
  - `shelters`: Shelter UUID, name, district, coordinates, total capacity, current occupancy, amenities list.
- **Non-Superuser Execution:** Enforced under application user `cyclone_app` to prevent superuser RLS bypass.

---

## 6. Frontend UI Architecture & Function Documentation

### 6.1 Executive Disaster Command Layout (`frontend/index.html`)
The frontend is constructed using semantic HTML5 without bloated JavaScript frameworks, designed to Government of India (Navy `#0B2F5E`, tricolor ribbon, flat 1px borders) specifications:
- **Tricolor Accent Ribbon & Accessibility Bar:** National flag stripes with screen contrast and language selectors.
- **Command Telemetry Ribbon (`.command-telemetry-bar`):** A consolidated 6-metric critical overview banner placed directly below the alert status:
  - `Active System`: Cyclone Dana (VSCS)
  - `Eye Position`: 18.42°N, 86.85°E
  - `Landfall Window`: 12–14 Hours (Puri-Dhamra)
  - `Sustained Wind`: 120 km/h (Gusts: 145)
  - `Storm Surge`: 2.0 – 3.2 m
  - `Shelters Ready`: 8 Active (3.6k Beds Free)
- **Horizontal Operational Horizon Stepper (`.horizon-stepper`):** A compact interactive timeline stepper displaying the 4 early-warning stages with active badges and emergency action directives.
- **Map Control Toolbar (`.map-controls-toolbar`):** Flat segmented controls for switching visualization layers: `Forecast Cone` | `Wind Swaths` | `Combined`, plus `Center Eye` and `Basemap Toggle`.
- **Dual-Column Command Grid:**
  - Left: Interactive Leaflet GIS Map + Coastal District Vulnerability Matrix table.
  - Right: Synoptic Live Telemetry Card + Official IMD Meteorological Bulletin box.

### 6.2 Client Application Controller (`frontend/app.js`)
All client interactions are handled via native asynchronous JavaScript:

| Function Name | Description & Parameters |
| :--- | :--- |
| `initMap()` | Initializes Leaflet map over the Bay of Bengal ($18.80^\circ\text{N}, 85.50^\circ\text{E}$), mounts IMD Cartographic Paper Chart and NASA GIBS satellite basemaps. |
| `selectFunnelStage(stageKey)` | Dynamically switches active funnel stage (`t18_regional_risk`, `t14_cyclogenesis`, `t7_system_id`, `t3_track_intensity`), renders stage-specific geometries (e.g., basin thermal anomaly polygons, nascent vortex centers), and updates telemetry readouts. |
| `togglePlayFunnel()` | Starts/stops automated progression looping through the 4 early-warning horizons with a 2.8s interval. |
| `setMapVisualizationMode(mode)` | Controls rendering between `'cone'` (72h uncertainty envelope), `'swaths'` (IMD 3-tier wind radii capsules), and `'both'`. |
| `toggleBasemapStyle()` | Toggles between IMD Nautical Topographic Chart and NASA Earthdata GIBS satellite imagery. |
| `centerStormMap()` | Centers the GIS camera directly over the active cyclone's eye coordinates. |
| `fetchActiveStorm()` | Calls `GET /api/storms/active`, updates DOM telemetry cells, binds Leaflet track markers, and triggers shelter pin loading. |
| `renderStormOnMap(advisory)` | Draws the past observed track (black polyline), synoptic eye marker, 72h forecast cone, and IMD synoptic track tooltip labels (`DD/HH, KT, CODE`). |
| `createTrackCapsule(pts, rS, rE)` | Geometrical algorithm that calculates normal vectors and cap arcs to generate smooth polygon capsules around storm trajectory points for wind swaths. |
| `switchView(viewName)` | Switches active application tab (`landing`, `citizen`, `authority`, `shelters`, `assistant`, `login`) with automatic authorization redirects. |
| `performCitizenLogin()` | Authenticates citizen via Aadhaar input and saves session state in `localStorage`. |
| `performAuthorityLogin()` | Authenticates authority credentials and opens the district-filtered rescue operations queue. |
| `triggerCitizenSOS()` | Reads device GPS coordinates via HTML5 Geolocation and issues emergency broadcast via `POST /api/sos`. |
| `loadAuthorityQueue()` | Fetches active SOS requests under authority's district RLS and renders the live triage table. |
| `patchSOSStatus(sosId, status)` | Updates emergency incident status (`in_progress` / `resolved`) from the authority command desk. |
| `getUserCurrentLocation()` | Queries high-accuracy device GPS position via HTML5 Geolocation API (`navigator.geolocation.getCurrentPosition`) with coastal high-risk coordinate fallback. |
| `openNearestShelterInGoogleMaps()` | Acquires live user GPS coordinates, resolves closest available shelter, and launches Google Maps turn-by-turn navigation in a new tab while rendering the surge-safe route on the GIS map. |
| `routeToShelterWithGoogleMaps(shelterId, lat, lon, name)` | Directly opens Google Maps directions to a specific shelter from user's live coordinates. |
| `findAndRenderSaferRoute(shelterId)` | Calculates surge-safe inland evacuation path away from coastal danger zones, renders animated emerald corridor on Leaflet, and opens turn-by-turn guidance panel with walking/driving ETAs, bed availability, and Google Maps direct link. |
| `closeEvacuationRoute()` | Cleans up evacuation polylines and waypoints from map and collapses guidance card. |
| `getOfflinePrecomputedRoute(shelterId)` | Offline fallback route generator returning elevation-safe inland corridors when offline. |
| `initOfflineEngine()` | Pre-caches complete emergency disaster pack in `localStorage` and binds online/offline synchronization handlers. |
| `toggleOfflineMode(forceState)` | Simulates offline emergency mode, drops `#offline-status-banner`, and triggers local-only routing. |
| `downloadOfflineEmergencyPack()` | Generates and initiates browser download of standalone JSON emergency pack file. |
| `syncOfflineSOSQueue()` | Automatically syncs queued offline distress broadcasts to DEOC when connectivity restores. |
| `sendCompact2GSMS()` | Generates standard 140-char GSM SMS URI (`sms:112?body=...`) for zero-internet emergency dispatch. |
| `copy2GSMS()` | Copies compact SMS template text to clipboard. |
| `toggleAudibleDistressBeacon()` | Web Audio API piercing 960Hz / 650Hz acoustic siren for search & rescue locating. |
| `setLanguage(lang)` | Switches real-time localization across 5 coastal languages (English, Odia, Hindi, Bengali, Telugu). |
| `speakEmergencyAdvisory()` | Uses browser voice speech synthesis (`SpeechSynthesisUtterance`) to voice the active advisory siren in the chosen language. |

### 6.3 Government of India Design System (`frontend/style.css`)
- **Color Palette:** Primary Navy (`#0B2F5E`), Saffron (`#FF9933`), White (`#FFFFFF`), Green (`#138808`), Alert Red (`#DC2626`).
- **Typography:** Modern clean system sans-serif hierarchy, monospaced synoptic bulletin display.
- **Component Design:** Flat 1px borders (`#D1D5DB`), crisp square-cornered government buttons (`border-radius: 0px` or `2px`), zero unstyled DOM elements, and responsive CSS grid breakpoints (1440px / 1024px / 768px / 640px).

---

## 7. PostgreSQL Row-Level Security (RLS) Defense In Depth

Multi-tenant emergency dispatch systems face severe security risks if district boundaries are only enforced at the application code layer. Cyclone Shield AI enforces security directly in the **PostgreSQL 18 database engine**:

```sql
-- 1. Enable & Force RLS on SOS Requests
ALTER TABLE sos_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE sos_requests FORCE ROW LEVEL SECURITY;

-- 2. Citizens can only select and insert their own requests
CREATE POLICY sos_citizen_select ON sos_requests
  FOR SELECT USING (citizen_id::text = current_setting('app.current_citizen_id', true));

CREATE POLICY sos_citizen_insert ON sos_requests
  FOR INSERT WITH CHECK (citizen_id::text = current_setting('app.current_citizen_id', true));

-- 3. District Administrators can ONLY query SOS requests within their assigned district
CREATE POLICY sos_authority_select ON sos_requests
  FOR SELECT USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

-- 4. Status updates are restricted to the authority's own district
CREATE POLICY sos_authority_update ON sos_requests
  FOR UPDATE USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );
```

### Security Guarantees:
- **Zero Data Leakage:** A District Admin logged into Puri cannot see or modify distress calls originating from Balasore or Kendrapara.
- **SQL Injection Immune:** Even if an attacker attempts raw SQL injection via the query parameter, the PostgreSQL RLS policy filters records before returning rows to the connection cursor.
- **Nationwide Forecaster Support:** National roles (`imd_forecaster`) operate with `district_scope = NULL`, allowing nationwide situational awareness without weakening district barriers.

---

## 8. Automated Test Suite & Verification Matrix

The codebase includes two comprehensive test suites validating all API contracts and database security invariants:

### 8.1 API Integration Tests (`tests/test_api_endpoints.py`)
Run command:
```powershell
python tests/test_api_endpoints.py
```
- [x] **Test 1:** Citizen Login (Aadhaar authentication & token issuance)
- [x] **Test 2:** Citizen Profile (RLS-protected personal data retrieval)
- [x] **Test 3:** Active Storm Advisory (Public synoptic feed for Cyclone Dana)
- [x] **Test 4:** Authority Login (Validates `admin_puri` and `forecaster_hq`)
- [x] **Test 5:** Citizen SOS Submission (Creates new emergency broadcast)
- [x] **Test 6:** Citizen Get My SOS (Citizen views personal distress history)
- [x] **Test 7:** Authority SOS Queue Isolation (Puri Admin sees only Puri requests; zero cross-district leaks)
- [x] **Test 8:** Authority Patch SOS Status (Updates status to `in_progress`)
- [x] **Test 9:** Shelters API (Validates live capacity and bed availability)
- [x] **Test 10:** Scoped AI Emergency Assistant (Validates shelter and safety guidance)
- [x] **Test 11:** ML Pipeline Replay (Advances synoptic observation step)
- [x] **Test 12:** OASIS CAP v1.2 Alert Generator (Verifies compliant XML/JSON payload)
- [x] **Test 13:** AI Architecture & Explainability Diagnostics (6-layer pipeline verification)
- [x] **Test 14:** Multi-Stakeholder Intelligence Metrics (4-sector readiness stats)
- [x] **Test 15:** Odisha Zero Casualty Model Benchmarks (OSDMA historical comparisons)
- [x] **Test 16:** Staged Prediction Funnel API (Verifies all 4 forecasting horizons)
- [x] **Test 17:** Surge-Safe Evacuation Corridor Routing (Computes inland flood-free path with turn-by-turn guidance)
- [x] **Test 18:** Standalone Offline Emergency Pack (Verifies offline JSON bundle with shelters and survival guides)
- [x] **Test 19:** Multilingual Coastal Localization Engine (Validates 5 coastal languages: EN, OR, HI, BN, TE)
- [x] **Test 20:** Multi-Hazard Local Risk Score (LRS) Engine (Computes 3-factor composite score [0-100] and validates district vulnerability matrix)

**Result: 20/20 Integration Tests Passing (100% Success).**

### 8.2 Direct PostgreSQL RLS Tests (`tests/test_rls_direct.py`)
Run command:
```powershell
python tests/test_rls_direct.py
```
- [x] **Test 1:** Citizen Self-Isolation (Citizen A cannot select Citizen B's profile)
- [x] **Test 2:** Citizen SOS Isolation (Citizen A cannot query Citizen B's distress calls)
- [x] **Test 3:** Citizen SOS Insert Spoofing Prevention (Citizen A cannot forge an SOS under Citizen B's ID)
- [x] **Test 4:** District Admin Boundary Isolation (Puri Admin queries 0 Balasore records)
- [x] **Test 5:** District Admin Update Protection (Puri Admin cannot modify Balasore SOS status)
- [x] **Test 6:** National Forecaster Oversight (Forecaster HQ accesses cross-district records)

**Result: 6/6 Direct RLS Security Tests Passing (100% Success).**

---

## 9. Hyper-Localized Multi-Hazard Local Risk Score (LRS) Engine

Instead of generic regional alerts, Cyclone Shield AI calculates a dynamic, point-specific **Local Risk Score ($LRS \in [0, 100]$)** for any coordinate or coastal ward. The engine fuses meteorological predictions with demographic vulnerability across three core pillars:

$$LRS = 0.35 \cdot S_{\text{wind}} + 0.35 \cdot S_{\text{rain}} + 0.30 \cdot S_{\text{pop}}$$

```
                      ┌─────────────────────────────────────────┐
                      │    LOCAL RISK SCORE ENGINE (0 - 100)    │
                      └────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │ (35% Weight)                    │ (35% Weight)                    │ (30% Weight)
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│   WIND HAZARD    │             │  RAIN & SURGE    │             │    POPULATION    │
│   INDEX (Swind)  │             │   INUNDATION     │             │   EXPOSURE &     │
│                  │             │   (Srain)        │             │  VULNERABILITY   │
├──────────────────┤             ├──────────────────┤             ├──────────────────┤
│• Holland Radial  │             │• 24h Projected   │             │• Demographic     │
│  Vortex Decay    │             │  Precipitation   │             │  Density (/km²)  │
│• Distance to Eye │             │• Coastal Storm   │             │• Kutcha/Thatched │
│• Peak Gust Ratio │             │  Surge Height    │             │  Dwelling Ratio  │
│  (1.15x - 1.25x) │             │• Tidal Phase     │             │• Shoreline Band  │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

### Risk Severity Tiers & Operational Directives
| LRS Range | Risk Tier | Hex Color | Operational & Evacuation Directive | Warning Signal |
|:---:|:---:|:---:|:---|:---:|
| **80 – 100** | **Extreme Hazard** | `#DC2626` | **Mandatory Evacuation:** Clear 5 km coastal band immediately to concrete multipurpose shelters. | Great Danger GD-10 |
| **60 – 79.9** | **Severe Hazard** | `#EA580C` | **Targeted Relocation:** Evacuate vulnerable/kutcha dwellings and low-lying flood-prone zones. | Danger Signal D-7 |
| **40 – 59.9** | **Moderate Hazard** | `#F59E0B` | **Coastal Watch & Preparedness:** Secure roofing, verify generator fuel, stock drinking water. | Local Cautionary LC-3 |
| **0 – 39.9** | **Low Advisory** | `#16A34A` | **Advisory Vigil:** Monitor official IMD bulletins; no active evacuation required. | Cautionary C-1 |

### API Endpoints
- `GET /api/risk/local-score?lat={lat}&lon={lon}&district={name}`: Returns point-specific LRS with full factor sub-scores, weights, and operational directives.
- `GET /api/risk/matrix`: Returns sorted multi-hazard risk matrix for all coastal districts (Puri, Jagatsinghpur, Kendrapara, Balasore, Bhadrak, Ganjam).


---

## 10. Installation, Setup & Deployment Guide

### Prerequisites
- **Python 3.10+**
- **PostgreSQL 15+** (configured on port `5433` or custom port via `DB_PORT`)
- Standard Python libraries: `fastapi`, `uvicorn`, `psycopg2-binary`, `pydantic`, `requests`, `bcrypt`, `pyjwt`, `pandas`, `numpy`

### Step 1: Database Initialization
Start PostgreSQL and seed the schema:
```powershell
# Run database schema DDL with RLS policies
psql -h 127.0.0.1 -p 5433 -U postgres -d postgres -f backend/database/schema.sql

# Seed initial operational data (Cyclone Dana, Shelters, Citizens, Authorities)
python backend/database/seed.py
```

### Step 2: Start the FastAPI Application Server
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Access the Platform
- **Public Executive Command Dashboard:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
- **Interactive Swagger API Docs:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **OASIS CAP v1.2 XML Feed:** Open [http://127.0.0.1:8000/api/alerts/cap.xml](http://127.0.0.1:8000/api/alerts/cap.xml).

### Demo Credentials for Verification:
- **Citizen Account:**
  - Aadhaar Number: `1234 1234 1234` (Citizen *Amanullah*, Puri District)
- **District Incident Commander (Puri):**
  - User ID: `admin_puri` | Password: `admin@puri`
- **National Forecaster (IMD HQ):**
  - User ID: `forecaster_hq` | Password: `imd@2026`
- **Disaster Relief NGO Coordinator:**
  - User ID: `ngo_puri` | Password: `ngo@puri`

---
*Built for Smart India Hackathon (SIH 2026) by Team Hexaminds in collaboration with India Meteorological Department (IMD) standards.*
