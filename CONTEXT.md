# Cyclone Shield AI — Master System & Architecture Context Document
### Technical Architecture, Data Flow Diagrams (DFDs), User Flows, Machine Learning Pipeline & Security Specification
**Designed for Review & Audit by AI Agents (Codex, Claude Code, Cursor, GPT-4) & Human Evaluators**  
**Smart India Hackathon (SIH 2026) | Problem Statement ID: 26070 | Team: Hexaminds (Team ID: 36)**  
**Target Beneficiaries:** National Disaster Management Authority (NDMA), India Meteorological Department (IMD), State Disaster Management Authorities (OSDMA, APSDMA), Coastal District Administrations & Indian Coastal Citizens.

---

## 📑 Document Structure & Navigation
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Proposed Solution & Unique Value Propositions (USP)](#2-proposed-solution--unique-value-propositions-usp)
3. [The Staged Prediction Funnel (T-18d to T-3d)](#3-the-staged-prediction-funnel-t-18d-to-t-3d)
4. [Machine Learning & Atmospheric Modeling Architecture](#4-machine-learning--atmospheric-modeling-architecture)
5. [Hyper-Localized Multi-Hazard Local Risk Score (LRS) Engine](#5-hyper-localized-multi-hazard-local-risk-score-lrs-engine)
6. [Data Flow Diagrams (DFDs) — Levels 0, 1, and 2](#6-data-flow-diagrams-dfds--levels-0-1-and-2)
   - [6.1 DFD Level-0: Context-Level Diagram](#61-dfd-level-0-context-level-diagram)
   - [6.2 DFD Level-1: Major Subsystems & Data Stores](#62-dfd-level-1-major-subsystems--data-stores)
   - [6.3 DFD Level-2: Subsystem 1 — Ingestion, Staged Funnel & ML Inference](#63-dfd-level-2-subsystem-1--ingestion-staged-funnel--ml-inference)
   - [6.4 DFD Level-2: Subsystem 2 — Database Engine & RLS Multi-Tenant Security](#64-dfd-level-2-subsystem-2--database-engine--rls-multi-tenant-security)
   - [6.5 DFD Level-2: Subsystem 3 — Citizen Safety, Surge-Safe Routing & Shelter Allocation](#65-dfd-level-2-subsystem-3--citizen-safety-surge-safe-routing--shelter-allocation)
   - [6.6 DFD Level-2: Subsystem 4 — Inter-Agency Dissemination & OASIS CAP v1.2](#66-dfd-level-2-subsystem-4--inter-agency-dissemination--oasis-cap-v12)
7. [Comprehensive End-to-End User Flows & Journeys](#7-comprehensive-end-to-end-user-flows--journeys)
   - [7.1 User Flow 1: Public Citizen Disaster Vigil & GIS Surveillance](#71-user-flow-1-public-citizen-disaster-vigil--gis-surveillance)
   - [7.2 User Flow 2: Citizen Aadhaar Auth, Risk Check, Surge-Safe Evacuation & SOS](#72-user-flow-2-citizen-aadhaar-auth-risk-check-surge-safe-evacuation--sos)
   - [7.3 User Flow 3: District Administrator Incident Triage (RLS Enforced)](#73-user-flow-3-district-administrator-incident-triage-rls-enforced)
   - [7.4 User Flow 4: IMD National Forecaster Synoptic Surveillance & Replay](#74-user-flow-4-imd-national-forecaster-synoptic-surveillance--replay)
   - [7.5 User Flow 5: Offline Emergency Disaster Mode (GSM SMS & USSD)](#75-user-flow-5-offline-emergency-disaster-mode-gsm-sms--ussd)
   - [7.6 User Flow 6: Scoped Emergency AI Assistant with Deterministic Fallback](#76-user-flow-6-scoped-emergency-ai-assistant-with-deterministic-fallback)
8. [Database Schema & PostgreSQL Row-Level Security (RLS)](#8-database-schema--postgresql-row-level-security-rls)
9. [Complete Backend API Reference (20 Endpoints)](#9-complete-backend-api-reference-20-endpoints)
10. [Frontend UI/UX Architecture & Government Design System](#10-frontend-uiux-architecture--government-design-system)
11. [Automated Verification, Security Audit & Test Results](#11-automated-verification-security-audit--test-results)
12. [Runtime Operations & Quick-Start Guide](#12-runtime-operations--quick-start-guide)

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement Details
- **Competition:** Smart India Hackathon (SIH 2026)
- **Problem Statement ID:** 26070
- **Title:** *Identification, Classification and Prediction of Tropical Cyclones using AI/ML*
- **Theme:** Disaster Management
- **PS Category:** Software
- **Team Registered:** Hexaminds (Team ID: 36)

### 1.2 The Core Problem in Tropical Cyclone Early Warning
India features over **7,500 km of coastline**, with ~5,700 km exposed to recurrent severe tropical cyclonic disturbances across the North Indian Ocean basin (Bay of Bengal and Arabian Sea). Over **40% of India's population** resides within 100 km of coastal shorelines.

Traditional software and naive machine learning approaches treat cyclone detection as a simplistic, single-frame binary classification task (*"Is a cyclone present in this satellite tile: Yes/No?"*). This legacy paradigm suffers from three critical flaws:
1. **Zero Operational Lead Time:** Classifiers only fire when a mature, well-organized cyclonic vortex with a distinct eyewall is already within 24–48 hours of coastline, leaving disaster management authorities inadequate time for logistical mobilization.
2. **Catastrophic False Alarm Costs:** High false alarm rates trigger premature, statewide evacuations costing hundreds of crores in exchequer expenditures and eroding citizen trust in warning alerts.
3. **Absence of Actionable Civil Protection:** An isolated label (*"VSCS Detected"*) provides no actionable intelligence on surge-safe inland corridors, real-time shelter bed capacity, hyper-localized district risk, or authenticated multi-agency emergency rescue dispatch.

### 1.3 System Mission
**Cyclone Shield AI** (internally designated *Cyclone Horizon*) bridges satellite atmospheric physics, deep sequence modeling, automated inter-agency alerting, and on-ground citizen safety into a unified, secure, high-performance platform.

---

## 2. Proposed Solution & Unique Value Propositions (USP)

### 2.1 Solution Overview
Cyclone Shield AI provides an **end-to-end national cyclone early warning and emergency response system** that seamlessly connects:
- Automated ingestion of ISRO INSAT-3D/3DR calibrated brightness radiometry and IMD Best Track archives.
- A 4-stage physics-constrained prediction funnel providing up to **18 days of decision runway**.
- A 5-model deep learning inference pipeline delivering 72-hour forecast waypoints, uncertainty envelopes, and 3-tier wind hazard swaths ($\ge 34\text{ KT}, 50\text{–}63\text{ KT}, \ge 64\text{ KT}$).
- Database-level multi-tenant security using **PostgreSQL Row-Level Security (RLS)** ensuring zero cross-district leakage.
- Dynamic **capacity-aware shelter management** with real-time bed count tracking.
- **Surge-Safe Evacuation Corridor Routing** directing citizens away from coastal flood zones onto elevated terrain.
- Standalone **Offline Emergency Disaster Pack** with 140-char 2G GSM SMS dispatch, USSD codes (`*112*1#`), and browser Web Audio acoustic sirens.
- Multilingual localization across the **5 primary coastal languages** (English, Odia, Hindi, Bengali, Telugu).
- Automated dissemination of **OASIS Common Alerting Protocol (CAP v1.2)** XML/JSON feeds to NDMA SACHET and state sirens.

### 2.2 System Unique Selling Points (USPs)

| Feature Pillar | Cyclone Shield AI Implementation | Legacy / Alternative Systems |
| :--- | :--- | :--- |
| **Prediction Horizon** | **4-Tier Staged Funnel:** T-18d Basin Risk &rarr; T-14d Cyclogenesis GPI &rarr; T-7d Nascent Vortex &rarr; T-3d Landfall Cone. | 0–24h single snapshot binary detection. |
| **Data Isolation & Security** | **PostgreSQL Native Row-Level Security (RLS)** enforced by database engine using session context variables. | Application-layer `WHERE district = ...` filters vulnerable to SQL injection or code bugs. |
| **Shelter Management** | **Capacity-Aware Dynamic Allocation:** Real-time bed occupancy, generator fuel status, medical amenity tracking. | Static PDF shelter directory without live occupancy counts. |
| **Evacuation Guidance** | **Surge-Safe Inland Corridors:** Elevation-aware routing bypassing low-lying coastal strips (<2 km from sea) to high ground. | Generic routing directing citizens along coastal highways into storm surges. |
| **Resilience & Offline Ops** | **Zero-Internet Fallback:** Standalone offline pack in `localStorage`, 140-char GSM SMS (`sms:112?body=...`), USSD dial codes, 960Hz acoustic siren. | App fails entirely during cellular tower collapse. |
| **Explainability** | **Grad-CAM & MC-Dropout UQ:** Layer-wise relevance heatmaps on Central Dense Overcast (CDO) + 50-pass epistemic variance bounds. | Opaque black-box outputs with no confidence metrics. |
| **Standards Compliance** | **OASIS CAP v1.2 XML/JSON:** Automated alert feed conforming to NDMA SACHET and ITU standards. | Proprietary custom JSON formats requiring manual conversion. |

---

## 3. The Staged Prediction Funnel (T-18d to T-3d)

Rather than forcing a single model to make impossible long-range landfall predictions, Cyclone Shield AI structures forecasting into four calibrated operational horizons:

```
[ T-18 Days: REGIONAL BASIN RISK ] ──────────────────────────────────────────┐
  • SST Anomaly (+1.4°C, 29.8°C), TCHP > 90 kJ/cm²                            │ Lead Time: 432 Hours
  • MJO Phase 3/4 Convective Enhancement                                      │ Action: Audit shelter supplies
                                                                              │
[ T-14 Days: CYCLOGENESIS PROBABILITY ] ────────────────────────────────────┤
  • Genesis Potential Index (GPI = 8.4)                                       │ Lead Time: 336 Hours
  • 20-Member Dynamical Ensemble (68.5% Consensus)                            │ Action: Port Signal 1, fishing recall
                                                                              │
[ T-7 Days: SYSTEM IDENTIFICATION (BOB-06) ] ───────────────────────────────┤
  • Nascent Vortex Center (14.2°N, 89.5°E), Pressure Drop -6 hPa              │ Lead Time: 168 Hours
  • Dvorak T1.5 Curved Band Organization                                      │ Action: DEOC 24x7 monitoring
                                                                              │
[ T-3 Days: TRACK & INTENSITY FORECAST (72h) ] ─────────────────────────────┘
  • Deep Sequence Physics-Informed GRU + MC-Dropout UQ                        │ Lead Time: 72 Hours
  • 72h Track Cone, 3-Tier Wind Radii (34/50/64 KT), Storm Surge 2.0-3.2m     │ Action: Mandatory 5km evacuation
```

### Horizon Specification Table

| Stage Key | Horizon | Atmospheric & Physical Triggers | Operational Command Actions | Geospatial GIS Layer | Alert Color |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `t18_regional_risk` | **T-18 Days** (432h) | SST: 29.8°C (+1.4°C anomaly), TCHP: 92 kJ/cm², MJO Phase 3/4 active, low-level vorticity $+12.4 \times 10^{-6}\text{ s}^{-1}$. | Pre-season advisory to NDMA/OSDMA/APSDMA. Verify shelter readiness and generator fuel reserves. | Basin-scale thermal anomaly polygon (dashed amber) & hotspot centroid. | Amber (`#D97706`) |
| `t14_cyclogenesis` | **T-14 Days** (336h) | Genesis Potential Index (GPI = 8.4), vertical wind shear < 12 kt, mid-tropospheric RH: 78%, 14/20 ensemble consensus (68.5%). | Hoist Port Cautionary Signal No. 1 at Paradeep & Vizag. Direct deep-sea fishing trawlers to return. | Probabilistic genesis ellipse contours (40%, 60%, 68.5% confidence envelopes). | Gold (`#CA8A04`) |
| `t7_system_id` | **T-7 Days** (168h) | Vortex centroid BOB-06 localized (14.2°N, 89.5°E), Dvorak T1.5 curved band pattern, pressure deficit: -6 hPa, winds: 52 km/h. | Mandatory recall of all coastal fishermen. DEOC activated in 24x7 mode. Multi-purpose shelter structural audit. | Developing vortex center marker (orange) & projected NNW trajectory line. | Orange (`#EA580C`) |
| `t3_track_intensity` | **T-3 Days** (72h) | Very Severe Cyclonic Storm (VSCS Dana), central pressure 982 hPa, sustained winds 120 km/h (gusts 145), landfall in 12–14h. | Hoist Great Danger Signal GD-10. Mandatory evacuation of coastal zones (<5 km). Pre-position 18 NDRF & 24 ODRAF teams. | 72h IMD Uncertainty Cone, 34/50/64 KT wind hazard swaths & track waypoints. | Crimson (`#DC2626`) |

---

## 4. Machine Learning & Atmospheric Modeling Architecture

### 4.1 The 6-Layer Operational Pipeline
The machine learning pipeline is structured into six discrete, modular layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   LAYER 1: SENSOR & DATA INGESTION                     │
│  • ISRO MOSDAC INSAT-3D/3DR Calibrated Radiance (HDF5 IR1 10.8µm)      │
│  • IMD Best Track Dataset (1990–2025 Historical Storm Trajectories)    │
│  • Negative Control GeoTIFFs (Fair weather, non-cyclonic cloud clusters)│
│  • NOAA OISST (Sea Surface Temperature) & GFS Wind Shear Open Data     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              LAYER 2: PREPROCESSING & CALIBRATION                      │
│  • Planck Inverse Radiance to Calibrated Brightness Temp (Kelvin)      │
│  • Nadir Alignment & 0.04° Spatial Resampling Grid                     │
│  • Temporal Sequencing & Stratified Spatio-Temporal Train/Val Splits   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              LAYER 3: 5-MODEL DEEP LEARNING INFERENCE                  │
│  [1. Vortex Detector]      U-Net CNN IR Center Localizer (0.04° res)   │
│  [2. Pattern Classifier]   Dvorak Multi-Task CNN + Grad-CAM Heatmaps   │
│  [3. Intensity Classifier] Hybrid Multi-Head XGBoost + Visual Stacking │
│  [4. Track Predictor]      Physics-Informed Beta-Advection GRU Seq2Seq │
│  [5. RL Correction]        PPO / CQL Offline Policy Bias Correction    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│             LAYER 4: EXPLAINABILITY & UNCERTAINTY (UQ)                 │
│  • Eigen-CAM / Grad-CAM Attention Heatmaps on CDO & Primary Eyewall    │
│  • Monte Carlo Dropout (N=50 passes) Epistemic Uncertainty Envelopes   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              LAYER 5: FASTAPI SERVICE & DATA PERSISTENCE               │
│  • Asynchronous Background Worker executing inference loop             │
│  • PostgreSQL Row-Level Security (RLS) Multi-Tenant Data Isolation     │
│  • Multi-Hazard Local Risk Score (LRS) & OASIS CAP v1.2 XML/JSON feeds │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               LAYER 6: OPERATIONAL CIVIL PROTECTION                    │
│  • Dual-Basemap Leaflet GIS (IMD Nautical Paper Chart + NASA GIBS)     │
│  • Capacity-Aware Shelters, Surge-Safe Inland Corridors & SOS Dispatch │
│  • Standalone Offline Emergency Pack (2G GSM SMS / USSD / Audio Beacon)│
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Sourcing Integrity
- **Model Input:** Ingests **calibrated physical brightness temperatures ($T_b$ in Kelvin)** from ISRO MOSDAC INSAT-3DR/3DS radiometry.
- **Strict Distinction:** NASA GIBS tiles are RGB color-mapped display composites and are **never** used as neural network model inputs. NASA GIBS is strictly integrated as an optional visual basemap layer in the frontend Leaflet GIS viewer.

---

## 5. Hyper-Localized Multi-Hazard Local Risk Score (LRS) Engine

### 5.1 Mathematical Formulation
Cyclone Shield AI computes a point-specific **Local Risk Score ($LRS \in [0, 100]$)** for any coordinate or coastal ward, fusing atmospheric dynamics with demographic vulnerability:

$$LRS = 0.35 \cdot S_{\text{wind}} + 0.35 \cdot S_{\text{rain}} + 0.30 \cdot S_{\text{pop}}$$

Where:
1. **$S_{\text{wind}}$ (Wind Hazard Index, 35% weight):**
   Computed using a modified Holland/Rankine vortex radial decay from the cyclone eye:
   $$V(r) = V_{\max} \cdot \left(\frac{R_{\max}}{r}\right)^{0.58} \quad \text{for } r > R_{\max}$$
   $$S_{\text{wind}} = \min\left(100, \frac{V_{\text{local}}}{145} \times 100\right)$$
2. **$S_{\text{rain}}$ (Rainfall & Storm Surge Inundation Index, 35% weight):**
   Compounds 24-hour accumulated rainfall bands with projected astronomical tidal surge heights:
   $$S_{\text{rain}} = \min\left(100, \left(\frac{R_{24\text{h}}}{300} \times 60\right) + \left(\frac{\text{Surge}_{m}}{3.5} \times 40\right)\right)$$
3. **$S_{\text{pop}}$ (Demographic Vulnerability & Exposure Index, 30% weight):**
   Integrates coastal proximity ($D_{\text{coast}}$), population density ($\rho$), and the ratio of fragile kutcha/thatched dwellings ($K_{\text{dwell}}$):
   $$S_{\text{pop}} = \min\left(100, \left(\frac{\rho}{700} \times 35\right) + \max(5, 35 - 4.5 \cdot D_{\text{coast}}) + \left(\frac{K_{\text{dwell}}}{50} \times 30\right)\right)$$

### 5.2 Severity Tier & Evacuation Directives

| LRS Range | Risk Tier | Hex Color | Evacuation Directive | Port Warning Signal |
|:---:|:---:|:---:|:---|:---:|
| **80.0 – 100** | **EXTREME RISK** | `#DC2626` | **Mandatory Immediate Evacuation:** Clear 5 km coastal band immediately to concrete multipurpose shelters. | Great Danger GD-10 |
| **60.0 – 79.9** | **HIGH RISK** | `#EA580C` | **Targeted Relocation:** Relocate vulnerable kutcha dwellers and low-lying coastal wards. | Danger Signal D-8 |
| **40.0 – 59.9** | **MODERATE RISK** | `#F59E0B` | **Coastal Watch & Preparedness:** Secure loose structures, verify generator fuel and drinking water. | Local Cautionary LC-3 |
| **0.0 – 39.9** | **LOW RISK** | `#10B981` | **Advisory Vigil:** Normal tracking surveillance with marine fishermen warnings active. | Information Bulletin |

---

## 6. Data Flow Diagrams (DFDs) — Levels 0, 1, and 2

### 6.1 DFD Level-0: Context-Level Diagram
The Level-0 Context DFD captures the entire Cyclone Shield AI system as a single process interacting with its external environmental entities:

```mermaid
graph TD
    classDef ext fill:#0B2F5E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF;
    classDef sys fill:#F4F5F7,stroke:#0B2F5E,stroke-width:3px,color:#0B2F5E;

    MOSDAC["🛰️ ISRO MOSDAC / IMD Data Feeds"]:::ext
    CITIZEN["👤 Coastal Citizen"]:::ext
    DIST_ADMIN["🏛️ District Admin (DEOC Puri)"]:::ext
    NAT_FORECASTER["🌐 IMD HQ / National Forecaster"]:::ext
    NDMA_SACHET["📡 NDMA SACHET / Alert Gateways"]:::ext
    SHELTER_OPS["🏠 Cyclone Shelter Managers"]:::ext

    SYS(("[0.0] CYCLONE SHIELD AI PLATFORM")):::sys

    MOSDAC -->|INSAT-3DR IR1 HDF5 & Best Track Coordinates| SYS
    CITIZEN -->|Aadhaar Auth, GPS Coordinates, SOS Distress Call| SYS
    SYS -->|Active Synoptic Feed, Surge-Safe Route, LRS, SMS Template| CITIZEN
    
    DIST_ADMIN -->|Credential Auth, Incident Status Updates, Shelter Logs| SYS
    SYS -->|RLS-Scoped SOS Queue, District Vulnerability Matrix| DIST_ADMIN

    NAT_FORECASTER -->|Inference Replay Triggers, Meteorological Override| SYS
    SYS -->|Nationwide Synoptic Intelligence, 6-Layer Architecture Diagnostics| NAT_FORECASTER

    SYS -->|OASIS CAP v1.2 XML / JSON Official Feeds| NDMA_SACHET
    SHELTER_OPS -->|Real-Time Bed Occupancy & Amenity Status| SYS
    SYS -->|Evacuee Allocation Recommendations| SHELTER_OPS
```

---

### 6.2 DFD Level-1: Major Subsystems & Data Stores
The Level-1 DFD decomposes Cyclone Shield AI into its core operational processes and persistent database stores:

```mermaid
graph TD
    classDef proc fill:#FFFFFF,stroke:#0B2F5E,stroke-width:2px,color:#0B2F5E;
    classDef store fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A;
    classDef ext fill:#0B2F5E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF;

    EXT_SATELLITE["ISRO MOSDAC / NOAA OISST / GFS"]:::ext
    EXT_CITIZEN["Citizen Device / GPS"]:::ext
    EXT_AUTHORITY["Authority Terminal"]:::ext
    EXT_NDMA["NDMA SACHET Gateway"]:::ext

    P1["1.0 Ingest & Preprocess Satellite Imagery"]:::proc
    P2["2.0 Execute 5-Model Cyclone Inference Pipeline"]:::proc
    P3["3.0 Compute Local Risk Scores & Advisories"]:::proc
    P4["4.0 Authenticate & Set Session RLS Variables"]:::proc
    P5["5.0 Manage Emergency SOS Triage"]:::proc
    P6["6.0 Allocate Capacity-Aware Shelters & Routes"]:::proc
    P7["7.0 Generate OASIS CAP v1.2 XML Alerts"]:::proc

    D_STORMS[("D1: storms")]:::store
    D_ADVISORIES[("D2: advisories (JSONB)")]:::store
    D_CITIZENS[("D3: citizens (RLS)")]:::store
    D_AUTHORITIES[("D4: authorities")]:::store
    D_SOS[("D5: sos_requests (RLS)")]:::store
    D_SHELTERS[("D6: shelters")]:::store

    EXT_SATELLITE -->|Raw Radiance & Atmospheric Grids| P1
    P1 -->|Standardized Kelvin Tb 0.04° Array| P2
    P2 -->|Track Waypoints, Category, Wind Radii| P3
    P3 -->|Write Active Synoptic Record| D_ADVISORIES
    P3 -->|Update Storm Status| D_STORMS

    EXT_CITIZEN -->|Aadhaar Number| P4
    EXT_AUTHORITY -->|User ID & Password| P4
    P4 -->|Verify Citizen Record| D_CITIZENS
    P4 -->|Verify Authority Hash| D_AUTHORITIES
    P4 -->|Issue Scoped JWT Token| EXT_CITIZEN
    P4 -->|Issue Scoped JWT Token| EXT_AUTHORITY

    EXT_CITIZEN -->|POST SOS with GPS Lat/Lon| P5
    P5 -->|Insert Record with citizen_id| D_SOS
    D_SOS -->|RLS-Scoped Queue Query| P5
    P5 -->|Display Queue to Authorized Admin| EXT_AUTHORITY
    EXT_AUTHORITY -->|PATCH Status to in_progress / resolved| P5
    P5 -->|Update Record| D_SOS

    EXT_CITIZEN -->|Request Evacuation Corridor| P6
    D_SHELTERS -->|Live Available Bed Counts| P6
    P6 -->|Surge-Safe Turn-by-Turn Route + Google Maps Link| EXT_CITIZEN

    D_ADVISORIES -->|Read Latest Advisory Payload| P7
    P7 -->|Publish Official CAP v1.2 XML / JSON| EXT_NDMA
```

---

### 6.3 DFD Level-2: Subsystem 1 — Ingestion, Staged Funnel & ML Inference
Details the internal steps of satellite radiance calibration, Dvorak analysis, ensemble tracking, and uncertainty quantification:

```mermaid
graph TD
    classDef proc fill:#FFFFFF,stroke:#0B2F5E,stroke-width:2px,color:#0B2F5E;
    classDef store fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A;

    IN_RAD["MOSDAC INSAT-3DR HDF5 IR1 Radiance"] --> P2_1["2.1 Planck Inversion & Brightness Temp Calibration (Kelvin)"]:::proc
    IN_BEST["IMD Best Track History (1990–2025)"] --> P2_2["2.2 Spatial Resampling & Center Extraction (0.04° Grid)"]:::proc
    P2_1 --> P2_2
    
    P2_2 --> P2_3["2.3 U-Net CNN Eye Localization & Vortex Detection"]:::proc
    P2_3 -->|Eye Centroid Coordinates| P2_4["2.4 Dvorak Pattern CNN + Grad-CAM Heatmap Extraction"]:::proc
    
    IN_MET["NOAA OISST & GFS Deep Vertical Wind Shear"] --> P2_5["2.5 Hybrid XGBoost + Multi-Head Intensity Classifier"]:::proc
    P2_4 --> P2_5
    
    P2_5 -->|Estimated Wind Knots & Pressure Deficit| P2_6["2.6 Physics-Informed GRU Seq2Seq + MC-Dropout UQ (N=50)"]:::proc
    
    P2_6 -->|72h Trajectory Waypoints + Uncertainty Bounds| P2_7["2.7 PPO / CQL Offline RL Policy Bias Correction"]:::proc
    
    P2_7 --> P2_8["2.8 Construct Synoptic Advisory Payload & 3-Tier Wind Radii"]:::proc
    P2_8 --> STORE_ADV[("D2: advisories table")]:::store
```

---

### 6.4 DFD Level-2: Subsystem 2 — Database Engine & RLS Multi-Tenant Security
Illustrates the transaction boundary, context injection, and row filtering within PostgreSQL 18:

```mermaid
sequenceDiagram
    autonumber
    actor Client as Citizen / Authority HTTP Client
    participant API as FastAPI Backend Engine
    participant Pool as psycopg2 ThreadedConnectionPool
    participant Conn as PostgreSQL Dedicated Connection (cyclone_app)
    participant RLS as PostgreSQL Row-Level Security Engine
    participant Table as Database Storage Engine (sos_requests)

    Client->>API: HTTP Request + Bearer JWT Token
    API->>API: Validate JWT Signature & Extract Claims (sub, role, district_scope)
    API->>Pool: Borrow Connection from Pool
    Pool-->>API: Active Connection (conn.autocommit = False)
    
    rect rgb(238, 242, 255)
    Note over API,Conn: Transaction Boundary & Session Variable Injection
    API->>Conn: SELECT set_config('app.current_citizen_id', $sub, true);
    API->>Conn: SELECT set_config('app.current_role', $role, true);
    API->>Conn: SELECT set_config('app.current_district_scope', $district_scope, true);
    end

    API->>Conn: Execute SQL Query (e.g. SELECT * FROM sos_requests;)
    Conn->>RLS: Evaluate Table Security Policies Against Local Session Settings
    
    alt Citizen Access
        RLS->>Table: Filter: citizen_id::text = current_setting('app.current_citizen_id')
    else District Admin (Puri)
        RLS->>Table: Filter: district = current_setting('app.current_district_scope')
    else National Forecaster (HQ)
        RLS->>Table: District Scope is NULL -> Permit All Records
    end

    Table-->>Conn: Return Permitted Rows Only (0 Leaked Records)
    Conn-->>API: Query Results
    API->>Conn: conn.commit()
    API->>Pool: Release Connection Back to Pool
    API-->>Client: JSON Response (Structurally Immune to Leaks)
```

---

### 6.5 DFD Level-2: Subsystem 3 — Citizen Safety, Surge-Safe Routing & Shelter Allocation
Details the routing calculation, coastal surge avoidance, and capacity reservation:

```mermaid
graph TD
    classDef proc fill:#FFFFFF,stroke:#0B2F5E,stroke-width:2px,color:#0B2F5E;
    classDef store fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A;

    CIT_GPS["Citizen Device Coordinates (lat, lon)"] --> P6_1["6.1 Haversine Distance Search to Operational Shelters"]:::proc
    D_SHELTER[("D6: shelters table")] --> P6_1
    
    P6_1 --> P6_2["6.2 Capacity & Inundation Filter (Available Beds > 50)"]:::proc
    
    P6_2 -->|Closest Capacity-Ready Shelter Identified| P6_3["6.3 Inland Offset Calculator (Bypass <2km Shoreline Marine Drive)"]:::proc
    
    P6_3 --> P6_4["6.4 Generate Turn-by-Turn Waypoints via High-Ground Bada Danda"]:::proc
    P6_4 --> P6_5["6.5 Calculate Walking ETA (12 min/km) & Evac Bus ETA (2.8 min/km)"]:::proc
    P6_5 --> P6_6["6.6 Generate Deep-Linked Google Maps Navigation URL"]:::proc
    
    P6_6 --> OUT_GUIDE["Surge-Safe Evacuation Guidance Card & Animated Map Polyline"]
```

---

### 6.6 DFD Level-2: Subsystem 4 — Inter-Agency Dissemination & OASIS CAP v1.2
Covers automated conversion of synoptic outputs to OASIS CAP XML/JSON:

```mermaid
graph TD
    classDef proc fill:#FFFFFF,stroke:#0B2F5E,stroke-width:2px,color:#0B2F5E;
    classDef store fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A;

    D_ADV[("D2: advisories table")] --> P7_1["7.1 Extract Latest Synoptic Warning & Wind Radii"]:::proc
    P7_1 --> P7_2["7.2 Format CAP v1.2 Alert Header (Sender, Urgency, Severity)"]:::proc
    P7_2 --> P7_3["7.3 Geocode Coastal Wards & Target Impact Polygon"]:::proc
    P7_3 --> P7_4["7.4 Populate Multilingual Instruction Directives"]:::proc
    
    P7_4 --> P7_5["7.5 Serialize into OASIS CAP v1.2 XML Feed (/api/alerts/cap.xml)"]:::proc
    P7_4 --> P7_6["7.6 Serialize into JSON Feed (/api/alerts/cap)"]:::proc
    
    P7_5 --> OUT_SACHET["NDMA SACHET National Cell Broadcast"]
    P7_6 --> OUT_DEOC["State / District Emergency Control Desks"]
```

---

## 7. Comprehensive End-to-End User Flows & Journeys

### 7.1 User Flow 1: Public Citizen Disaster Vigil & GIS Surveillance

```mermaid
sequenceDiagram
    autonumber
    actor PublicUser as Coastal Citizen / Public
    participant Browser as Web Browser (app.js)
    participant API as FastAPI Server
    participant DB as PostgreSQL Database

    PublicUser->>Browser: Opens https://cycloneshield.gov.in / http://127.0.0.1:8000
    Browser->>API: GET /api/storms/active (Public, No Auth)
    API->>DB: Query Latest Advisory Record from 'advisories'
    DB-->>API: JSONB Advisory Payload (Cyclone Dana, VSCS)
    API-->>Browser: HTTP 200 OK + Full Synoptic Data
    Browser->>Browser: Update Header Telemetry Strip (Eye, Landfall ETA, Surge, Beds)
    Browser->>Browser: Render Leaflet GIS Map: Observed Track + 72h Cone + Wind Swaths
    
    PublicUser->>Browser: Clicks Funnel Stepper: 'STAGE 1 (T-18d)'
    Browser->>API: GET /api/ml/prediction-funnel/stage/t18_regional_risk
    API-->>Browser: Spatial Geometry + SST Anomaly (+1.4°C) + MJO Phase 3/4
    Browser->>Browser: Re-render GIS Map: Draw Basin Thermal Polygon & Update Readout
    
    PublicUser->>Browser: Clicks 'Basemap Toggle'
    Browser->>Browser: Switch from IMD Nautical Chart to NASA GIBS Satellite Tiles
```

---

### 7.2 User Flow 2: Citizen Aadhaar Auth, Risk Check, Surge-Safe Evacuation & SOS

```mermaid
sequenceDiagram
    autonumber
    actor Citizen as Citizen (Amanullah)
    participant Browser as Frontend Client
    participant API as FastAPI Backend
    participant DB as PostgreSQL Database (RLS)

    Citizen->>Browser: Enters 12-digit Aadhaar Number (1234 1234 1234)
    Browser->>API: POST /api/auth/citizen {"aadhaar_number": "123412341234"}
    API->>DB: Auth Lookup with app.is_auth_service = 'true'
    DB-->>API: Citizen Record (Name: Amanullah, District: Puri, Risk: High)
    API-->>Browser: Scoped JWT Token + Citizen Profile
    Browser->>Browser: Save Token in localStorage & Switch to Citizen Portal
    
    Browser->>API: GET /api/risk/local-score?district=Puri
    API-->>Browser: Local Risk Score: 56.7/100 (Moderate Risk, GD-10 Signal)
    Browser->>Browser: Display Point-Specific Risk Breakdown
    
    Citizen->>Browser: Clicks 'Find Safer Route to Shelter'
    Browser->>API: GET /api/shelters/evacuation-route?origin_lat=19.805&origin_lon=85.830&district=Puri
    API->>DB: Query Shelters with available_beds > 50 in Puri
    DB-->>API: Puri Zilla School Shelter (720 Beds Available)
    API-->>Browser: Surge-Safe Turn-by-Turn Waypoints + Walking ETA (21 min) + Google Maps URL
    Browser->>Browser: Draw Animated Green Evacuation Corridor on Map & Open Guidance Card
    
    opt Emergency Distress Trigger
        Citizen->>Browser: Clicks 'BROADCAST EMERGENCY SOS NOW'
        Browser->>Browser: Obtain Device GPS via navigator.geolocation (19.815°N, 85.832°E)
        Browser->>API: POST /api/sos (Authorization: Bearer <CitizenToken>)
        API->>DB: SET LOCAL app.current_citizen_id = '<citizen_uuid>'
        API->>DB: INSERT INTO sos_requests (citizen_id, district, lat, lon, status='pending')
        DB-->>API: Insert Confirmed (Structurally Bound to Citizen)
        API-->>Browser: HTTP 200 OK: 'Emergency SOS Broadcasted'
        Browser->>Browser: Trigger Visual Flashing Siren & Display Dispatch ID
    end
```

---

### 7.3 User Flow 3: District Administrator Incident Triage (RLS Enforced)

```mermaid
sequenceDiagram
    autonumber
    actor Admin as District Admin (Puri DEOC)
    actor Attacker as Malicious Script / Hacker
    participant API as FastAPI Server
    participant DB as PostgreSQL Engine (RLS)

    Admin->>API: POST /api/auth/authority {"user_id": "admin_puri", "password": "..."}
    API->>DB: Verify bcrypt password hash
    DB-->>API: Authority Record (role='district_admin', district_scope='Puri')
    API-->>Admin: Scoped JWT Token (role claim: district_admin, scope: Puri)
    
    Admin->>API: GET /api/sos/queue (Authorization: Bearer <AdminToken>)
    API->>DB: SET LOCAL app.current_role = 'district_admin';
    API->>DB: SET LOCAL app.current_district_scope = 'Puri';
    API->>DB: SELECT * FROM sos_requests;
    DB->>DB: Evaluate RLS: district = 'Puri'
    DB-->>API: 21 Puri Distress Calls Returned (0 from Balasore / Kendrapara)
    API-->>Admin: Render Live Triage Table
    
    Admin->>API: PATCH /api/sos/{id} {"status": "in_progress"}
    API->>DB: UPDATE sos_requests SET status = 'in_progress', handled_by = ... WHERE id = ...
    DB->>DB: Evaluate RLS Update Scope: Must be within Puri
    DB-->>API: Update Committed
    API-->>Admin: Incident Marked 'In Progress', Rescue Unit Dispatched
    
    opt Cross-District Security Attack Attempt
        Attacker->>API: PATCH /api/sos/{balasore_sos_id} {"status": "resolved"} (Using Puri Admin Token)
        API->>DB: UPDATE sos_requests WHERE id = '{balasore_id}' with scope 'Puri'
        DB->>DB: RLS Engine Evaluates Policy -> 0 Rows Matched (Permission Denied)
        DB-->>API: Rows Affected: 0
        API-->>Attacker: HTTP 404 / 403 Forbidden (Tamper Proof)
    end
```

---

### 7.4 User Flow 4: IMD National Forecaster Synoptic Surveillance & Replay

```mermaid
sequenceDiagram
    autonumber
    actor Forecaster as IMD National Forecaster (HQ)
    participant UI as Authority Portal
    participant API as FastAPI Server
    participant ML as Cyclone ML Inference Service
    participant DB as PostgreSQL Database

    Forecaster->>API: POST /api/auth/authority {"user_id": "forecaster_hq", "password": "..."}
    API-->>Forecaster: JWT Token (role='imd_forecaster', district_scope=NULL)
    
    Forecaster->>API: GET /api/sos/queue
    API->>DB: SET LOCAL app.current_role = 'imd_forecaster', district_scope = NULL
    API->>DB: SELECT * FROM sos_requests;
    DB->>DB: RLS Policy: district_scope is NULL -> Full Nationwide Access
    DB-->>API: All Coastal Incidents (Puri, Balasore, Kendrapara)
    API-->>Forecaster: Nationwide Emergency Incident Map
    
    Forecaster->>UI: Clicks 'Step Replay (+6h)'
    UI->>API: POST /api/ml/replay
    API->>ML: ml_service.step_replay()
    ML->>ML: Advance Replay Index to Frame 4 (19.10°N, 86.20°E, 72 KT)
    ML->>ML: Execute CycloneInferencePipeline & Physics Enricher
    ML->>DB: Persist New Advisory (BOB/06/2026/16) in 'advisories'
    ML-->>API: New Bulletin JSON
    API-->>UI: Updated Coordinates & Track Extension
    UI->>UI: Leaflet Map Steps Eye Forward, Recalculates Wind Swaths
```

---

### 7.5 User Flow 5: Offline Emergency Disaster Mode (GSM SMS & USSD)

```mermaid
sequenceDiagram
    autonumber
    actor Citizen as Stranded Citizen (No 4G/5G Cellular Data)
    participant Device as Mobile Browser / Phone
    participant SMS_App as Native GSM SMS Client (2G Voice/SMS)
    participant DEOC as District Emergency Operations Center (1077)

    Note over Citizen,Device: Storm Surge Knocks Out Local Cellular Towers
    Device->>Device: window.navigator.onLine -> false (Triggers 'offline' event)
    Device->>Device: Drop Amber Offline Banner & Load Emergency Pack from localStorage
    Device->>Device: Render Pre-Cached Shelter List & Offline Inland Routing
    
    Citizen->>Device: Clicks 'Send Compact 2G SMS'
    Device->>Device: Generate GSM URI: sms:112?body=SOS%20DANA%20%7C%20LOC%3A%2019.81%2C85.83
    Device->>SMS_App: Open Native SMS Application with Pre-filled Payload
    Citizen->>SMS_App: Hits Send via 2G GSM SMS
    SMS_App->>DEOC: SMS Delivered via Standard 2G Network
    
    opt Acoustic Beacon Search & Rescue
        Citizen->>Device: Clicks 'Trigger Audible Beacon'
        Device->>Device: Web Audio API Oscillator triggers piercing 960Hz / 650Hz Siren
        Note over Citizen,Device: Audible to NDRF Boat Responders within 300 meters
    end
```

---

### 7.6 User Flow 6: Scoped Emergency AI Assistant with Deterministic Fallback

```mermaid
sequenceDiagram
    autonumber
    actor User as Citizen / Responder
    participant ChatUI as Scoped AI Assistant Tab
    participant API as FastAPI Backend (/api/chat)
    participant Groq as Groq AI Cloud (Llama-3.3 / Qwen-2.5)

    User->>ChatUI: Types: "Where is the nearest shelter with available beds in Puri?"
    ChatUI->>API: POST /api/chat {"message": "Where is the nearest shelter...", "language": "en"}
    
    alt Groq AI Available (Online & Valid Key)
        API->>Groq: POST /v1/chat/completions (Model: openai/gpt-oss-120b, temp=0.3)
        Groq-->>API: Authoritative Emergency Guidance with Shelter & Helpline Numbers
        API-->>ChatUI: {"reply": "...", "source": "groq_ai"}
    else Groq Rate-Limited / Offline / Connection Timeout (7s)
        API->>API: Activate Deterministic Local Rule Fallback Engine
        API->>API: Match Keyword 'shelter' -> Extract Designated Shelters & Bed Counts
        API-->>ChatUI: {"reply": "📍 Shelter Guidance: Puri Zilla School Shelter (720 beds)...", "source": "local_fallback"}
    end
    
    ChatUI->>ChatUI: Render Markdown Formatted Emergency Response with Direct Links
```

---

## 8. Database Schema & PostgreSQL Row-Level Security (RLS)

### 8.1 Schema DDL & Non-Superuser Role Separation
To enforce Row-Level Security, an application-level non-superuser role (`cyclone_app`) is provisioned. Superuser accounts intentionally bypass PostgreSQL RLS policies; running under `cyclone_app` guarantees that policies are mathematically enforced by the database core:

```sql
-- Cyclone Shield AI — PostgreSQL Database DDL
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Application non-superuser role
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cyclone_app') THEN
      CREATE USER cyclone_app WITH PASSWORD 'cyclone_secure_pass';
   END IF;
END
$$;

-- 1. Citizens Table
CREATE TABLE citizens (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  aadhaar_number TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  district TEXT NOT NULL,
  mobile_masked TEXT,
  risk_zone TEXT
);

-- 2. Authorities Table
CREATE TABLE authorities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,        -- 'imd_forecaster' | 'district_admin' | 'ngo'
  district_scope TEXT        -- NULL for nationwide access
);

-- 3. Storms Table
CREATE TABLE storms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' -- 'active' | 'dissipated'
);

-- 4. Advisories Table (JSONB)
CREATE TABLE advisories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  storm_id UUID REFERENCES storms(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  payload JSONB NOT NULL
);

-- 5. SOS Requests Table
CREATE TABLE sos_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  citizen_id UUID REFERENCES citizens(id) ON DELETE CASCADE,
  district TEXT NOT NULL,
  location_lat DOUBLE PRECISION,
  location_lon DOUBLE PRECISION,
  status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'in_progress' | 'resolved'
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  handled_by UUID REFERENCES authorities(id) ON DELETE SET NULL,
  handled_at TIMESTAMPTZ
);

-- 6. Shelters Table (Capacity-Aware)
CREATE TABLE shelters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  district TEXT NOT NULL,
  location_lat DOUBLE PRECISION NOT NULL,
  location_lon DOUBLE PRECISION NOT NULL,
  total_capacity INTEGER NOT NULL,
  current_occupancy INTEGER NOT NULL DEFAULT 0,
  contact_number TEXT,
  facilities TEXT[] DEFAULT ARRAY['Water', 'Medical Aid', 'Food', 'Power Backup'],
  status TEXT DEFAULT 'open'
);
```

### 8.2 Row-Level Security (RLS) Policy Definitions

```sql
-- ============================================================
-- ROW-LEVEL SECURITY POLICIES
-- ============================================================

-- A. Citizens RLS
ALTER TABLE citizens ENABLE ROW LEVEL SECURITY;
ALTER TABLE citizens FORCE ROW LEVEL SECURITY;

CREATE POLICY citizen_auth_lookup ON citizens
  FOR SELECT USING (current_setting('app.is_auth_service', true) = 'true');

CREATE POLICY citizen_self_only ON citizens
  FOR SELECT USING (id::text = current_setting('app.current_citizen_id', true));

CREATE POLICY citizen_authority_read ON citizens
  FOR SELECT USING (
    current_setting('app.current_role', true) IN ('imd_forecaster', 'district_admin', 'ngo')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

-- B. Authorities RLS
ALTER TABLE authorities ENABLE ROW LEVEL SECURITY;
ALTER TABLE authorities FORCE ROW LEVEL SECURITY;

CREATE POLICY authority_auth_lookup ON authorities
  FOR SELECT USING (current_setting('app.is_auth_service', true) = 'true');

CREATE POLICY authority_self_only ON authorities
  FOR SELECT USING (id::text = current_setting('app.current_authority_id', true));

-- C. SOS Requests RLS
ALTER TABLE sos_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE sos_requests FORCE ROW LEVEL SECURITY;

CREATE POLICY sos_citizen_select ON sos_requests
  FOR SELECT USING (citizen_id::text = current_setting('app.current_citizen_id', true));

CREATE POLICY sos_citizen_insert ON sos_requests
  FOR INSERT WITH CHECK (citizen_id::text = current_setting('app.current_citizen_id', true));

CREATE POLICY sos_authority_select ON sos_requests
  FOR SELECT USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

CREATE POLICY sos_authority_update ON sos_requests
  FOR UPDATE USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

-- Grants to non-superuser role
GRANT CONNECT ON DATABASE postgres TO cyclone_app;
GRANT USAGE ON SCHEMA public TO cyclone_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cyclone_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cyclone_app;
```

---

## 9. Complete Backend API Reference (20 Endpoints)

The FastAPI backend (`backend/main.py`) exposes **20 production REST endpoints**:

| # | HTTP Method | Endpoint Path | Auth Required | RLS Context Variable Injected | Description |
|:---:|:---:|:---|:---:|:---|:---|
| 1 | `POST` | `/api/auth/citizen` | Public | `app.is_auth_service='true'` | Citizen login via 12-digit Aadhaar. Issues scoped JWT token. |
| 2 | `POST` | `/api/auth/authority` | Public | `app.is_auth_service='true'` | Authority login (bcrypt verification). Issues JWT with `role` & `district_scope`. |
| 3 | `GET` | `/api/citizen/me` | Citizen | `app.current_citizen_id` | Returns authenticated citizen's personal profile. |
| 4 | `GET` | `/api/storms/active` | Public | None | Returns active storm, latest synoptic state, track points & 3-tier wind swaths. |
| 5 | `GET` | `/api/advisories/history` | Public | None | Returns historical bulletins for hindcast review. |
| 6 | `POST` | `/api/sos` | Citizen | `app.current_citizen_id` | Broadcasts emergency distress call. RLS structurally binds `citizen_id`. |
| 7 | `GET` | `/api/sos/my` | Citizen | `app.current_citizen_id` | Retrieves citizen's own past distress calls. |
| 8 | `GET` | `/api/sos/queue` | Authority | `app.current_role`, `app.current_district_scope` | Returns active SOS queue strictly filtered by authority district scope. |
| 9 | `PATCH` | `/api/sos/{sos_id}` | Authority | `app.current_role`, `app.current_district_scope` | Updates incident status (`in_progress` / `resolved`). Forbidden cross-district. |
| 10 | `GET` | `/api/shelters` | Public | None | Lists capacity-aware cyclone shelters with live occupancy and available beds. |
| 11 | `GET` | `/api/shelters/evacuation-route`| Public | None | Calculates surge-safe inland route avoiding <2 km coastal inundation zones. |
| 12 | `GET` | `/api/risk/local-score` | Public | None | Computes hyper-localized Local Risk Score ($LRS \in [0, 100]$) with factor breakdowns. |
| 13 | `GET` | `/api/risk/matrix` | Public | None | Returns District Vulnerability Matrix for all 6 coastal districts. |
| 14 | `GET` | `/api/offline/emergency-pack` | Public | None | Returns standalone JSON emergency safety pack cached into `localStorage`. |
| 15 | `GET` | `/api/i18n/translations` | Public | None | Delivers complete localization dictionaries across 5 coastal languages (EN, OR, HI, BN, TE). |
| 16 | `POST` | `/api/chat` | Public | None | Scoped disaster emergency AI chatbot (Groq AI with deterministic local fallback). |
| 17 | `POST` | `/api/ml/replay` | Authority | `app.current_role` | Steps ML inference pipeline forward to ingest next frame and publish bulletin. |
| 18 | `GET` | `/api/ml/prediction-funnel` | Public | None | Returns 4-horizon Staged Prediction Funnel and architectural comparison matrix. |
| 19 | `GET` | `/api/ml/prediction-funnel/stage/{key}` | Public | None | Returns deep physical parameters and spatial geometries for a specific funnel stage. |
| 20 | `GET` | `/api/alerts/cap` & `.xml` | Public | None | Official OASIS CAP v1.2 XML & JSON feeds for NDMA SACHET alert dissemination. |

---

## 10. Frontend UI/UX Architecture & Government Design System

### 10.1 Government of India Design System
The frontend adheres to the official **Government of India National Portal Aesthetic**:
- **Palette:** Deep Navy (`#0B2F5E`), Indian Saffron (`#FF9933`), Pure White (`#FFFFFF`), India Green (`#138808`), Crimson Alert Red (`#DC2626`).
- **Tricolor Ribbon:** Prominently positioned at the top of every page.
- **Ashoka Lion Emblem SVG:** Officially rendered in the brand header.
- **Card Styling:** Clean white containers with flat `1px solid #D1D5DB` borders (no bloated drop shadows, no soft-rounded corners, no dark mode).
- **Square-Cornered Buttons:** Modern, crisp button design (`border-radius: 0px` or `2px`).

### 10.2 Key UI Components
1. **Command Telemetry Ribbon (`.command-telemetry-bar`):** A consolidated 6-metric critical overview banner placed directly below the alert status:
   - `Active System`: Cyclone Dana (VSCS)
   - `Eye Position`: 18.42°N, 86.85°E
   - `Landfall Window`: 12–14 Hours (Puri-Dhamra)
   - `Sustained Wind`: 120 km/h (Gusts: 145)
   - `Storm Surge`: 2.0 – 3.2 m
   - `Shelters Ready`: 8 Active (3.6k Beds Free)
2. **Horizontal Operational Horizon Stepper (`.horizon-stepper`):** An interactive 4-stage stepper allowing instantaneous exploration of forecasting stages (T-18d &rarr; T-14d &rarr; T-7d &rarr; T-3d) with dynamic map layer switching and telemetry updates.
3. **Interactive Leaflet GIS Map:**
   - Dual Basemap Support: Toggle between IMD Nautical Topographic Chart and NASA Earthdata GIBS satellite imagery.
   - Symbology: 72h IMD Uncertainty Cone, 3-tier wind hazard swaths, past track polyline, active eye radar marker, and operational shelter pins.
   - Dynamic Animated Surge-Safe Route Polyline with turn-by-turn navigation card.
4. **Offline Emergency Mode Banner:** Appears automatically if device disconnects from internet, exposing download links for the offline pack and compact SMS dispatch buttons.
5. **Audible Speech Siren:** Integrates `SpeechSynthesisUtterance` to voice the active emergency warning bulletin in the user's selected coastal language.

---

## 11. Automated Verification, Security Audit & Test Results

### 11.1 API Integration Test Suite (`tests/test_api_endpoints.py`)
Run command:
```powershell
python tests/test_api_endpoints.py
```
**Results: 20/20 Integration Tests Passing (100% Success Rate).**

```
--- 1. Testing Citizen Login ---
PASS: Citizen logged in as Amanullah
--- 2. Testing Citizen Profile (RLS Protected) ---
PASS: Citizen profile retrieved under RLS.
--- 3. Testing Active Storm Advisory (Public) ---
PASS: Active storm advisory retrieved: Cyclone Dana, Warning: RED ALERT
--- 4. Testing Authority Login ---
PASS: Authority admin_puri logged in.
PASS: Authority forecaster_hq logged in.
--- 5. Testing Citizen SOS Submission ---
PASS: New SOS created by citizen (ID: dea6936b-..., Status: pending)
--- 6. Testing Citizen Get My SOS Requests ---
PASS: Citizen can view own SOS requests (21 found).
--- 7. Testing Authority SOS Queue with District RLS ---
PASS: Puri Admin sees only Puri SOS requests (21 items, 0 leaked from other districts).
PASS: Forecaster sees nationwide SOS requests (22 items).
--- 8. Testing Authority Patch SOS Status ---
PASS: Authority updated SOS status to 'in_progress'.
--- 9. Testing Shelters API ---
PASS: Retrieved 4 shelters in Puri with live capacity data.
--- 10. Testing Scoped AI Emergency Chatbot ---
PASS: Emergency chatbot returned scoped shelter guidance.
--- 11. Testing ML Pipeline Replay Trigger ---
PASS: ML replay step executed. Bulletin: BOB/06/2026/16
--- 12. Testing OASIS CAP v1.2 Alert Generator ---
PASS: CAP v1.2 alert verified (ID: IN-IMD-CYCLONE-DANA-...).
--- 13. Testing AI/ML Pipeline Status & Explainability ---
PASS: 6-Layer AI architecture verified (CycloneInferencePipeline v2.4).
--- 14. Testing Multi-Stakeholder Intelligence Metrics ---
PASS: Stakeholder intelligence metrics verified across all 4 sectors.
--- 15. Testing Odisha Zero Casualty Model Benchmarks ---
PASS: Odisha OSDMA Zero Casualty benchmarks verified.
--- 16. Testing Staged Prediction Funnel API (T-18d to T-3d) ---
PASS: Staged Prediction Funnel verified across all 4 horizons.
--- 17. Testing Surge-Safe Evacuation Corridor Routing ---
PASS: Surge-safe evacuation route calculated to Puri Zilla School Shelter.
--- 18. Testing Offline Emergency Pack Generation ---
PASS: Standalone offline emergency disaster pack generated.
--- 19. Testing Multilingual Coastal Localization Engine ---
PASS: Localized dictionaries verified for EN, OR, HI, BN, TE.
--- 20. Testing Multi-Hazard Local Risk Score (LRS) Engine ---
PASS: Hyper-localized risk score calculated: 56.7/100 (MODERATE RISK).
PASS: District Vulnerability Matrix verified for 6 coastal districts.
```

### 11.2 Direct PostgreSQL RLS Penetration Suite (`tests/test_rls_direct.py`)
Run command:
```powershell
python tests/test_rls_direct.py
```
**Results: 6/6 Direct Database Security Tests Passing (100% Success Rate).**

```
PASS [1/6]: Unauthenticated query to 'citizens' returns 0 rows.
PASS [2/6]: Unauthenticated query to 'sos_requests' returns 0 rows.
PASS [3/6]: Citizen Amanullah can read own profile (Found Amanullah).
PASS [4/6]: Citizen Amanullah CANNOT read Ramesh's profile (RLS strictly enforced).
PASS [5/6]: Citizen Amanullah only sees their own SOS request (20 found, 0 from others).
PASS [6/6]: District Admin (Puri) sees only Puri SOS requests (20 found), zero from Balasore.
PASS [BONUS]: IMD Forecaster (nationwide role) sees all districts (21 total requests).
```

---

## 12. Runtime Operations & Quick-Start Guide

### 12.1 Prerequisites
- **Python 3.10+** (Active in virtual environment)
- **PostgreSQL 18** (Listening on `127.0.0.1:5433`)

### 12.2 Launch Commands

#### Step 1: Start PostgreSQL Database
```powershell
& "C:\Program Files\PostgreSQL\18\bin\postgres.exe" -D "d:\sih70\hehe\pgdata" -p 5433
```

#### Step 2: Initialize Database & Seed Operational Data
```powershell
# (Only required on fresh setup)
psql -h 127.0.0.1 -p 5433 -U postgres -d postgres -f backend/database/schema.sql
python backend/database/seed.py
```

#### Step 3: Launch FastAPI Application Server
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 12.3 Active Application URLs
- **Executive Disaster Command Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger REST API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OASIS CAP v1.2 XML Feed:** [http://127.0.0.1:8000/api/alerts/cap.xml](http://127.0.0.1:8000/api/alerts/cap.xml)
- **OASIS CAP v1.2 JSON Feed:** [http://127.0.0.1:8000/api/alerts/cap](http://127.0.0.1:8000/api/alerts/cap)

### 12.4 Demo Verification Credentials

| Role / User Profile | Identifier / User ID | Password / Credential | Scope & Permitted Actions |
| :--- | :--- | :--- | :--- |
| **Citizen (Puri District)** | `1234 1234 1234` | Aadhaar Auth (No password) | Access personal profile, trigger GPS SOS, compute surge-safe route. |
| **District Admin (Puri DEOC)** | `admin_puri` | `admin@puri` | View and manage live Puri distress queue. Zero access to Balasore. |
| **National Forecaster (IMD HQ)** | `forecaster_hq` | `imd@2026` | Nationwide incident queue oversight, trigger ML replay step (+6h). |
| **Relief NGO Coordinator** | `ngo_puri` | `ngo@puri` | District relief triage and shelter capacity coordination. |

---
*Cyclone Shield AI — Master System Architecture & Context Specification | Team Hexaminds | Smart India Hackathon (SIH 2026)*
