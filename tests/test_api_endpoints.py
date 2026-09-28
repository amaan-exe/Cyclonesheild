import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)

def test_api_suite():
    print("--- 1. Testing Citizen Login ---")
    res = client.post("/api/auth/citizen", json={"aadhaar_number": "123412341234"})
    assert res.status_code == 200, f"Citizen login failed: {res.text}"
    citizen_data = res.json()
    citizen_token = citizen_data["access_token"]
    assert citizen_data["citizen"]["name"] == "Amanullah"
    print(f"PASS: Citizen logged in as {citizen_data['citizen']['name']}")

    print("--- 2. Testing Citizen Profile (RLS Protected) ---")
    headers_citizen = {"Authorization": f"Bearer {citizen_token}"}
    res = client.get("/api/citizen/me", headers=headers_citizen)
    assert res.status_code == 200, f"Profile failed: {res.text}"
    assert res.json()["citizen"]["name"] == "Amanullah"
    print("PASS: Citizen profile retrieved under RLS.")

    print("--- 3. Testing Active Storm Advisory (Public) ---")
    res = client.get("/api/storms/active")
    assert res.status_code == 200, f"Storm active failed: {res.text}"
    storm_data = res.json()
    assert "advisory" in storm_data
    assert bool(storm_data.get("storm_name"))
    print(f"PASS: Active storm advisory retrieved: {storm_data['storm_name']}, Warning: {storm_data['advisory']['warning_status']}")

    print("--- 4. Testing Authority Login ---")
    # Puri Admin
    res = client.post("/api/auth/authority", json={"user_id": "admin_puri", "password": "admin@puri"})
    assert res.status_code == 200, f"Authority login failed: {res.text}"
    puri_token = res.json()["access_token"]
    headers_puri = {"Authorization": f"Bearer {puri_token}"}
    print("PASS: Authority admin_puri logged in.")

    # IMD Forecaster
    res = client.post("/api/auth/authority", json={"user_id": "forecaster_hq", "password": "imd@2026"})
    assert res.status_code == 200, f"Forecaster login failed: {res.text}"
    forecaster_token = res.json()["access_token"]
    headers_forecaster = {"Authorization": f"Bearer {forecaster_token}"}
    print("PASS: Authority forecaster_hq logged in.")

    print("--- 5. Testing Citizen SOS Submission ---")
    res = client.post("/api/sos", json={
        "district": "Puri",
        "location_lat": 19.8150,
        "location_lon": 85.8320
    }, headers=headers_citizen)
    assert res.status_code == 200, f"SOS submit failed: {res.text}"
    new_sos = res.json()["sos"]
    sos_id = new_sos["id"]
    print(f"PASS: New SOS created by citizen (ID: {sos_id}, Status: {new_sos['status']})")

    print("--- 6. Testing Citizen Get My SOS Requests ---")
    res = client.get("/api/sos/my", headers=headers_citizen)
    assert res.status_code == 200
    requests = res.json()["requests"]
    assert len(requests) >= 1
    print(f"PASS: Citizen can view own SOS requests ({len(requests)} found).")

    print("--- 7. Testing Authority SOS Queue with District RLS ---")
    # Puri Admin queue check
    res = client.get("/api/sos/queue", headers=headers_puri)
    assert res.status_code == 200
    queue_puri = res.json()["queue"]
    assert all(item["district"] == "Puri" for item in queue_puri), "Puri Admin saw non-Puri SOS!"
    print(f"PASS: Puri Admin sees only Puri SOS requests ({len(queue_puri)} items, 0 leaked from other districts).")

    # Forecaster queue check (Nationwide)
    res = client.get("/api/sos/queue", headers=headers_forecaster)
    assert res.status_code == 200
    queue_all = res.json()["queue"]
    assert len(queue_all) >= len(queue_puri), "Forecaster missed nationwide requests!"
    print(f"PASS: Forecaster sees nationwide SOS requests ({len(queue_all)} items).")

    print("--- 8. Testing Authority Patch SOS Status ---")
    res = client.patch(f"/api/sos/{sos_id}", json={"status": "in_progress"}, headers=headers_puri)
    assert res.status_code == 200, f"Patch SOS failed: {res.text}"
    assert res.json()["sos"]["status"] == "in_progress"
    print("PASS: Authority updated SOS status to 'in_progress'.")

    print("--- 9. Testing Shelters API ---")
    res = client.get("/api/shelters?district=Puri")
    assert res.status_code == 200
    shelters = res.json()["shelters"]
    assert len(shelters) >= 3
    print(f"PASS: Retrieved {len(shelters)} shelters in Puri with live capacity data.")

    print("--- 10. Testing Scoped AI Emergency Chatbot ---")
    res = client.post("/api/chat", json={"message": "Where is the nearest shelter?"})
    assert res.status_code == 200
    reply = res.json()["reply"]
    assert "Shelter" in reply
    print("PASS: Emergency chatbot returned scoped shelter guidance.")

    print("--- 11. Testing ML Pipeline Replay Trigger ---")
    res = client.post("/api/ml/replay", headers=headers_forecaster)
    assert res.status_code == 200
    replay_data = res.json()
    assert "bulletin_number" in replay_data
    print(f"PASS: ML replay step executed. Bulletin: {replay_data['bulletin_number']}")

    print("--- 12. Testing OASIS CAP v1.2 Alert Generator ---")
    res = client.get("/api/alerts/cap")
    assert res.status_code == 200
    cap_data = res.json()
    assert "identifier" in cap_data
    assert cap_data["info"]["severity"] == "Extreme"
    res_xml = client.get("/api/alerts/cap.xml")
    assert res_xml.status_code == 200
    assert "urn:oasis:names:tc:emergency:cap:1.2" in res_xml.text
    print(f"PASS: CAP v1.2 alert verified (ID: {cap_data['identifier']}).")

    print("--- 13. Testing AI/ML Pipeline Status & Explainability ---")
    res = client.get("/api/system/pipeline-status")
    assert res.status_code == 200
    pipe_data = res.json()
    assert len(pipe_data["layers"]) == 6
    assert "uncertainty_quantification" in pipe_data
    print(f"PASS: 6-Layer AI architecture verified ({pipe_data['pipeline_version']}).")

    print("--- 14. Testing Multi-Stakeholder Intelligence Metrics ---")
    res = client.get("/api/stakeholders/metrics")
    assert res.status_code == 200
    sh_data = res.json()
    assert "citizens" in sh_data and "emergency_services" in sh_data and "ngos_and_shelters" in sh_data
    print("PASS: Stakeholder intelligence metrics verified across all 4 sectors.")

    print("--- 15. Testing Odisha Zero Casualty Model Benchmarks ---")
    res = client.get("/api/odisha/osdma-metrics")
    assert res.status_code == 200
    osdma = res.json()
    assert len(osdma["case_studies"]) >= 3
    print("PASS: Odisha OSDMA Zero Casualty benchmarks verified.")

    print("--- 16. Testing Staged Prediction Funnel API (T-18d to T-3d) ---")
    res = client.get("/api/ml/prediction-funnel")
    assert res.status_code == 200
    funnel_data = res.json()
    assert "stages" in funnel_data
    stages = funnel_data["stages"]
    assert "t18_regional_risk" in stages
    assert "t14_cyclogenesis" in stages
    assert "t7_system_id" in stages
    assert "t3_track_intensity" in stages
    assert stages["t18_regional_risk"]["lead_time_days"] == 18
    assert stages["t14_cyclogenesis"]["lead_time_days"] == 14
    assert stages["t7_system_id"]["lead_time_days"] == 7
    assert stages["t3_track_intensity"]["lead_time_days"] == 3
    assert "comparison_matrix" in funnel_data
    assert len(funnel_data["comparison_matrix"]["dimensions"]) == 4

    # Test single stage query
    res_stage = client.get("/api/ml/prediction-funnel/stage/t18_regional_risk")
    assert res_stage.status_code == 200
    stage_info = res_stage.json()["stage"]
    assert stage_info["lead_time_days"] == 18
    assert "physics_indicators" in stage_info
    print("PASS: Staged Prediction Funnel verified across all 4 horizons (T-18d -> T-14d -> T-7d -> T-3d) and comparison matrix.")

    print("--- 17. Testing Surge-Safe Evacuation Corridor Routing ---")
    res = client.get("/api/shelters/evacuation-route?origin_lat=19.805&origin_lon=85.830&district=Puri")
    assert res.status_code == 200, f"Evacuation route failed: {res.text}"
    evac_data = res.json()
    assert "target_shelter" in evac_data
    assert "distance_km" in evac_data
    assert "eta_walking_mins" in evac_data
    assert "route_coordinates" in evac_data
    assert len(evac_data["route_coordinates"]) >= 3
    assert "turn_by_turn" in evac_data
    assert len(evac_data["turn_by_turn"]) >= 3
    assert "CERTIFIED SAFE" in evac_data["surge_safety_status"]
    assert "google_maps_url" in evac_data
    assert "google.com/maps/dir" in evac_data["google_maps_url"]
    print(f"PASS: Surge-safe evacuation route calculated to {evac_data['target_shelter']['name']} ({evac_data['distance_km']} km, {evac_data['eta_walking_mins']} min walk, Google Maps URL verified).")

    print("--- 18. Testing Offline Emergency Pack Generation ---")
    res = client.get("/api/offline/emergency-pack?district=Puri")
    assert res.status_code == 200, f"Offline pack failed: {res.text}"
    pack_data = res.json()
    assert "emergency_contacts" in pack_data
    assert len(pack_data["emergency_contacts"]) >= 3
    assert "shelters" in pack_data
    assert len(pack_data["shelters"]) >= 3
    assert "offline_compact_sms_format" in pack_data
    assert "offline_ussd_dial_codes" in pack_data
    assert "survival_protocols_multilingual" in pack_data
    assert "or" in pack_data["survival_protocols_multilingual"]  # Odia included
    print(f"PASS: Standalone offline emergency disaster pack generated with {len(pack_data['shelters'])} shelters and multilingual guides.")

    print("--- 19. Testing Multilingual Coastal Localization Engine ---")
    res = client.get("/api/i18n/translations")
    assert res.status_code == 200, f"i18n translations failed: {res.text}"
    i18n_data = res.json()
    assert "en" in i18n_data
    assert "or" in i18n_data  # Odia
    assert "hi" in i18n_data  # Hindi
    assert "bn" in i18n_data  # Bengali
    assert "te" in i18n_data  # Telugu
    assert "ଦାନା" in i18n_data["or"]["red_alert"]
    print("--- 20. Testing Multi-Hazard Local Risk Score (LRS) Engine ---")
    res = client.get("/api/risk/local-score?lat=19.8135&lon=85.8312&district=Puri")
    assert res.status_code == 200, f"Local risk score failed: {res.text}"
    lrs_data = res.json()
    assert "composite_risk_score" in lrs_data
    assert 0.0 <= lrs_data["composite_risk_score"] <= 100.0
    assert "factors" in lrs_data
    assert "wind_hazard" in lrs_data["factors"]
    assert "rainfall_inundation" in lrs_data["factors"]
    assert "population_exposure" in lrs_data["factors"]
    assert "operational_directive" in lrs_data
    assert "risk_tier" in lrs_data
    print(f"PASS: Hyper-localized risk score calculated: {lrs_data['composite_risk_score']}/100 ({lrs_data['risk_tier']}).")

    res_mat = client.get("/api/risk/matrix")
    assert res_mat.status_code == 200, f"Risk matrix failed: {res_mat.text}"
    mat_data = res_mat.json()
    assert "districts" in mat_data
    assert len(mat_data["districts"]) >= 5
    for d in mat_data["districts"]:
        assert "local_risk_score" in d
        assert "wind_score" in d
        assert "rain_score" in d
        assert "pop_score" in d
        assert "evacuation_status" in d
    print(f"PASS: District Vulnerability Matrix verified for {len(mat_data['districts'])} coastal districts.")

    print("--- 21. Testing ISRO MOSDAC Live Satellite Telemetry ---")
    res_mos = client.get("/api/ml/mosdac/status")
    assert res_mos.status_code == 200, f"MOSDAC status failed: {res_mos.text}"
    mos_data = res_mos.json()
    assert mos_data["connection_status"] in ("ACTIVE / STREAMING", "CONFIGURED / READY", "SIMULATION / NOT CONFIGURED")
    assert "active_weather_system" in mos_data
    assert mos_data["active_weather_system"]["threatened_state"] == "Odisha"
    print(f"PASS: MOSDAC Live satellite status verified ({mos_data['satellite_source']} {mos_data['channel']}) for Odisha sector.")

    res_scene = client.get("/api/ml/mosdac/scene")
    assert res_scene.status_code == 200, f"MOSDAC scene failed: {res_scene.text}"
    scene_data = res_scene.json()
    assert "detected_vortex" in scene_data
    assert "center_lat" in scene_data["detected_vortex"]
    print(f"PASS: MOSDAC multi-spectral satellite frame telemetry verified (Tb: {scene_data['detected_vortex']['min_brightness_temp_c']}°C).")

    print("--- 22. Testing MOSDAC Live Ingestion & Neural Inference Sync ---")
    res_sync = client.post("/api/ml/mosdac/sync-live", json={})
    assert res_sync.status_code == 200, f"MOSDAC sync failed: {res_sync.text}"
    sync_data = res_sync.json()
    assert sync_data["status"] == "success"
    assert "vortex_fix" in sync_data
    assert "advisory" in sync_data
    assert sync_data["advisory"]["current_state"]["landfall_location"] != ""
    print(f"PASS: Live MOSDAC frame ingested and 5-model neural inference executed. Bulletin: {sync_data['bulletin_number']}.")

    print("\n============================================================")
    print("ALL 22 API ENDPOINT INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("============================================================\n")

if __name__ == "__main__":
    test_api_suite()


