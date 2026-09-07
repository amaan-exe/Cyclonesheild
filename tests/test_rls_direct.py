import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "dbname": "postgres",
    "user": "cyclone_app",
    "password": "cyclone_secure_pass",
    "host": "127.0.0.1",
    "port": 5433
}


def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn

def test_rls_policies():
    # Fetch citizen IDs using auth service flag in dedicated connection
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SET LOCAL app.is_auth_service = 'true';")
    cur.execute("SELECT id, name, district FROM citizens ORDER BY name;")
    citizens = cur.fetchall()
    aman_id = str(citizens[0]["id"]) if citizens[0]["name"] == "Amanullah" else str(citizens[1]["id"])
    ramesh_id = str(citizens[1]["id"]) if citizens[1]["name"] == "Ramesh Kumar" else str(citizens[0]["id"])
    cur.close()
    conn.close()
    print(f"Test Citizen IDs - Amanullah: {aman_id}, Ramesh: {ramesh_id}")

    # TEST 1: Fresh connection with NO session variables set (unauthenticated)
    conn1 = get_conn()
    cur1 = conn1.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur1.execute("SELECT * FROM citizens;")
    rows = cur1.fetchall()
    assert len(rows) == 0, f"Unauthenticated request leaked {len(rows)} citizens!"
    print("PASS [1/6]: Unauthenticated query to 'citizens' returns 0 rows.")

    cur1.execute("SELECT * FROM sos_requests;")
    rows = cur1.fetchall()
    assert len(rows) == 0, f"Unauthenticated request leaked {len(rows)} sos_requests!"
    print("PASS [2/6]: Unauthenticated query to 'sos_requests' returns 0 rows.")
    cur1.close()
    conn1.close()

    # TEST 2: Amanullah session
    conn2 = get_conn()
    cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur2.execute("SET LOCAL app.current_citizen_id = %s;", (aman_id,))
    cur2.execute("SELECT * FROM citizens;")
    rows = cur2.fetchall()
    assert len(rows) == 1 and str(rows[0]["id"]) == aman_id, "Amanullah could not read own profile!"
    print(f"PASS [3/6]: Citizen Amanullah can read own profile (Found {rows[0]['name']}).")

    # Try to query Ramesh's row while authenticated as Amanullah
    cur2.execute("SELECT * FROM citizens WHERE id = %s;", (ramesh_id,))
    rows = cur2.fetchall()
    assert len(rows) == 0, "SECURITY VIOLATION: Amanullah could read Ramesh's profile!"
    print("PASS [4/6]: Citizen Amanullah CANNOT read Ramesh's profile (RLS strictly enforced).")

    # Amanullah queries SOS requests
    cur2.execute("SELECT * FROM sos_requests;")
    rows = cur2.fetchall()
    assert len(rows) >= 1 and all(str(r["citizen_id"]) == aman_id for r in rows), "Amanullah saw someone else's SOS!"
    print(f"PASS [5/6]: Citizen Amanullah only sees their own SOS request ({len(rows)} found, 0 from others).")
    cur2.close()
    conn2.close()

    # TEST 3: District Admin for Puri vs Balasore
    conn3 = get_conn()
    cur3 = conn3.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur3.execute("SELECT set_config('app.current_role', 'district_admin', true);")
    cur3.execute("SELECT set_config('app.current_district_scope', 'Puri', true);")
    cur3.execute("SELECT * FROM sos_requests;")
    rows = cur3.fetchall()
    assert len(rows) >= 1 and all(r["district"] == "Puri" for r in rows), "Puri Admin saw other district SOS!"
    print(f"PASS [6/6]: District Admin (Puri) sees only Puri SOS requests ({len(rows)} found), zero from Balasore.")
    cur3.close()
    conn3.close()

    # BONUS TEST 4: IMD Forecaster nationwide access
    conn4 = get_conn()
    cur4 = conn4.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur4.execute("SELECT set_config('app.current_role', 'imd_forecaster', true);")
    cur4.execute("SELECT set_config('app.current_district_scope', '', true);")
    cur4.execute("SELECT * FROM sos_requests;")
    all_sos = cur4.fetchall()
    assert len(all_sos) >= 2, f"IMD forecaster did not see all nationwide SOS requests! (saw {len(all_sos)})"
    print(f"PASS [BONUS]: IMD Forecaster (nationwide role) sees all districts ({len(all_sos)} total requests).")
    cur4.close()
    conn4.close()


    print("\n=======================================================")
    print("ALL RLS VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_rls_policies()
