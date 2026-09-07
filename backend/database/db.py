import os
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", 5433))
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "cyclone_app")
DB_PASS = os.environ.get("DB_PASS", "cyclone_secure_pass")

pool = None

def get_pool():
    global pool
    if pool is None:
        pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
    return pool

@contextmanager
def get_db_cursor(session_context=None):
    """
    Context manager yielding a RealDictCursor with PostgreSQL RLS session variables set.
    session_context can contain:
      - citizen_id: UUID or str
      - authority_id: UUID or str
      - role: str ('imd_forecaster', 'district_admin', 'ngo')
      - district_scope: str or None
      - is_auth_service: bool
    """
    p = get_pool()
    conn = p.getconn()
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        if session_context:
            if session_context.get("is_auth_service"):
                cur.execute("SELECT set_config('app.is_auth_service', 'true', true);")
            if session_context.get("citizen_id"):
                cur.execute("SELECT set_config('app.current_citizen_id', %s, true);", (str(session_context["citizen_id"]),))
            if session_context.get("authority_id"):
                cur.execute("SELECT set_config('app.current_authority_id', %s, true);", (str(session_context["authority_id"]),))
            if session_context.get("role"):
                cur.execute("SELECT set_config('app.current_role', %s, true);", (str(session_context["role"]),))
            if "district_scope" in session_context:
                scope = session_context.get("district_scope") or ""
                cur.execute("SELECT set_config('app.current_district_scope', %s, true);", (str(scope),))

        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        p.putconn(conn)
