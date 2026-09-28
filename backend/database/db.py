import os
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from urllib.parse import urlparse

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
    DB_USER = parsed.username or "cyclone_app"
    DB_PASS = parsed.password or "cyclone_secure_pass"
    DB_HOST = parsed.hostname or "127.0.0.1"
    DB_PORT = parsed.port or 5432
    DB_NAME = parsed.path.lstrip("/") or "postgres"
else:
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.environ.get("DB_PORT", 5432))
    DB_NAME = os.environ.get("DB_NAME", "postgres")
    DB_USER = os.environ.get("DB_USER", "cyclone_app")
    DB_PASS = os.environ.get("DB_PASS", "cyclone_secure_pass")

pool = None

def get_pool():
    global pool
    if pool is None:
        if DATABASE_URL:
            dsn = DATABASE_URL.replace("postgres://", "postgresql://", 1) if DATABASE_URL.startswith("postgres://") else DATABASE_URL
            pool = ThreadedConnectionPool(
                minconn=int(os.environ.get("DB_POOL_MIN", 2)),
                maxconn=int(os.environ.get("DB_POOL_MAX", 20)),
                dsn=dsn
            )
        else:
            kwargs = {
                "minconn": int(os.environ.get("DB_POOL_MIN", 2)),
                "maxconn": int(os.environ.get("DB_POOL_MAX", 20)),
                "host": DB_HOST,
                "port": DB_PORT,
                "dbname": DB_NAME,
                "user": DB_USER,
                "password": DB_PASS
            }
            if DB_HOST not in ("127.0.0.1", "localhost") or "DB_SSLMODE" in os.environ:
                kwargs["sslmode"] = os.environ.get("DB_SSLMODE", "require")
            pool = ThreadedConnectionPool(**kwargs)
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
