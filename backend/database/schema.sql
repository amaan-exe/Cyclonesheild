-- Cyclone Horizon - PostgreSQL Database Schema with Row-Level Security (RLS)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Drop existing tables if re-initializing
DROP TABLE IF EXISTS sos_requests CASCADE;
DROP TABLE IF EXISTS advisories CASCADE;
DROP TABLE IF EXISTS storms CASCADE;
DROP TABLE IF EXISTS authorities CASCADE;
DROP TABLE IF EXISTS citizens CASCADE;
DROP TABLE IF EXISTS shelters CASCADE;

-- Create application non-superuser role to enforce RLS (superusers always bypass RLS)
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cyclone_app') THEN
      BEGIN
         CREATE USER cyclone_app WITH PASSWORD 'cyclone_secure_pass';
      EXCEPTION WHEN OTHERS THEN
         -- When connecting on managed cloud PostgreSQL (e.g. Neon, Supabase) as non-superuser
         RAISE NOTICE 'Skipping CREATE USER cyclone_app: %', SQLERRM;
      END;
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
  district_scope TEXT        -- NULL for roles with nationwide access
);

-- 3. Storms Table
CREATE TABLE storms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' -- 'active' | 'dissipated'
);

-- 4. Advisories Table
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

-- 6. Shelters Table (supporting capacity-aware shelter allocation)
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

-- ============================================================
-- ROW-LEVEL SECURITY (RLS) ENFORCEMENT
-- ============================================================

-- A. Citizens RLS
ALTER TABLE citizens ENABLE ROW LEVEL SECURITY;
ALTER TABLE citizens FORCE ROW LEVEL SECURITY;

CREATE POLICY citizen_auth_lookup ON citizens
  FOR SELECT
  USING (
    current_setting('app.is_auth_service', true) = 'true'
  );

CREATE POLICY citizen_self_only ON citizens
  FOR SELECT
  USING (
    id::text = current_setting('app.current_citizen_id', true)
  );

CREATE POLICY citizen_authority_read ON citizens
  FOR SELECT
  USING (
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
  FOR SELECT
  USING (
    current_setting('app.is_auth_service', true) = 'true'
  );

CREATE POLICY authority_self_only ON authorities
  FOR SELECT
  USING (
    id::text = current_setting('app.current_authority_id', true)
  );

-- C. SOS Requests RLS
ALTER TABLE sos_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE sos_requests FORCE ROW LEVEL SECURITY;

-- Citizens can only select their own SOS requests
CREATE POLICY sos_citizen_select ON sos_requests
  FOR SELECT
  USING (
    citizen_id::text = current_setting('app.current_citizen_id', true)
  );

-- Citizens can only insert SOS requests for themselves
CREATE POLICY sos_citizen_insert ON sos_requests
  FOR INSERT
  WITH CHECK (
    citizen_id::text = current_setting('app.current_citizen_id', true)
  );

-- Authorities can view SOS requests within their district scope
-- (nationwide roles like imd_forecaster get district_scope of NULL or empty,
-- which the policy treats as no restriction)
CREATE POLICY sos_authority_select ON sos_requests
  FOR SELECT
  USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

-- Authorities can update SOS status within their district scope
CREATE POLICY sos_authority_update ON sos_requests
  FOR UPDATE
  USING (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  )
  WITH CHECK (
    current_setting('app.current_role', true) IN ('district_admin', 'ngo', 'imd_forecaster')
    AND (
      current_setting('app.current_district_scope', true) IS NULL
      OR current_setting('app.current_district_scope', true) = ''
      OR district = current_setting('app.current_district_scope', true)
    )
  );

-- Grant privileges to application user if role exists
DO $$
BEGIN
   IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cyclone_app') THEN
      BEGIN
         GRANT USAGE ON SCHEMA public TO cyclone_app;
         GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cyclone_app;
         GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cyclone_app;
      EXCEPTION WHEN OTHERS THEN
         RAISE NOTICE 'Privilege grant to cyclone_app skipped: %', SQLERRM;
      END;
   END IF;
END
$$;

