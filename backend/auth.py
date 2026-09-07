import os
import datetime
from typing import Optional, Dict, Any
import jwt
import bcrypt
from fastapi import Depends, HTTPException, status, Header
from backend.database.db import get_db_cursor

SECRET_KEY = os.environ.get("JWT_SECRET", "cyclone_horizon_sih2026_super_secret_key_987654321")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 # 2 hours

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + (expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.datetime.now(datetime.timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please log in again."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token."
        )

def get_current_token_payload(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header."
        )
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'."
        )
    return decode_token(parts[1])

def get_current_citizen(payload: Dict[str, Any] = Depends(get_current_token_payload)) -> Dict[str, Any]:
    if payload.get("type") != "citizen" or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to authenticated citizens."
        )
    # Validate citizen exists under RLS
    citizen_id = payload["sub"]
    with get_db_cursor({"citizen_id": citizen_id}) as cur:
        cur.execute("SELECT id, aadhaar_number, name, district, mobile_masked, risk_zone FROM citizens WHERE id = %s;", (citizen_id,))
        citizen = cur.fetchone()
        if not citizen:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Citizen account not found or access denied by database RLS."
            )
        return dict(citizen)

def get_current_authority(payload: Dict[str, Any] = Depends(get_current_token_payload)) -> Dict[str, Any]:
    if payload.get("type") != "authority" or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to authorized disaster authorities/responders."
        )
    authority_id = payload["sub"]
    with get_db_cursor({"authority_id": authority_id}) as cur:
        cur.execute("SELECT id, user_id, role, district_scope FROM authorities WHERE id = %s;", (authority_id,))
        auth = cur.fetchone()
        if not auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authority credentials not found or revoked."
            )
        return dict(auth)
