"""
Cyclone Horizon — I/O Utilities
Safe file downloads, Zarr/Parquet/NetCDF helpers, config loading.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import requests
import pandas as pd
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("io")


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load YAML config file and return as dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load model hyperparameter config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# File downloads with retry, resume, and checksum
# ============================================================================

def download_file(
    url: str,
    dest_path: str,
    max_retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 8192,
    expected_hash: Optional[str] = None,
) -> str:
    """
    Download a file with retry/resume support and optional SHA256 verification.
    
    Parameters
    ----------
    url : source URL
    dest_path : local destination path
    max_retries : number of retry attempts
    timeout : request timeout in seconds
    chunk_size : download chunk size in bytes
    expected_hash : optional SHA256 hex digest to verify
    
    Returns
    -------
    str : path to downloaded file
    
    Raises
    ------
    RuntimeError : if download fails after all retries
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} → {dest_path} (attempt {attempt}/{max_retries})")
            
            headers = {}
            mode = "wb"
            existing_size = 0
            
            # Resume partial download
            if dest.exists():
                existing_size = dest.stat().st_size
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"
            
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            
            if response.status_code == 416:
                # Range not satisfiable — file already complete
                logger.info(f"File already complete: {dest_path}")
                break
            
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0)) + existing_size
            
            with open(dest_path, mode) as f:
                downloaded = existing_size
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            logger.info(f"Downloaded {downloaded:,} bytes to {dest_path}")
            break
            
        except (requests.RequestException, IOError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # exponential backoff
    
    # Verify checksum if provided
    if expected_hash:
        actual_hash = file_sha256(dest_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Checksum mismatch for {dest_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        logger.info(f"Checksum verified: {dest_path}")
    
    return dest_path


def file_sha256(filepath: str) -> str:
    """Compute SHA256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ============================================================================
# Data I/O helpers
# ============================================================================

def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame to Parquet with directory creation."""
    ensure_dir(str(Path(path).parent))
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet file into DataFrame."""
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df
