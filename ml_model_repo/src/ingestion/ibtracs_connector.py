"""
Cyclone Horizon — IBTrACS Connector
Downloads and parses IBTrACS v04r01 best-track data for the North Indian Ocean.
This is the backbone of the system — everything joins to storm IDs from here.

Source: https://www.ncei.noaa.gov/products/international-best-track-archive
Format: CSV (one row per 3-hourly fix per storm)
License: Public domain (NOAA)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ingestion.base_connector import BaseConnector, RawFile
from src.utils.constants import (
    IBTRACS_COLUMNS, NIO_BOUNDS, IMD_CATEGORIES,
    wind_to_imd_index, wind_to_imd_category,
)
from src.utils.io import download_file, save_parquet, file_sha256
from src.utils.logging_config import get_logger

logger = get_logger("ingestion.ibtracs")


class IBTrACSConnector(BaseConnector):
    """
    Connector for IBTrACS v04r01 North Indian Ocean best-track data.
    
    Downloads the basin-specific CSV, parses it into a clean DataFrame,
    computes derived fields (IMD category, time since genesis), and
    exports to Parquet.
    """
    
    @property
    def source_name(self) -> str:
        return "IBTrACS"
    
    def fetch(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        region: Optional[dict] = None,
    ) -> List[RawFile]:
        """Download IBTrACS NI basin CSV and parse to Parquet."""
        
        url = self.config.get("url",
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
        )
        
        csv_path = str(self.data_dir / "ibtracs_ni_raw.csv")
        parquet_path = self.config.get("output_file", str(self.data_dir / "ibtracs_ni.parquet"))
        
        # Download CSV
        download_file(url, csv_path)
        
        # Parse and clean
        df = self._parse_csv(csv_path)
        
        # Filter by date range
        if date_range:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            logger.info(f"Filtered to {date_range}: {len(df)} records, {df['storm_id'].nunique()} storms")
        
        # Filter by year range from config
        year_range = self.config.get("year_range", [2000, 2025])
        df = df[(df["timestamp"].dt.year >= year_range[0]) & (df["timestamp"].dt.year <= year_range[1])]
        logger.info(f"Year filter [{year_range[0]}-{year_range[1]}]: {len(df)} records, {df['storm_id'].nunique()} storms")
        
        # Compute derived features
        df = self._compute_derived(df)
        
        # Save to Parquet
        save_parquet(df, parquet_path)
        
        raw_file = RawFile(
            source=self.source_name,
            filepath=parquet_path,
            file_hash=file_sha256(parquet_path),
            byte_size=os.path.getsize(parquet_path),
            metadata={
                "num_storms": int(df["storm_id"].nunique()),
                "num_records": len(df),
                "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
                "imd_category_counts": df["imd_category_name"].value_counts().to_dict(),
            },
        )
        
        logger.info(
            f"IBTrACS ingested: {raw_file.metadata['num_storms']} storms, "
            f"{raw_file.metadata['num_records']} records"
        )
        
        return [raw_file]
    
    def validate(self, raw_file: RawFile) -> bool:
        """Validate the parsed Parquet file."""
        try:
            df = pd.read_parquet(raw_file.filepath)
            required_cols = ["storm_id", "timestamp", "lat", "lon", "max_wind_kt", "min_pressure_hpa"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return False
            if len(df) == 0:
                logger.error("Empty DataFrame")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _parse_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse raw IBTrACS CSV into a clean DataFrame.
        
        IBTrACS CSV has a header row + a units row (row 1) that we skip.
        Handles missing values coded as ' ' or empty strings.
        """
        logger.info(f"Parsing IBTrACS CSV: {csv_path}")
        
        # Read CSV, skip the units row (row index 1 after header)
        df = pd.read_csv(
            csv_path,
            skiprows=[1],  # skip units row
            na_values=[" ", "", "  ", "   "],
            low_memory=False,
        )
        
        # Rename to our standard column names
        col_map = {v: k for k, v in IBTRACS_COLUMNS.items() if v in df.columns}
        df = df.rename(columns=col_map)
        
        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "ISO_TIME" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
        
        # Parse numeric columns
        for col in ["lat", "lon", "max_wind", "min_pressure"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Standardize column names
        rename_map = {}
        if "max_wind" in df.columns:
            rename_map["max_wind"] = "max_wind_kt"
        if "min_pressure" in df.columns:
            rename_map["min_pressure"] = "min_pressure_hpa"
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Keep only needed columns
        keep_cols = [
            "storm_id", "name", "timestamp", "lat", "lon",
            "max_wind_kt", "min_pressure_hpa", "basin",
        ]
        # Add optional columns if present
        for opt_col in ["nature", "dist2land", "storm_speed", "storm_dir"]:
            if opt_col in df.columns:
                keep_cols.append(opt_col)
        
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()
        
        # Drop rows with no position
        df = df.dropna(subset=["lat", "lon", "timestamp"])
        
        # Sort by storm and time
        df = df.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
        
        logger.info(f"Parsed {len(df)} records for {df['storm_id'].nunique()} storms")
        return df
    
    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived fields per record:
        - IMD category index and name
        - Time since genesis (hours)
        - Year, month for stratification
        - Intensity change rates (causal: past-only)
        """
        # IMD category from wind speed
        df["imd_category_idx"] = df["max_wind_kt"].apply(
            lambda w: wind_to_imd_index(w) if pd.notna(w) else np.nan
        )
        df["imd_category_name"] = df["max_wind_kt"].apply(
            lambda w: wind_to_imd_category(w).name if pd.notna(w) else "Unknown"
        )
        df["imd_category_code"] = df["max_wind_kt"].apply(
            lambda w: wind_to_imd_category(w).code if pd.notna(w) else "UNK"
        )
        
        # Time since genesis (hours from first fix of each storm)
        genesis_times = df.groupby("storm_id")["timestamp"].transform("min")
        df["time_since_genesis_hours"] = (
            (df["timestamp"] - genesis_times).dt.total_seconds() / 3600.0
        )
        
        # Year and month for stratified splitting
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        
        # Intensity rate-of-change (CAUSAL: backward difference only)
        # ΔWind/Δt and ΔPressure/Δt over the previous interval
        df["delta_wind_6h"] = np.nan
        df["delta_pressure_6h"] = np.nan
        
        for sid, group in df.groupby("storm_id"):
            idx = group.index
            if len(idx) < 2:
                continue
            
            wind_vals = group["max_wind_kt"].values
            pres_vals = group["min_pressure_hpa"].values
            times = group["timestamp"].values
            
            for i in range(1, len(idx)):
                dt_hours = (times[i] - times[i-1]) / np.timedelta64(1, 'h')
                if dt_hours > 0:
                    df.loc[idx[i], "delta_wind_6h"] = (wind_vals[i] - wind_vals[i-1])
                    df.loc[idx[i], "delta_pressure_6h"] = (pres_vals[i] - pres_vals[i-1])
        
        # Rapid intensification flag: ≥30 kt increase in 24h
        # We compute this as a rolling sum of delta_wind over 24h-equivalent steps
        # For 3-hourly data, that's 8 steps
        df["is_rapid_intensification"] = False
        for sid, group in df.groupby("storm_id"):
            idx = group.index
            wind_vals = group["max_wind_kt"].values
            for i in range(8, len(idx)):
                wind_change_24h = wind_vals[i] - wind_vals[i - 8]
                if wind_change_24h >= 30:
                    df.loc[idx[i], "is_rapid_intensification"] = True
        
        # Storm maximum intensity (for stratification)
        max_wind_per_storm = df.groupby("storm_id")["max_wind_kt"].transform("max")
        df["storm_max_wind_kt"] = max_wind_per_storm
        df["storm_peak_category"] = max_wind_per_storm.apply(
            lambda w: wind_to_imd_category(w).code if pd.notna(w) else "UNK"
        )
        
        logger.info(
            f"Derived features computed. "
            f"RI events: {df['is_rapid_intensification'].sum()} records"
        )
        
        return df
