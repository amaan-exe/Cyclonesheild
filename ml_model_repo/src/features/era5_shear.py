"""
Cyclone Horizon — ERA5 Deep-Layer Vertical Wind Shear Utility
Computes 850-200 hPa vertical wind shear from ERA5 reanalysis data.

If ERA5 data is available locally, samples real deep-layer shear at (lat, lon, timestamp).
Otherwise, falls back to a climatological proxy formula and CLEARLY FLAGS the source.

Shear Magnitude = sqrt((U200-U850)² + (V200-V850)²) in knots.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("features.era5_shear")

# Try importing ERA5 dependencies
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False


class ERA5ShearSampler:
    """
    Samples 850-200 hPa deep-layer vertical wind shear from ERA5 pressure-level data.

    Data directory structure expected:
        era5_data/
            era5_shear_YYYY.nc   — Monthly or yearly NetCDF with U/V at 200 & 850 hPa.

    If no ERA5 data is found, falls back to a latitude-based climatological proxy
    and flags the source as 'CLIMATOLOGICAL_PROXY'.
    """

    # North Indian Ocean monthly climatological shear (kt) by latitude band.
    # Source: DeMaria & Kaplan (1994) SHIPS climatology adapted for NIO.
    # This is still a proxy, but much better than a single sine formula.
    CLIMO_SHEAR_BY_MONTH = {
        # month: (base_kt, lat_coeff, lat2_coeff) → shear ≈ base + coeff*lat + coeff2*lat²
        1:  (12.0, 0.8, -0.01),   # January  — strong upper-level westerlies
        2:  (11.0, 0.7, -0.01),
        3:  (10.0, 0.5, -0.005),
        4:  (8.0,  0.3, -0.003),  # Pre-monsoon — shear weakens
        5:  (9.0,  0.4, -0.005),
        6:  (15.0, 0.6, -0.008),  # Monsoon onset — strong easterly shear
        7:  (18.0, 0.7, -0.01),   # Peak monsoon
        8:  (17.0, 0.6, -0.01),
        9:  (14.0, 0.5, -0.008),  # Monsoon withdrawal
        10: (10.0, 0.4, -0.005),  # Post-monsoon — favorable for cyclogenesis
        11: (9.0,  0.5, -0.005),
        12: (11.0, 0.7, -0.01),
    }

    def __init__(self, era5_dir: str = "datasets/ERA5_SHEAR"):
        self.era5_dir = Path(era5_dir)
        self.cache = {}
        self.has_era5_data = False

        if self.era5_dir.exists() and HAS_XARRAY:
            nc_files = list(self.era5_dir.glob("era5_shear_*.nc"))
            if nc_files:
                self.has_era5_data = True
                logger.info(f"ERA5 shear data found: {len(nc_files)} files in {self.era5_dir}")
            else:
                logger.warning(f"ERA5 directory exists but no NetCDF files found. Using climatological proxy.")
        else:
            logger.warning(
                f"ERA5 shear data not available at {self.era5_dir}. "
                f"Using climatological proxy. To use real shear, download ERA5 850/200 hPa "
                f"wind data using `download_era5_shear()` with CDS API credentials."
            )

    def get_shear(
        self,
        lat: float,
        lon: float,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
    ) -> Tuple[float, str]:
        """
        Sample vertical wind shear magnitude at a given location and time.

        Returns
        -------
        (shear_kt, source) : tuple
            shear_kt: deep-layer shear magnitude in knots
            source: "ERA5" or "CLIMATOLOGICAL_PROXY"
        """
        if self.has_era5_data:
            shear_val = self._sample_era5(lat, lon, year, month, day, hour)
            if shear_val is not None:
                return round(shear_val, 1), "ERA5"

        # Fallback: improved climatological proxy
        shear_val = self._climatological_proxy(lat, lon, month)
        return round(shear_val, 1), "CLIMATOLOGICAL_PROXY"

    def _sample_era5(
        self, lat: float, lon: float, year: int, month: int, day: int, hour: int
    ) -> Optional[float]:
        """Sample shear from ERA5 NetCDF if available."""
        if not HAS_XARRAY:
            return None

        cache_key = f"{year}_{month:02d}"
        if cache_key not in self.cache:
            # Try yearly file first, then monthly
            for pattern in [f"era5_shear_{year}.nc", f"era5_shear_{year}_{month:02d}.nc"]:
                fpath = self.era5_dir / pattern
                if fpath.exists():
                    try:
                        ds = xr.open_dataset(fpath)
                        self.cache[cache_key] = ds
                        break
                    except Exception as e:
                        logger.warning(f"Failed to open ERA5 file {fpath}: {e}")
                        return None
            else:
                return None

        ds = self.cache[cache_key]
        try:
            # Select nearest grid point and time
            target_time = pd.Timestamp(year=year, month=month, day=day, hour=hour)

            # Try to select by time (nearest)
            if "time" in ds.dims:
                ds_sel = ds.sel(
                    latitude=lat, longitude=lon, time=target_time,
                    method="nearest"
                )
            else:
                ds_sel = ds.sel(latitude=lat, longitude=lon, method="nearest")

            # Compute deep-layer shear: sqrt((u200-u850)² + (v200-v850)²)
            # Variable names depend on ERA5 format
            if "u200" in ds_sel and "u850" in ds_sel:
                du = float(ds_sel["u200"]) - float(ds_sel["u850"])
                dv = float(ds_sel["v200"]) - float(ds_sel["v850"])
            elif "u_component_of_wind" in ds_sel:
                # Pressure level format
                u200 = float(ds_sel["u_component_of_wind"].sel(pressure_level=200, method="nearest"))
                u850 = float(ds_sel["u_component_of_wind"].sel(pressure_level=850, method="nearest"))
                v200 = float(ds_sel["v_component_of_wind"].sel(pressure_level=200, method="nearest"))
                v850 = float(ds_sel["v_component_of_wind"].sel(pressure_level=850, method="nearest"))
                du = u200 - u850
                dv = v200 - v850
            else:
                logger.warning(f"Unrecognized ERA5 variable names: {list(ds_sel.data_vars)}")
                return None

            # ERA5 winds are in m/s, convert to knots (1 m/s = 1.94384 kt)
            shear_ms = np.sqrt(du ** 2 + dv ** 2)
            shear_kt = shear_ms * 1.94384
            return float(shear_kt)

        except Exception as e:
            logger.debug(f"ERA5 sampling failed at ({lat}, {lon}, {year}-{month}-{day}): {e}")
            return None

    def _climatological_proxy(self, lat: float, lon: float, month: int) -> float:
        """
        Improved climatological proxy for NIO deep-layer shear.
        Uses monthly-varying coefficients instead of a single sine formula.
        
        This is clearly labeled as a PROXY and should not be presented as real data.
        """
        month = max(1, min(12, month))
        base, lat_coeff, lat2_coeff = self.CLIMO_SHEAR_BY_MONTH[month]

        abs_lat = abs(lat)
        shear = base + lat_coeff * abs_lat + lat2_coeff * abs_lat ** 2

        # Arabian Sea generally has higher shear than Bay of Bengal
        if lon < 77.0:
            shear *= 1.15  # Arabian Sea shear enhancement

        # Add small deterministic perturbation based on lon for spatial variation
        shear += 0.5 * np.sin(np.radians(lon * 3))

        return max(3.0, min(45.0, shear))  # Physical bounds


def download_era5_shear(
    years: list = None,
    months: list = None,
    output_dir: str = "datasets/ERA5_SHEAR",
    area: list = None,
):
    """
    Downloads ERA5 850 hPa and 200 hPa U/V wind components from CDS API.

    Requires cdsapi package and a valid .cdsapirc configuration file.
    See: https://cds.climate.copernicus.eu/api-how-to

    Parameters
    ----------
    years : list of int
        Years to download (e.g., [2020, 2021, 2022])
    months : list of int
        Months to download (1-12)
    output_dir : str
        Directory to save NetCDF files
    area : list of float
        Bounding box [north, west, south, east] in degrees.
        Default: North Indian Ocean [30, 50, 0, 100]
    """
    if not HAS_CDSAPI:
        logger.error("cdsapi package not installed. Run: pip install cdsapi")
        return

    if years is None:
        years = list(range(1980, 2025))
    if months is None:
        months = list(range(1, 13))
    if area is None:
        area = [30, 50, 0, 100]  # North Indian Ocean

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    for year in years:
        for month in months:
            fname = out_path / f"era5_shear_{year}_{month:02d}.nc"
            if fname.exists():
                logger.info(f"Skipping {fname.name} (already exists)")
                continue

            logger.info(f"Downloading ERA5 shear data for {year}-{month:02d}...")
            try:
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": [
                            "u_component_of_wind",
                            "v_component_of_wind",
                        ],
                        "pressure_level": ["200", "850"],
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area,
                    },
                    str(fname),
                )
                logger.info(f"Saved: {fname.name}")
            except Exception as e:
                logger.error(f"Failed to download ERA5 for {year}-{month:02d}: {e}")


if __name__ == "__main__":
    # Quick test of the climatological proxy
    sampler = ERA5ShearSampler()
    for lat in [5, 10, 15, 20, 25]:
        for month in [4, 7, 10]:
            val, src = sampler.get_shear(lat, 85.0, 2023, month, 15)
            print(f"  Lat={lat:2d}°N  Month={month:2d}  Shear={val:5.1f} kt  Source={src}")
