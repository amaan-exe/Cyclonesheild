"""
Cyclone Horizon — NASA GIBS Satellite Data Connector
Connects to NASA Global Imagery Browse Services (GIBS) for near-real-time and historical
Earth Observing System (EOS) satellite imagery:
- MODIS (Terra & Aqua) Corrected Reflectance True Color
- VIIRS (Suomi NPP & NOAA-20) 250m High-Resolution True Color
- MODIS Aqua False Color (Bands 7-2-1) for Convective Cloud / Storm Structure
- MODIS Terra Brightness Temperature (Band 31) for Deep Eyewall Convection
- IMERG GPM Near-Real-Time Precipitation Rate

Source: https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/gibs
License: NASA Open Data (Public Domain / Free and Open)
"""

import os
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.ingestion.base_connector import BaseConnector, RawFile
from src.utils.constants import NIO_BOUNDS
from src.utils.logging_config import get_logger

logger = get_logger("ingestion.gibs")

# NASA GIBS Endpoints
GIBS_WMS_EPSG4326_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_WMTS_EPSG3857_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"

# Curated satellite products relevant to tropical cyclone observation
GIBS_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "MODIS_Terra_CorrectedReflectance_TrueColor": {
        "id": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "name": "MODIS Terra True Color",
        "sensor": "MODIS (Terra Satellite)",
        "resolution": "250m",
        "format": "image/jpeg",
        "ext": "jpg",
        "tile_matrix_set": "GoogleMapsCompatible_Level9",
        "max_zoom": 9,
        "description": "Daily natural color imagery highlighting cloud bands, spiral structure, and eye formation.",
        "category": "True Color Visible",
    },
    "VIIRS_SNPP_CorrectedReflectance_TrueColor": {
        "id": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "name": "VIIRS SNPP True Color",
        "sensor": "VIIRS (Suomi NPP Satellite)",
        "resolution": "250m",
        "format": "image/jpeg",
        "ext": "jpg",
        "tile_matrix_set": "GoogleMapsCompatible_Level9",
        "max_zoom": 9,
        "description": "Ultra-sharp 250m true color imagery from NASA/NOAA Suomi NPP.",
        "category": "True Color Visible",
    },
    "MODIS_Aqua_CorrectedReflectance_Bands721": {
        "id": "MODIS_Aqua_CorrectedReflectance_Bands721",
        "name": "MODIS Aqua False Color (7-2-1)",
        "sensor": "MODIS (Aqua Satellite)",
        "resolution": "250m",
        "format": "image/jpeg",
        "ext": "jpg",
        "tile_matrix_set": "GoogleMapsCompatible_Level9",
        "max_zoom": 9,
        "description": "False color infrared composite separating high ice clouds, thunderstorm updrafts, and sea surface.",
        "category": "Convective Cloud Structure",
    },
    "MODIS_Terra_Brightness_Temp_Band31_Day": {
        "id": "MODIS_Terra_Brightness_Temp_Band31_Day",
        "name": "Thermal IR Brightness Temp (Band 31)",
        "sensor": "MODIS (Terra Satellite)",
        "resolution": "1km",
        "format": "image/png",
        "ext": "png",
        "tile_matrix_set": "GoogleMapsCompatible_Level7",
        "max_zoom": 7,
        "description": "Thermal infrared measuring cloud-top temperature to detect vigorous deep eyewall convection (< -70°C).",
        "category": "Thermal Infrared",
    },
    "IMERG_Precipitation_Rate": {
        "id": "IMERG_Precipitation_Rate",
        "name": "GPM IMERG Rain Rate",
        "sensor": "GPM Microwave/Radar Ensemble",
        "resolution": "0.1° (~10km)",
        "format": "image/png",
        "ext": "png",
        "tile_matrix_set": "GoogleMapsCompatible_Level6",
        "max_zoom": 6,
        "description": "Near-real-time rainfall intensity (mm/hr) capturing torrential spiral rainbands.",
        "category": "Precipitation & Microwave",
    }
}


class GibsConnector(BaseConnector):
    """
    Data connector for NASA Global Imagery Browse Services (GIBS).
    Supports WMS georeferenced bounding box snapshots and WMTS map tiling.
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "data/raw/gibs"):
        super().__init__(config or {}, data_dir=data_dir)
        self.user_agent = self.config.get("user_agent", "CycloneHorizon-DisasterShield/2.0 (Academic/DisasterRelief)")
        self.timeout = self.config.get("timeout_seconds", 12)

    @property
    def source_name(self) -> str:
        return "NASA_GIBS"

    def check_connection(self) -> Dict[str, Any]:
        """Verify direct handshake and latency with NASA GIBS Earthdata."""
        test_url = (
            f"{GIBS_WMS_EPSG4326_URL}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"
        )
        start_t = datetime.now()
        try:
            req = urllib.request.Request(
                test_url,
                headers={"User-Agent": self.user_agent, "Accept": "application/xml"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.status
                duration_ms = (datetime.now() - start_t).total_seconds() * 1000
                is_ok = status_code == 200
                return {
                    "online": is_ok,
                    "status_code": status_code,
                    "latency_ms": round(duration_ms, 1),
                    "endpoint": GIBS_WMS_EPSG4326_URL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "supported_layers": list(GIBS_PRODUCTS.keys()),
                }
        except Exception as e:
            logger.warning(f"GIBS connection check failed: {e}")
            duration_ms = (datetime.now() - start_t).total_seconds() * 1000
            return {
                "online": False,
                "status_code": 0,
                "error": str(e),
                "latency_ms": round(duration_ms, 1),
                "endpoint": GIBS_WMS_EPSG4326_URL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "supported_layers": list(GIBS_PRODUCTS.keys()),
            }

    def get_latest_available_date(self) -> str:
        """
        Return the latest imagery date string (YYYY-MM-DD).
        NASA GIBS imagery is updated continuously; yesterday or today is typically fully ready.
        """
        now_utc = datetime.now(timezone.utc)
        # If early in UTC morning (< 04:00 UTC), yesterday's composite is guaranteed complete
        if now_utc.hour < 4:
            target = now_utc - timedelta(days=1)
        else:
            target = now_utc
        return target.strftime("%Y-%m-%d")

    def build_wms_url(
        self,
        layer: str,
        date_str: str,
        bbox: Tuple[float, float, float, float],  # min_lat, min_lon, max_lat, max_lon
        width: int = 512,
        height: int = 512,
        image_format: str = "image/jpeg"
    ) -> str:
        """Build standard WMS 1.3.0 GetMap request URL for EPSG:4326."""
        min_lat, min_lon, max_lat, max_lon = bbox
        # In WMS 1.3.0 for EPSG:4326, the coordinate order is minLat,minLon,maxLat,maxLon
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
        url = (
            f"{GIBS_WMS_EPSG4326_URL}?"
            f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&"
            f"LAYERS={layer}&STYLES=&FORMAT={image_format}&"
            f"TRANSPARENT=FALSE&HEIGHT={height}&WIDTH={width}&"
            f"TIME={date_str}&CRS=EPSG:4326&BBOX={bbox_str}"
        )
        return url

    def build_wmts_tile_url(
        self,
        layer: str,
        date_str: str,
        z: int,
        x: int,
        y: int,
    ) -> str:
        """Build RESTful WMTS tile URL for Web Mercator EPSG:3857."""
        prod = GIBS_PRODUCTS.get(layer, GIBS_PRODUCTS["MODIS_Terra_CorrectedReflectance_TrueColor"])
        tms = prod["tile_matrix_set"]
        ext = prod["ext"]
        return f"{GIBS_WMTS_EPSG3857_URL}/{layer}/default/{date_str}/{tms}/{z}/{y}/{x}.{ext}"

    def fetch_snapshot(
        self,
        layer: str = "MODIS_Terra_CorrectedReflectance_TrueColor",
        date_str: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        width: int = 512,
        height: int = 512,
    ) -> RawFile:
        """
        Download a single georeferenced satellite snapshot from NASA GIBS.
        
        Parameters
        ----------
        layer : str
            GIBS product identifier.
        date_str : str (optional)
            Date in YYYY-MM-DD. Defaults to latest available date.
        bbox : (min_lat, min_lon, max_lat, max_lon)
            Bounding box in EPSG:4326 degrees. Defaults to North Indian Ocean basin.
        width : int
            Image pixel width (default: 512).
        height : int
            Image pixel height (default: 512).
            
        Returns
        -------
        RawFile metadata record with local filepath and SHA-256 hash.
        """
        if layer not in GIBS_PRODUCTS:
            layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
            
        prod = GIBS_PRODUCTS[layer]
        image_format = prod["format"]
        ext = prod["ext"]
        
        if not date_str:
            date_str = self.get_latest_available_date()
            
        if not bbox:
            bbox = (
                NIO_BOUNDS["lat_min"],
                NIO_BOUNDS["lon_min"],
                NIO_BOUNDS["lat_max"],
                NIO_BOUNDS["lon_max"],
            )

        url = self.build_wms_url(layer, date_str, bbox, width=width, height=height, image_format=image_format)
        
        # Consistent filename schema
        b_tag = f"{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
        filename = f"{layer}_{date_str}_{b_tag}_{width}x{height}.{ext}"
        target_path = self.data_dir / filename

        # Idempotent caching: check if already downloaded and valid
        if target_path.exists() and target_path.stat().st_size > 1024:
            with open(target_path, "rb") as f:
                content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()
            logger.info(f"Loaded cached GIBS snapshot: {target_path.name}")
            return RawFile(
                source=self.source_name,
                filepath=str(target_path),
                timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                file_hash=file_hash,
                byte_size=len(content),
                metadata={
                    "layer": layer,
                    "date": date_str,
                    "bbox": list(bbox),
                    "dimensions": [width, height],
                    "sensor": prod["sensor"],
                    "category": prod["category"],
                    "url": url,
                    "cached": True
                }
            )

        # Download from NASA GIBS
        logger.info(f"Fetching NASA GIBS live snapshot: {layer} for {date_str} over {bbox}...")
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
        except urllib.error.HTTPError as e:
            # If latest date had no imagery yet (e.g., today pre-dawn), fallback to previous day
            logger.warning(f"GIBS HTTP {e.code} for date {date_str}. Attempting fallback to previous day...")
            prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            fallback_url = self.build_wms_url(layer, prev_date, bbox, width=width, height=height, image_format=image_format)
            req_fb = urllib.request.Request(fallback_url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req_fb, timeout=self.timeout) as resp:
                content = resp.read()
            date_str = prev_date
            filename = f"{layer}_{date_str}_{b_tag}_{width}x{height}.{ext}"
            target_path = self.data_dir / filename

        # Write to file
        with open(target_path, "wb") as f:
            f.write(content)
            
        file_hash = hashlib.sha256(content).hexdigest()
        byte_size = len(content)
        
        raw_file = RawFile(
            source=self.source_name,
            filepath=str(target_path),
            timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
            file_hash=file_hash,
            byte_size=byte_size,
            metadata={
                "layer": layer,
                "date": date_str,
                "bbox": list(bbox),
                "dimensions": [width, height],
                "sensor": prod["sensor"],
                "category": prod["category"],
                "url": url,
                "cached": False
            }
        )
        
        if not self.validate(raw_file):
            logger.error(f"GIBS file validation failed for {target_path}")
            raise ValueError(f"Downloaded GIBS imagery failed validation: {target_path}")

        logger.info(f"NASA GIBS snapshot saved: {target_path.name} ({byte_size} bytes)")
        return raw_file

    def fetch(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        region: Optional[dict] = None,
    ) -> List[RawFile]:
        """
        Batch fetch satellite scenes across date range and region.
        
        Parameters
        ----------
        date_range : (start_date, end_date) in YYYY-MM-DD, or None for latest date
        region : dict with lat_min, lat_max, lon_min, lon_max, or None for NIO_BOUNDS
        
        Returns
        -------
        List of RawFile metadata records
        """
        if region:
            bbox = (
                float(region.get("lat_min", NIO_BOUNDS["lat_min"])),
                float(region.get("lon_min", NIO_BOUNDS["lon_min"])),
                float(region.get("lat_max", NIO_BOUNDS["lat_max"])),
                float(region.get("lon_max", NIO_BOUNDS["lon_max"])),
            )
        else:
            bbox = (
                NIO_BOUNDS["lat_min"],
                NIO_BOUNDS["lon_min"],
                NIO_BOUNDS["lat_max"],
                NIO_BOUNDS["lon_max"],
            )

        # Determine dates
        if date_range:
            start_dt = datetime.strptime(date_range[0], "%Y-%m-%d")
            end_dt = datetime.strptime(date_range[1], "%Y-%m-%d")
            # Cap at max 5 dates to avoid excessive network calls in tests
            dates = []
            curr = start_dt
            while curr <= end_dt and len(dates) < 5:
                dates.append(curr.strftime("%Y-%m-%d"))
                curr += timedelta(days=1)
        else:
            dates = [self.get_latest_available_date()]

        fetched: List[RawFile] = []
        # Fetch primary True Color and False Color for each date
        layers_to_fetch = [
            "MODIS_Terra_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_Bands721",
        ]
        
        for d in dates:
            for lyr in layers_to_fetch:
                try:
                    rf = self.fetch_snapshot(layer=lyr, date_str=d, bbox=bbox)
                    fetched.append(rf)
                except Exception as e:
                    logger.warning(f"Failed to fetch {lyr} for {d}: {e}")

        return fetched

    def validate(self, raw_file: RawFile) -> bool:
        """
        Validate downloaded satellite imagery file:
        - Must exist on disk
        - File size > 1024 bytes (not an error HTML/XML page or empty response)
        - Must match JPEG or PNG binary magic header
        """
        p = Path(raw_file.filepath)
        if not p.exists():
            return False
            
        size = p.stat().st_size
        if size < 1024:
            return False

        try:
            with open(p, "rb") as f:
                header = f.read(16)
            # JPEG magic bytes: 0xFF 0xD8
            is_jpeg = header[:2] == b"\xff\xd8"
            # PNG magic bytes: 0x89 'P' 'N' 'G'
            is_png = header[:4] == b"\x89PNG"
            return is_jpeg or is_png
        except Exception:
            return False
