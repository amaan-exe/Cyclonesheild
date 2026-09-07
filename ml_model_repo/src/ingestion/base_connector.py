"""
Cyclone Horizon — Abstract Base Connector
Every data source connector extends this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class RawFile:
    """Metadata for one ingested raw file."""
    source: str
    filepath: str
    timestamp: Optional[datetime] = None
    file_hash: Optional[str] = None
    byte_size: int = 0
    storm_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """
    Abstract connector interface. Each data source implements:
      fetch(date_range, region) -> list[RawFile]
    
    Connectors must:
    - Be independently testable with small date ranges
    - Log every fetch (source, timestamp, file hash, byte size)
    - Never silently skip failures (log to failed_downloads)
    - Support retry and idempotent re-runs
    """
    
    def __init__(self, config: dict, data_dir: str = "data/raw"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        region: Optional[dict] = None,
    ) -> List[RawFile]:
        """
        Download/fetch raw data for the given date range and region.
        
        Parameters
        ----------
        date_range : (start_date, end_date) as ISO strings, or None for all
        region : dict with lat_min, lat_max, lon_min, lon_max, or None
        
        Returns
        -------
        List of RawFile descriptors for successfully fetched files
        """
        ...
    
    @abstractmethod
    def validate(self, raw_file: RawFile) -> bool:
        """Basic validation that the fetched file is readable and non-corrupt."""
        ...
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source identifier (e.g., 'IBTrACS', 'ERA5')."""
        ...
