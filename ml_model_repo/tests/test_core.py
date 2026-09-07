"""
Cyclone Horizon — Unit Tests for Core Utilities
Tests: geo calculations, constants, physical checks.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.constants import (
    wind_to_imd_category, wind_to_imd_index, t_number_to_wind_kt,
    IMD_CATEGORIES,
)
from src.utils.geo import (
    haversine_distance, bearing, destination_point,
    bounding_box, compute_storm_motion,
)


class TestIMDCategories:
    """Tests for IMD intensity scale mapping."""
    
    def test_low_pressure_area(self):
        cat = wind_to_imd_category(10)
        assert cat.code == "LPA"
        assert cat.index == 0
    
    def test_depression(self):
        cat = wind_to_imd_category(20)
        assert cat.code == "D"
    
    def test_cyclonic_storm(self):
        cat = wind_to_imd_category(40)
        assert cat.code == "CS"
    
    def test_vscs(self):
        cat = wind_to_imd_category(75)
        assert cat.code == "VSCS"
    
    def test_super_cyclonic(self):
        cat = wind_to_imd_category(130)
        assert cat.code == "SuCS"
        assert cat.index == 7
    
    def test_boundary_values(self):
        """Test values at category boundaries."""
        assert wind_to_imd_index(17) == 1   # exactly at Depression threshold
        assert wind_to_imd_index(33) == 2   # top of Deep Depression
        assert wind_to_imd_index(34) == 3   # bottom of CS
    
    def test_all_categories_present(self):
        assert len(IMD_CATEGORIES) == 8


class TestDvorakMapping:
    """Tests for Dvorak T-number to wind mapping."""
    
    def test_known_values(self):
        assert t_number_to_wind_kt(1.0) == 25
        assert t_number_to_wind_kt(5.0) == 90
        assert t_number_to_wind_kt(8.0) == 170
    
    def test_interpolation(self):
        """Interpolated T-numbers should fall between table values."""
        wind_35 = t_number_to_wind_kt(3.5)
        wind_30 = t_number_to_wind_kt(3.0)
        wind_40 = t_number_to_wind_kt(4.0)
        assert wind_30 <= wind_35 <= wind_40
    
    def test_monotonic(self):
        """Higher T-number should give higher wind."""
        t_values = np.arange(1.0, 8.1, 0.5)
        winds = [t_number_to_wind_kt(t) for t in t_values]
        for i in range(1, len(winds)):
            assert winds[i] >= winds[i-1], f"Non-monotonic at T={t_values[i]}"


class TestGeoUtilities:
    """Tests for geospatial calculations."""
    
    def test_haversine_zero_distance(self):
        dist = haversine_distance(20.0, 86.0, 20.0, 86.0)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_haversine_known_distance(self):
        # Mumbai (19.08°N, 72.88°E) to Delhi (28.61°N, 77.21°E) ≈ 1150 km
        dist = haversine_distance(19.08, 72.88, 28.61, 77.21)
        assert 1100 < dist < 1200
    
    def test_haversine_symmetry(self):
        d1 = haversine_distance(20, 80, 25, 85)
        d2 = haversine_distance(25, 85, 20, 80)
        assert d1 == pytest.approx(d2, rel=1e-6)
    
    def test_bearing_north(self):
        brng = bearing(10.0, 80.0, 15.0, 80.0)  # due north
        assert brng == pytest.approx(0.0, abs=1.0)
    
    def test_bearing_east(self):
        brng = bearing(10.0, 80.0, 10.0, 85.0)  # due east
        assert 85 < brng < 95
    
    def test_destination_roundtrip(self):
        """Go to destination, compute distance — should match original."""
        lat1, lon1 = 20.0, 86.0
        dist = 100.0
        brng = 45.0  # NE
        
        lat2, lon2 = destination_point(lat1, lon1, brng, dist)
        actual_dist = haversine_distance(lat1, lon1, lat2, lon2)
        
        assert actual_dist == pytest.approx(dist, rel=0.01)
    
    def test_bounding_box(self):
        lat, lon = 20.0, 86.0
        radius = 300.0  # km
        
        lat_min, lat_max, lon_min, lon_max = bounding_box(lat, lon, radius)
        
        assert lat_min < lat < lat_max
        assert lon_min < lon < lon_max
        # Box should be roughly 600km across in latitude
        lat_span_km = (lat_max - lat_min) * 111
        assert 550 < lat_span_km < 650
    
    def test_storm_motion_first_is_nan(self):
        """First position has no predecessor, so motion should be NaN."""
        lats = np.array([10.0, 11.0, 12.0])
        lons = np.array([80.0, 80.5, 81.0])
        times = np.array([0.0, 3.0, 6.0])  # hours
        
        speeds, bearings = compute_storm_motion(lats, lons, times)
        
        assert np.isnan(speeds[0])
        assert np.isnan(bearings[0])
        assert not np.isnan(speeds[1])
        assert not np.isnan(bearings[1])
    
    def test_storm_motion_positive_speed(self):
        """Moving storm should have positive speed."""
        lats = np.array([10.0, 11.0, 12.0, 13.0])
        lons = np.array([80.0, 80.0, 80.0, 80.0])
        times = np.array([0.0, 3.0, 6.0, 9.0])
        
        speeds, _ = compute_storm_motion(lats, lons, times)
        
        for s in speeds[1:]:
            assert s > 0


class TestPhysicalChecks:
    """Tests for QC validation."""
    
    def test_wind_bounds_pass(self):
        from src.validation.physical_checks import check_wind_bounds
        df = pd.DataFrame({"max_wind_kt": [30, 60, 90, 120]})
        assert check_wind_bounds(df).all()
    
    def test_wind_bounds_fail(self):
        from src.validation.physical_checks import check_wind_bounds
        df = pd.DataFrame({"max_wind_kt": [30, 60, 300]})  # 300 kt is impossible
        assert not check_wind_bounds(df).all()
    
    def test_pressure_bounds_pass(self):
        from src.validation.physical_checks import check_pressure_bounds
        df = pd.DataFrame({"min_pressure_hpa": [990, 950, 920]})
        assert check_pressure_bounds(df).all()
    
    def test_pressure_bounds_fail(self):
        from src.validation.physical_checks import check_pressure_bounds
        df = pd.DataFrame({"min_pressure_hpa": [990, 800]})  # 800 hPa impossible
        assert not check_pressure_bounds(df).all()
    
    def test_basin_bounds_nio(self):
        from src.validation.physical_checks import check_basin_bounds
        df = pd.DataFrame({
            "lat": [15.0, 20.0, 35.0],   # 35° is outside NIO
            "lon": [85.0, 90.0, 85.0],
        })
        result = check_basin_bounds(df)
        assert result.iloc[0] == True
        assert result.iloc[2] == False  # outside NIO
    
    def test_nan_wind_passes(self):
        """NaN wind speed should not be flagged as invalid."""
        from src.validation.physical_checks import check_wind_bounds
        df = pd.DataFrame({"max_wind_kt": [30, np.nan, 60]})
        assert check_wind_bounds(df).all()


class TestLeakageCheck:
    """Tests for data leakage detection."""
    
    def test_first_fix_delta_is_nan(self):
        """First fix of a storm should have NaN delta_wind."""
        from src.features.leakage_check import check_causal_feature
        
        df = pd.DataFrame({
            "storm_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 03:00", "2020-01-01 06:00"]),
            "max_wind_kt": [30, 40, 50],
            "min_pressure_hpa": [990, 985, 980],
            "delta_wind_6h": [np.nan, 10, 10],  # first is NaN — correct
        })
        
        passes, violations = check_causal_feature(df, "delta_wind_6h", n_samples=10)
        assert passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
