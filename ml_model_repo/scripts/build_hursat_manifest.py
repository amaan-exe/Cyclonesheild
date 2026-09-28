"""Build a labeled HURSAT NIO frame manifest without extracting every archive."""

import csv
import io
import tarfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

DEFAULT_ROOT = Path(r"D:\sih70\ML\datasets\HURSAT_NIO")
DEFAULT_OUTPUT = Path(r"D:\sih70\ML\datasets\hursat_nio_manifest.csv")

FIELDS = [
    "archive_path", "member_path", "storm_id", "storm_name", "timestamp",
    "center_lat", "center_lon", "wind_speed_kt", "central_pressure_hpa",
    "image_variable", "image_shape", "image_units", "latitude_variable",
    "longitude_variable", "latitude_units", "longitude_units",
    "calibration_source", "satellite_source",
]


def scalar(dataset, name, default=None):
    if name not in dataset.variables:
        return default
    value = np.asarray(dataset.variables[name][:]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        value = value.item()
    except ValueError:
        pass
    return value


def text(value):
    return "" if value is None else str(value).strip()


def main():
    root = Path(DEFAULT_ROOT)
    output = Path(DEFAULT_OUTPUT)
    archives = sorted(root.rglob("*.tar.gz"))
    rows = []
    skipped = 0

    for archive_path in archives:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".nc"):
                    continue
                try:
                    raw = archive.extractfile(member).read()
                    with Dataset("frame.nc", memory=raw) as dataset:
                        center_lat = float(scalar(dataset, "CentLat", np.nan))
                        center_lon = float(scalar(dataset, "CentLon", np.nan))
                        if not np.isfinite(center_lat) or not np.isfinite(center_lon):
                            skipped += 1
                            continue
                        timestamp = scalar(dataset, "htime", "")
                        wind = float(scalar(dataset, "WindSpd", np.nan))
                        pressure = float(scalar(dataset, "CentPrs", np.nan))
                        image = dataset.variables.get("IRWIN")
                        if image is None:
                            skipped += 1
                            continue
                        image_shape = "x".join(str(x) for x in image.shape)
                        rows.append({
                            "archive_path": str(archive_path),
                            "member_path": member.name,
                            "storm_id": text(scalar(dataset, "sid", dataset.getncattr("TC_id", ""))),
                            "storm_name": text(dataset.getncattr("TC_name", "")),
                            "timestamp": text(timestamp),
                            "center_lat": center_lat,
                            "center_lon": center_lon,
                            "wind_speed_kt": wind if np.isfinite(wind) and wind >= 0 else "",
                            "central_pressure_hpa": pressure if np.isfinite(pressure) and pressure > 0 else "",
                            "image_variable": "IRWIN",
                            "image_shape": image_shape,
                            "image_units": text(getattr(image, "units", "K")),
                            "latitude_variable": "lat",
                            "longitude_variable": "lon",
                            "latitude_units": text(getattr(dataset.variables["lat"], "units", "degrees_north")),
                            "longitude_units": text(getattr(dataset.variables["lon"], "units", "degrees_east")),
                            "calibration_source": "HURSAT-B1 calibrated IRWIN brightness temperature",
                            "satellite_source": text(dataset.getncattr("satellite", dataset.getncattr("source", "HURSAT-B1"))),
                        })
                except Exception as exc:
                    skipped += 1
                    print(f"skip {archive_path.name}:{member.name}: {exc}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"archives={len(archives)}")
    print(f"valid_frames={len(rows)}")
    print(f"skipped_frames={skipped}")
    print(f"storms={len({row['storm_id'] for row in rows})}")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
