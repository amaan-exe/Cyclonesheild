"""Train the satellite vortex detector from HURSAT NIO NetCDF labels."""

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detection.vortex_detector import CycloneVortexDetector


class HursatFrames(TorchDataset):
    def __init__(self, archives, max_samples):
        self.samples = []
        for archive_path in archives:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive.getmembers():
                    if member.name.endswith(".nc"):
                        self.samples.append((archive_path, member.name))
                        if len(self.samples) >= max_samples:
                            return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        archive_path, member_name = self.samples[index]
        with tarfile.open(archive_path, "r:gz") as archive:
            raw = archive.extractfile(member_name).read()
        with Dataset("frame.nc", memory=raw) as dataset:
            ir = np.asarray(dataset.variables["IRWIN"][0], dtype=np.float32)
            if "VSCHN" in dataset.variables:
                water_vapor = np.asarray(dataset.variables["VSCHN"][0], dtype=np.float32)
            else:
                water_vapor = np.zeros_like(ir, dtype=np.float32)
            latitudes = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
            longitudes = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
            center_lat = float(dataset.variables["CentLat"][0])
            center_lon = float(dataset.variables["CentLon"][0])

        if not np.isfinite(center_lat + center_lon) or center_lat < -10 or center_lat > 40:
            raise ValueError("invalid HURSAT center label")
        ir = np.nan_to_num(ir, nan=300.0, posinf=330.0, neginf=180.0)
        water_vapor = np.nan_to_num(water_vapor, nan=0.0)
        image = np.stack([
            np.clip((330.0 - ir) / 150.0, 0.0, 1.0),
            np.clip(water_vapor / 330.0, 0.0, 1.0),
            np.clip((330.0 - ir) / 150.0, 0.0, 1.0),
        ])
        image = torch.from_numpy(image).unsqueeze(0)
        image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)

        x = float(np.interp(center_lon, longitudes, np.arange(len(longitudes)))) * 256.0 / len(longitudes)
        y = float(np.interp(center_lat, latitudes, np.arange(len(latitudes)))) * 256.0 / len(latitudes)
        target = torch.zeros(1, 64, 64)
        cx, cy = x / 4.0, y / 4.0
        grid_y, grid_x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        target[0] = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * 2.0 ** 2))
        return image, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\sih70\ML\datasets\HURSAT_NIO")
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    archives = sorted(Path(args.data_root).rglob("*.tar.gz"))
    if not archives:
        raise RuntimeError(f"No HURSAT archives found under {args.data_root}")
    dataset = HursatFrames(archives, args.max_samples)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"HURSAT archives: {len(archives)} | labeled frames: {len(dataset)}")

    model = CycloneVortexDetector(in_channels=3, base_channels=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, target in loader:
            optimizer.zero_grad()
            output = model(images)
            loss = F.mse_loss(output["heatmap"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.6f}")

    output_path = PROJECT_ROOT / "outputs" / "checkpoints" / "vortex_detector_model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"saved={output_path}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")


if __name__ == "__main__":
    main()