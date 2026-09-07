"""
Cyclone Horizon — Track Visualization
Predicted track + uncertainty cone + actual track on NIO map.
This is the "money shot" for the demo.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.utils.constants import IMD_CATEGORIES, wind_to_imd_category, KT_TO_KPH
from src.utils.geo import destination_point
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger("visualization.track_plot")

# IMD category → color mapping
CATEGORY_COLORS = {
    "LPA":  "#87CEEB",  # light blue
    "D":    "#4169E1",  # royal blue
    "DD":   "#0000CD",  # medium blue
    "CS":   "#FFD700",  # gold
    "SCS":  "#FFA500",  # orange
    "VSCS": "#FF4500",  # red-orange
    "ESCS": "#FF0000",  # red
    "SuCS": "#8B0000",  # dark red
    "UNK":  "#808080",  # gray
}


def plot_track_prediction(
    past_track: List[Dict],
    predictions: Dict,
    actual_future: Optional[List[Dict]] = None,
    storm_name: str = "Cyclone",
    output_path: str = "outputs/visualizations/track_prediction.png",
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """
    Plot predicted track with uncertainty cone, past track, and optionally actual future.
    
    Parameters
    ----------
    past_track : list of {lat, lon, wind_kt, timestamp, imd_code}
    predictions : dict with 'track_points' and 'predictions'
    actual_future : ground truth future track (demo mode)
    storm_name : name for the title
    output_path : save path
    
    Returns
    -------
    str : path to saved figure
    """
    ensure_dir(str(Path(output_path).parent))
    
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0F172A")
    ax.set_facecolor("#1E293B")
    
    # Try to use Cartopy for map projection
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # Get map extent from data
        all_lats = [p["lat"] for p in past_track]
        all_lons = [p["lon"] for p in past_track]
        
        if predictions.get("track_points"):
            all_lats += [p["lat"] for p in predictions["track_points"]]
            all_lons += [p["lon"] for p in predictions["track_points"]]
        
        padding = 5
        extent = [
            min(all_lons) - padding, max(all_lons) + padding,
            min(all_lats) - padding, max(all_lats) + padding,
        ]
        
        plt.close(fig)
        fig = plt.figure(figsize=figsize, facecolor="#0F172A")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("#1E293B")
        
        # Map features
        ax.add_feature(cfeature.LAND, facecolor="#2D3748", edgecolor="#4A5568", linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#1E293B")
        ax.add_feature(cfeature.COASTLINE, edgecolor="#4A5568", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, edgecolor="#4A5568", linewidth=0.3, linestyle=":")
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#4A5568", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"color": "#94A3B8", "fontsize": 8}
        gl.ylabel_style = {"color": "#94A3B8", "fontsize": 8}
        
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        logger.warning("Cartopy not installed — plotting without map projection")
    
    # ---- Plot past track ----
    past_lats = [p["lat"] for p in past_track]
    past_lons = [p["lon"] for p in past_track]
    past_colors = [CATEGORY_COLORS.get(p.get("imd_code", "UNK"), "#808080") for p in past_track]
    
    # Track line
    ax.plot(past_lons, past_lats, color="#E2E8F0", linewidth=2, alpha=0.6,
            transform=ccrs.PlateCarree() if has_cartopy else None, label="_nolegend_")
    
    # Category-colored dots
    for i, (lat, lon, color) in enumerate(zip(past_lats, past_lons, past_colors)):
        size = 30 if i == len(past_lats) - 1 else 15
        kwargs = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        ax.scatter(lon, lat, c=color, s=size, zorder=5, edgecolors="white", linewidths=0.5, **kwargs)
    
    # Current position marker
    if past_lats:
        kwargs = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        ax.scatter(past_lons[-1], past_lats[-1], c="#FFFFFF", s=120, zorder=10,
                  marker="*", edgecolors="#FF4500", linewidths=1.5,
                  label="Current Position", **kwargs)
    
    # ---- Plot uncertainty cone ----
    pred_points = predictions.get("track_points", [])
    if pred_points:
        pred_lats = [p["lat"] for p in pred_points]
        pred_lons = [p["lon"] for p in pred_points]
        
        # Predicted track line (dashed)
        kwargs = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        ax.plot(pred_lons, pred_lats, color="#60A5FA", linewidth=2, linestyle="--",
                alpha=0.8, label="Predicted Track", **kwargs)
        
        # Uncertainty cone (widening with lead time)
        for i, pred in enumerate(predictions.get("predictions", [])):
            lead_h = pred.get("lead_h", (i+1)*3)
            lat = pred["lat"]
            lon = pred["lon"]
            
            # Uncertainty grows with lead time
            unc_km = pred.get("lat_uncertainty_km", 30 + 10 * (lead_h / 6))
            
            # Draw uncertainty circle
            circle_lats = []
            circle_lons = []
            for angle in range(0, 361, 10):
                clat, clon = destination_point(lat, lon, angle, unc_km)
                circle_lats.append(clat)
                circle_lons.append(clon)
            
            alpha = max(0.05, 0.3 - 0.03 * lead_h)
            ax.fill(circle_lons, circle_lats, color="#3B82F6", alpha=alpha, **kwargs)
        
        # Prediction dots
        for pred in pred_points:
            color = CATEGORY_COLORS.get(pred.get("imd_code", "UNK"), "#60A5FA")
            ax.scatter(pred["lon"], pred["lat"], c=color, s=25, zorder=6,
                      marker="D", edgecolors="white", linewidths=0.5, **kwargs)
    
    # ---- Plot actual future (ground truth overlay in demo mode) ----
    if actual_future:
        actual_lats = [p["lat"] for p in actual_future]
        actual_lons = [p["lon"] for p in actual_future]
        
        kwargs = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        ax.plot(actual_lons, actual_lats, color="#10B981", linewidth=2, linestyle=":",
                alpha=0.7, label="Actual Track (Ground Truth)", **kwargs)
        ax.scatter(actual_lons, actual_lats, c="#10B981", s=15, zorder=4,
                  marker="o", alpha=0.5, **kwargs)
    
    # ---- Title and legend ----
    current_cat = wind_to_imd_category(past_track[-1].get("wind_kt", 0)) if past_track else None
    model_type = predictions.get("model_type", "unknown")
    
    title = f"🌀 {storm_name} — Track Prediction"
    if current_cat:
        title += f"\n{current_cat.name} ({current_cat.code})"
    title += f"  |  Model: {model_type}"
    
    ax.set_title(title, color="#F1F5F9", fontsize=14, fontweight="bold", pad=15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#E2E8F0", label="Past Track"),
        mpatches.Patch(facecolor="#60A5FA", label="Predicted Track"),
        mpatches.Patch(facecolor="#3B82F6", alpha=0.3, label="Uncertainty Cone"),
    ]
    if actual_future:
        legend_elements.append(mpatches.Patch(facecolor="#10B981", label="Actual (Ground Truth)"))
    
    # Category color legend
    for cat in IMD_CATEGORIES:
        legend_elements.append(
            plt.scatter([], [], c=CATEGORY_COLORS[cat.code], s=20, label=f"{cat.code}: {cat.name}")
        )
    
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
             facecolor="#1E293B", edgecolor="#4A5568", labelcolor="#E2E8F0",
             framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
               facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    
    logger.info(f"Track prediction plot saved to {output_path}")
    return output_path


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str = "outputs/visualizations/training_curves.png",
) -> str:
    """Plot training and validation loss curves."""
    ensure_dir(str(Path(output_path).parent))
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#3B82F6", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, color="#F97316", linewidth=2, label="Validation Loss")
    
    ax.set_xlabel("Epoch", color="#94A3B8")
    ax.set_ylabel("Loss", color="#94A3B8")
    ax.set_title("Training Curves", color="#F1F5F9", fontweight="bold")
    ax.legend(facecolor="#1E293B", edgecolor="#4A5568", labelcolor="#E2E8F0")
    ax.tick_params(colors="#94A3B8")
    ax.grid(True, alpha=0.2, color="#4A5568")
    
    for spine in ax.spines.values():
        spine.set_color("#4A5568")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    
    logger.info(f"Training curves saved to {output_path}")
    return output_path
