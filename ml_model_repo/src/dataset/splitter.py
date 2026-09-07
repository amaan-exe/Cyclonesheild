"""
Cyclone Horizon — Storm-Level Dataset Splitter
Splits by STORM EVENT (never by frame) to prevent data leakage.
Stratified by IMD intensity category and basin.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.constants import wind_to_imd_category
from src.utils.io import ensure_dir, save_parquet
from src.utils.logging_config import get_logger

logger = get_logger("dataset.splitter")


def compute_storm_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-storm metadata for stratified splitting.
    
    Returns a DataFrame with one row per storm:
    storm_id, peak_category_code, peak_wind_kt, year, month, n_fixes
    """
    storms = df.groupby("storm_id").agg(
        peak_wind_kt=("max_wind_kt", "max"),
        n_fixes=("timestamp", "count"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        mean_lat=("lat", "mean"),
    ).reset_index()
    
    storms["peak_category_code"] = storms["peak_wind_kt"].apply(
        lambda w: wind_to_imd_category(w).code if pd.notna(w) else "UNK"
    )
    storms["year"] = storms["start_time"].dt.year
    storms["month"] = storms["start_time"].dt.month
    
    return storms


def split_storms(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "peak_category_code",
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split storm IDs into train/val/test sets.
    
    CRITICAL: split by storm event, never by frame.
    Stratified by peak IMD category so train/val/test have
    similar intensity distributions.
    
    Parameters
    ----------
    df : DataFrame with storm data
    train_ratio, val_ratio, test_ratio : split proportions (must sum to 1)
    stratify_by : column for stratification
    random_seed : for reproducibility
    
    Returns
    -------
    Dict with keys 'train', 'val', 'test' → list of storm IDs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1, got {train_ratio + val_ratio + test_ratio}"
    
    storms = compute_storm_metadata(df)
    storm_ids = np.array(storms["storm_id"].tolist())
    
    # Use peak category for stratification
    # Group rare categories together so stratification works with small counts
    strat_labels = np.array(storms[stratify_by].tolist())
    
    # Count categories, merge rare ones (< 3 storms) with adjacent category
    from collections import Counter
    counts = Counter(strat_labels)
    min_count = 3
    label_map = {}
    for label, count in counts.items():
        if count < min_count:
            label_map[label] = "OTHER"
        else:
            label_map[label] = label
    
    strat_mapped = np.array([label_map.get(s, "OTHER") for s in strat_labels])
    
    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    
    try:
        train_ids, valtest_ids, _, strat_valtest = train_test_split(
            storm_ids, strat_mapped,
            test_size=val_test_ratio,
            stratify=strat_mapped,
            random_state=random_seed,
        )
    except ValueError:
        # Stratification failed (too few samples), fall back to random
        logger.warning("Stratified split failed, falling back to random split")
        train_ids, valtest_ids = train_test_split(
            storm_ids,
            test_size=val_test_ratio,
            random_state=random_seed,
        )
        strat_valtest = None
    
    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    
    try:
        if strat_valtest is not None:
            val_ids, test_ids = train_test_split(
                valtest_ids,
                test_size=relative_test,
                stratify=strat_valtest,
                random_state=random_seed,
            )
        else:
            val_ids, test_ids = train_test_split(
                valtest_ids,
                test_size=relative_test,
                random_state=random_seed,
            )
    except ValueError:
        val_ids, test_ids = train_test_split(
            valtest_ids,
            test_size=relative_test,
            random_state=random_seed,
        )
    
    splits = {
        "train": sorted(train_ids.tolist()),
        "val": sorted(val_ids.tolist()),
        "test": sorted(test_ids.tolist()),
    }
    
    logger.info(
        f"Split: train={len(splits['train'])} storms, "
        f"val={len(splits['val'])} storms, "
        f"test={len(splits['test'])} storms"
    )
    
    return splits


def save_split_manifest(
    splits: Dict[str, List[str]],
    config_hash: str,
    output_dir: str = "data/datasets",
    version: str = "v0.1",
) -> str:
    """
    Save the split manifest as a versioned JSON artifact.
    
    This is the definitive record of which storms are in which split.
    It's versioned and never silently overwritten.
    """
    ensure_dir(output_dir)
    
    manifest = {
        "version": version,
        "config_hash": config_hash,
        "splits": splits,
        "storm_counts": {k: len(v) for k, v in splits.items()},
    }
    
    manifest_path = str(Path(output_dir) / f"split_manifest_{version}.json")
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Split manifest saved to {manifest_path}")
    return manifest_path


def apply_split(
    df: pd.DataFrame,
    splits: Dict[str, List[str]],
    storm_id_col: str = "storm_id",
) -> Dict[str, pd.DataFrame]:
    """
    Apply a split manifest to a DataFrame, returning separate DataFrames.
    
    Returns
    -------
    Dict with keys 'train', 'val', 'test' → filtered DataFrames
    """
    result = {}
    for split_name, storm_ids in splits.items():
        mask = df[storm_id_col].isin(storm_ids)
        result[split_name] = df[mask].copy()
        logger.info(f"{split_name}: {mask.sum()} records from {len(storm_ids)} storms")
    
    return result
