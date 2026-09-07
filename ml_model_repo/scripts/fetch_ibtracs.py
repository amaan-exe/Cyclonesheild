"""
Cyclone Horizon — IBTrACS Data Fetch Script
Downloads and parses IBTrACS North Indian Ocean best-track data.

Usage:
    python scripts/fetch_ibtracs.py
    python scripts/fetch_ibtracs.py --year-start 2010 --year-end 2024
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_config, load_parquet
from src.utils.logging_config import setup_logging, get_logger
from src.ingestion.ibtracs_connector import IBTrACSConnector
from src.validation.physical_checks import run_all_track_checks
from src.validation.qc_report import generate_qc_report
from src.features.environmental_features import compute_all_environmental_features
from src.features.leakage_check import run_all_leakage_checks
from src.dataset.splitter import split_storms, save_split_manifest


def main():
    parser = argparse.ArgumentParser(description="Fetch and process IBTrACS data")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--skip-split", action="store_true", help="Skip dataset splitting")
    args = parser.parse_args()
    
    # Setup
    run_id = setup_logging()
    logger = get_logger("scripts.fetch_ibtracs")
    logger.info(f"Starting IBTrACS pipeline (run_id={run_id})")
    
    # Load config
    config = load_config(args.config)
    ibtracs_config = config.get("ingestion", {}).get("ibtracs", {})
    
    # Override year range if specified
    if args.year_start:
        ibtracs_config["year_range"] = [args.year_start, args.year_end or 2025]
    
    # ---- Step 1: Ingest ----
    logger.info("=" * 60)
    logger.info("STEP 1: Ingesting IBTrACS data")
    logger.info("=" * 60)
    
    connector = IBTrACSConnector(ibtracs_config, data_dir="data/raw")
    raw_files = connector.fetch()
    
    if not raw_files:
        logger.error("No data fetched!")
        sys.exit(1)
    
    # Validate
    for rf in raw_files:
        if not connector.validate(rf):
            logger.error(f"Validation failed for {rf.filepath}")
            sys.exit(1)
    
    # ---- Step 2: QC Checks ----
    logger.info("=" * 60)
    logger.info("STEP 2: Running QC checks")
    logger.info("=" * 60)
    
    df = load_parquet(raw_files[0].filepath)
    qc_summary = run_all_track_checks(df)
    
    report_path = generate_qc_report(qc_summary, df, run_id=run_id)
    logger.info(f"QC report: {report_path}")
    
    # ---- Step 3: Feature Engineering ----
    if not args.skip_features:
        logger.info("=" * 60)
        logger.info("STEP 3: Computing environmental features")
        logger.info("=" * 60)
        
        df = compute_all_environmental_features(df)
        
        # Save feature-enriched data
        features_path = "data/features/ibtracs_ni_features.parquet"
        from src.utils.io import save_parquet
        save_parquet(df, features_path)
        
        # ---- Step 4: Leakage Check ----
        logger.info("=" * 60)
        logger.info("STEP 4: Running leakage checks")
        logger.info("=" * 60)
        
        all_pass, leakage_results = run_all_leakage_checks(df)
        if not all_pass:
            logger.error("LEAKAGE DETECTED! Fix before training.")
            for feat, result in leakage_results.items():
                if not result["passes"]:
                    logger.error(f"  {feat}: {len(result['violations'])} violations")
        else:
            logger.info("All leakage checks PASSED ✓")
    
    # ---- Step 5: Dataset Splitting ----
    if not args.skip_split:
        logger.info("=" * 60)
        logger.info("STEP 5: Splitting dataset (storm-level)")
        logger.info("=" * 60)
        
        splits = split_storms(
            df,
            train_ratio=config["splitting"]["ratios"]["train"],
            val_ratio=config["splitting"]["ratios"]["val"],
            test_ratio=config["splitting"]["ratios"]["test"],
            random_seed=config["splitting"]["random_seed"],
        )
        
        manifest_path = save_split_manifest(splits, config_hash="v0.1")
        logger.info(f"Split manifest: {manifest_path}")
    
    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    n_storms = df["storm_id"].nunique()
    n_records = len(df)
    
    print(f"\n{'='*60}")
    print(f"  IBTrACS Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Storms:  {n_storms}")
    print(f"  Records: {n_records}")
    print(f"  QC Report: {report_path}")
    
    if "imd_category_name" in df.columns:
        print(f"\n  Category Distribution (by storm):")
        cat_counts = df.groupby("storm_id")["imd_category_name"].apply(
            lambda x: x.value_counts().index[0]  # peak category
        ).value_counts()
        for cat, count in cat_counts.items():
            print(f"    {cat}: {count}")
    
    if not args.skip_features:
        ri_count = df["is_rapid_intensification"].sum() if "is_rapid_intensification" in df.columns else 0
        print(f"\n  Rapid Intensification records: {ri_count}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
