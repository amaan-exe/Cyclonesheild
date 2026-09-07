"""
Cyclone Horizon — Full Pipeline Runner
Runs ingestion → validation → features → split → train → evaluate.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/pipeline_config.yaml
    python scripts/run_pipeline.py --dry-run
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_config
from src.utils.logging_config import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Run the full Cyclone Horizon pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Report what would run without executing")
    parser.add_argument("--stage", choices=["ingest", "validate", "features", "split", "train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    
    run_id = setup_logging()
    logger = get_logger("scripts.run_pipeline")
    
    config = load_config(args.config)
    
    if args.dry_run:
        print("\n[DRY RUN] -- would execute the following stages:")
        print(f"  Config: {args.config}")
        print(f"  Stage:  {args.stage}")
        print(f"  Basin:  {config['project']['basin']}")
        print(f"  Year range: {config['ingestion']['ibtracs'].get('year_range', 'all')}")
        print(f"  Split ratios: {config['splitting']['ratios']}")
        print("\nNo data will be fetched or processed.")
        return
    
    stages = ["ingest", "validate", "features", "split", "train", "evaluate"]
    if args.stage != "all":
        stages = [args.stage]
    
    for stage in stages:
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"{'='*60}")
        
        if stage == "ingest":
            from src.ingestion.ibtracs_connector import IBTrACSConnector
            connector = IBTrACSConnector(config["ingestion"]["ibtracs"], data_dir="data/raw")
            connector.fetch()
        
        elif stage == "validate":
            from src.utils.io import load_parquet
            from src.validation.physical_checks import run_all_track_checks
            from src.validation.qc_report import generate_qc_report
            
            df = load_parquet("data/raw/ibtracs_ni.parquet")
            summary = run_all_track_checks(df)
            generate_qc_report(summary, df, run_id=run_id)
        
        elif stage == "features":
            from src.utils.io import load_parquet, save_parquet
            from src.features.environmental_features import compute_all_environmental_features
            from src.features.leakage_check import run_all_leakage_checks
            
            df = load_parquet("data/raw/ibtracs_ni.parquet")
            df = compute_all_environmental_features(df)
            save_parquet(df, "data/features/ibtracs_ni_features.parquet")
            run_all_leakage_checks(df)
        
        elif stage == "split":
            from src.utils.io import load_parquet
            from src.dataset.splitter import split_storms, save_split_manifest
            
            df = load_parquet("data/features/ibtracs_ni_features.parquet")
            splits = split_storms(df,
                train_ratio=config["splitting"]["ratios"]["train"],
                val_ratio=config["splitting"]["ratios"]["val"],
                test_ratio=config["splitting"]["ratios"]["test"],
            )
            save_split_manifest(splits, config_hash="v0.1")
        
        elif stage == "train":
            logger.info("Training stage — run separately with dedicated training script")
        
        elif stage == "evaluate":
            logger.info("Evaluation stage — run after training completes")
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
