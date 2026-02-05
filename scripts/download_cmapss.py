#!/usr/bin/env python3
"""
Download NASA C-MAPSS Dataset

This script downloads the NASA C-MAPSS turbofan engine degradation dataset
from Kaggle and extracts it to the data/raw/CMAPSS directory.

Requirements:
- Kaggle API credentials (~/.kaggle/kaggle.json)
- Install: pip install kaggle

Setup:
1. Download kaggle.json from https://www.kaggle.com/settings/account
2. Place it at ~/.kaggle/kaggle.json
3. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
import logging
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_json.exists():
        logger.error(
            "Kaggle API credentials not found at ~/.kaggle/kaggle.json\n"
            "Please:\n"
            "1. Go to https://www.kaggle.com/settings/account\n"
            "2. Click 'Create New API Token' (downloads kaggle.json)\n"
            "3. Move it to ~/.kaggle/kaggle.json\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
        return False
    
    return True


def download_cmapss_dataset(output_dir: str = "./data/raw/CMAPSS") -> bool:
    """
    Download NASA C-MAPSS dataset using Kaggle API.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful, False otherwise
    """
    # Check Kaggle setup
    if not check_kaggle_setup():
        return False
    
    # Import Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading NASA C-MAPSS dataset to {output_path}")
    
    try:
        # Authenticate with Kaggle
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        # Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
        api.dataset_download_files(
            'behrad3d/nasa-cmaps',
            path=output_path,
            unzip=True
        )
        
        logger.info("✅ Download completed!")
        
        # Verify files
        expected_files = [
            'train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt',
            'train_FD002.txt', 'test_FD002.txt', 'RUL_FD002.txt',
            'train_FD003.txt', 'test_FD003.txt', 'RUL_FD003.txt',
            'train_FD004.txt', 'test_FD004.txt', 'RUL_FD004.txt',
        ]
        
        missing = []
        for fname in expected_files:
            fpath = output_path / fname
            if fpath.exists():
                size = fpath.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  ✓ {fname} ({size:.1f} MB)")
            else:
                missing.append(fname)
        
        if missing:
            logger.warning(f"⚠️  Missing files: {missing}")
            return False
        
        logger.info(f"✅ All {len(expected_files)} C-MAPSS files verified!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def download_loghub_datasets(output_dir: str = "./data/raw/LogHub") -> bool:
    """
    Download LogHub datasets for text data.
    
    Popular datasets:
    - HDFS: Hadoop HDFS logs
    - BGL: Blue Gene/L supercomputer logs
    - OpenStack: OpenStack logs
    - Android: Android logs
    
    Args:
        output_dir: Directory to save datasets
        
    Returns:
        True if successful
    """
    # Check Kaggle setup
    if not check_kaggle_setup():
        logger.warning("Skipping LogHub download (Kaggle not configured)")
        return False
    
    # Import Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List of LogHub datasets to download
    datasets = [
        'logpai/loghub-2.0-hdfs',      # HDFS logs
        'logpai/loghub-2.0-bgl',       # BGL logs
    ]
    
    logger.info("Downloading LogHub datasets...")
    
    success_count = 0
    for dataset_id in datasets:
        try:
            logger.info(f"Downloading {dataset_id}...")
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_files(
                dataset_id,
                path=output_path / dataset_id.split('/')[-1],
                unzip=True
            )
            
            logger.info(f"  ✓ Downloaded {dataset_id}")
            success_count += 1
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to download {dataset_id}: {e}")
    
    if success_count > 0:
        logger.info(f"✅ Downloaded {success_count}/{len(datasets)} LogHub datasets")
        return True
    else:
        logger.warning("⚠️  No LogHub datasets downloaded")
        return False


def main():
    """Main download script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download datasets for EWIS project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_cmapss.py                    # Download C-MAPSS only
  python scripts/download_cmapss.py --all              # Download C-MAPSS + LogHub
  python scripts/download_cmapss.py --loghub-only      # Download LogHub only
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets (C-MAPSS + LogHub)'
    )
    
    parser.add_argument(
        '--loghub-only',
        action='store_true',
        help='Download only LogHub datasets'
    )
    
    parser.add_argument(
        '--cmapss-dir',
        default='./data/raw/CMAPSS',
        help='Output directory for C-MAPSS (default: ./data/raw/CMAPSS)'
    )
    
    parser.add_argument(
        '--loghub-dir',
        default='./data/raw/LogHub',
        help='Output directory for LogHub (default: ./data/raw/LogHub)'
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Download C-MAPSS
    if not args.loghub_only:
        logger.info("="*60)
        logger.info("DOWNLOADING NASA C-MAPSS DATASET")
        logger.info("="*60)
        success = download_cmapss_dataset(args.cmapss_dir) and success
    
    # Download LogHub
    if args.all or args.loghub_only:
        logger.info("\n" + "="*60)
        logger.info("DOWNLOADING LOGHUB DATASETS")
        logger.info("="*60)
        success = download_loghub_datasets(args.loghub_dir) and success
    
    # Summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ ALL DOWNLOADS COMPLETED")
    else:
        logger.warning("⚠️  SOME DOWNLOADS FAILED - Check logs above")
    logger.info("="*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
