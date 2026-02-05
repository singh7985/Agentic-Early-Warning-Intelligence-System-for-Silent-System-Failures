"""
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset Loader

Loads and parses turbofan engine degradation data.
- Engine ID, Cycle, Operational settings (3 features)
- 21 Sensor readings
- Training data has engines run until failure
- Test data has engines stopped before failure (RUL unknown)

Reference: https://data.nasa.gov/dataset/CMAPS
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CMAPSSDataLoader:
    """
    Loader for NASA C-MAPSS turbofan engine degradation dataset.
    
    Dataset structure:
    - FD001: 100 engines, normal operation
    - FD002: 260 engines, various operational conditions
    - FD003: 100 engines, normal operation with faults
    - FD004: 248 engines, various conditions with faults
    """
    
    # Column names for C-MAPSS data
    OPERATIONAL_FEATURES = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    SENSOR_FEATURES = [f'sensor_{i}' for i in range(1, 22)]
    
    def __init__(self, data_dir: str = "./data/raw/CMAPSS"):
        """
        Initialize loader.
        
        Args:
            data_dir: Directory containing C-MAPSS dataset files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {
            'FD001': {'train': None, 'test': None, 'rul': None},
            'FD002': {'train': None, 'test': None, 'rul': None},
            'FD003': {'train': None, 'test': None, 'rul': None},
            'FD004': {'train': None, 'test': None, 'rul': None},
        }
    
    def load_dataset(self, dataset_name: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load a specific C-MAPSS dataset.
        
        Args:
            dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
            
        Returns:
            Tuple of (train_df, test_df, rul_test_series)
        """
        assert dataset_name in self.datasets, f"Dataset must be one of {list(self.datasets.keys())}"
        
        # Check if already loaded
        if self.datasets[dataset_name]['train'] is not None:
            return (
                self.datasets[dataset_name]['train'],
                self.datasets[dataset_name]['test'],
                self.datasets[dataset_name]['rul']
            )
        
        # Define column names
        col_names = ['engine_id', 'cycle'] + self.OPERATIONAL_FEATURES + self.SENSOR_FEATURES
        
        # Load training data
        train_path = self.data_dir / f"train_{dataset_name}.txt"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_df = pd.read_csv(
            train_path,
            sep=' ',
            header=None,
            names=col_names,
            dtype={col: np.float32 for col in col_names}
        )
        
        # Load test data
        test_path = self.data_dir / f"test_{dataset_name}.txt"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        test_df = pd.read_csv(
            test_path,
            sep=' ',
            header=None,
            names=col_names,
            dtype={col: np.float32 for col in col_names}
        )
        
        # Load RUL (Remaining Useful Life) for test data
        rul_path = self.data_dir / f"RUL_{dataset_name}.txt"
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL data not found at {rul_path}")
        
        rul_test = pd.read_csv(
            rul_path,
            sep=' ',
            header=None,
            names=['RUL'],
            dtype=np.float32
        )['RUL']
        
        # Create RUL labels for training data
        train_df['RUL'] = train_df.groupby('engine_id')['cycle'].transform(
            lambda x: x.max() - x + 1
        )
        
        # Cache loaded data
        self.datasets[dataset_name]['train'] = train_df
        self.datasets[dataset_name]['test'] = test_df
        self.datasets[dataset_name]['rul'] = rul_test
        
        logger.info(f"Loaded {dataset_name}: Train={len(train_df)} rows, Test={len(test_df)} rows")
        
        return train_df, test_df, rul_test
    
    def load_all_datasets(self) -> dict:
        """
        Load all four C-MAPSS datasets.
        
        Returns:
            Dictionary with all datasets
        """
        result = {}
        for dataset_name in self.datasets.keys():
            try:
                train_df, test_df, rul_test = self.load_dataset(dataset_name)
                result[dataset_name] = {
                    'train': train_df,
                    'test': test_df,
                    'rul': rul_test
                }
            except FileNotFoundError as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
        
        return result
    
    @staticmethod
    def create_train_test_split(
        df: pd.DataFrame,
        test_engines_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data by engine ID (temporal split to avoid data leakage).
        
        Args:
            df: Full dataset with 'engine_id' column
            test_engines_ratio: Fraction of engines to use for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df)
        """
        np.random.seed(random_state)
        
        unique_engines = df['engine_id'].unique()
        num_test_engines = max(1, int(len(unique_engines) * test_engines_ratio))
        
        test_engines = np.random.choice(unique_engines, size=num_test_engines, replace=False)
        
        train_df = df[~df['engine_id'].isin(test_engines)].copy()
        val_df = df[df['engine_id'].isin(test_engines)].copy()
        
        logger.info(
            f"Split: Train engines={train_df['engine_id'].nunique()}, "
            f"Val engines={val_df['engine_id'].nunique()}"
        )
        
        return train_df, val_df
    
    @staticmethod
    def normalize_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using training set statistics (Z-score normalization).
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            feature_cols: Columns to normalize (default: all sensors)
            
        Returns:
            Tuple of (normalized_train, normalized_test)
        """
        if feature_cols is None:
            feature_cols = CMAPSSDataLoader.SENSOR_FEATURES
        
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Compute statistics on training set
        for col in feature_cols:
            mean = train_df[col].mean()
            std = train_df[col].std()
            
            # Avoid division by zero
            if std == 0:
                std = 1.0
            
            train_df[col] = (train_df[col] - mean) / std
            test_df[col] = (test_df[col] - mean) / std
        
        logger.info(f"Normalized {len(feature_cols)} sensor features")
        
        return train_df, test_df
    
    @staticmethod
    def get_failure_point(df: pd.DataFrame) -> pd.Series:
        """
        Get the cycle at which each engine failed (max cycle per engine).
        
        Args:
            df: Dataframe with 'engine_id' and 'cycle' columns
            
        Returns:
            Series with failure cycle per engine
        """
        return df.groupby('engine_id')['cycle'].max()


def prepare_cmapss_data(
    data_dir: str = "./data/raw/CMAPSS",
    dataset_name: str = 'FD001',
    test_engines_ratio: float = 0.2,
    normalize: bool = True
) -> dict:
    """
    Convenience function to load, split, and normalize C-MAPSS data.
    
    Args:
        data_dir: Path to C-MAPSS data directory
        dataset_name: Which dataset to use ('FD001', etc.)
        test_engines_ratio: Fraction of engines for validation
        normalize: Whether to normalize sensor features
        
    Returns:
        Dictionary with train/val/test splits
    """
    loader = CMAPSSDataLoader(data_dir)
    
    # Load dataset
    train_df, test_df, rul_test = loader.load_dataset(dataset_name)
    
    # Split training data by engine
    train_df, val_df = loader.create_train_test_split(train_df, test_engines_ratio)
    
    # Normalize features
    if normalize:
        train_df, val_df = loader.normalize_features(train_df, val_df)
        test_df_norm = test_df.copy()
        for col in loader.SENSOR_FEATURES:
            mean = train_df[col].mean()
            std = train_df[col].std() or 1.0
            test_df_norm[col] = (test_df_norm[col] - mean) / std
        test_df = test_df_norm
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'rul_test': rul_test,
        'dataset_name': dataset_name
    }
