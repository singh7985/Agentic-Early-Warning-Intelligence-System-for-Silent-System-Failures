"""
Sliding Window Feature Generation for Time-Series Data

Generates fixed-size windows of sensor readings to create sequences for RUL prediction.
Each window represents the most recent N cycles of an engine's operation.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SlidingWindowGenerator:
    """Generate fixed-size sliding windows from time-series data."""

    def __init__(self, window_size: int = 30, step_size: int = 1, min_window_samples: int = 5):
        """
        Initialize sliding window generator.

        Parameters
        ----------
        window_size : int, default=30
            Number of cycles per window
        step_size : int, default=1
            Step size for window advancement (1 = every cycle, 5 = every 5 cycles)
        min_window_samples : int, default=5
            Minimum samples required in a window (handles edge cases)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_window_samples = min_window_samples

    def generate_windows(
        self, df: pd.DataFrame, engine_col: str = "engine_id", cycle_col: str = "cycle"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sliding windows for each engine.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series dataframe with engine_id, cycle, and sensor columns
        engine_col : str, default='engine_id'
            Column name for engine identifier
        cycle_col : str, default='cycle'
            Column name for cycle number

        Returns
        -------
        X : np.ndarray
            Windows array of shape (num_windows, window_size, num_features)
        engine_ids : np.ndarray
            Engine ID for each window
        rul_labels : np.ndarray
            RUL label for each window (last cycle's RUL)
        """
        # Identify sensor columns (exclude metadata)
        metadata_cols = {engine_col, cycle_col, "RUL", "rul"}
        sensor_cols = [col for col in df.columns if col not in metadata_cols]

        windows_list = []
        engine_ids_list = []
        rul_labels_list = []

        logger.info(f"Generating sliding windows (size={self.window_size}, step={self.step_size})")

        for engine_id in sorted(df[engine_col].unique()):
            engine_data = df[df[engine_col] == engine_id].reset_index(drop=True)

            # Generate windows for this engine
            # Fix: Added +1 to ensure the last possible window is included
            for start_idx in range(0, len(engine_data) - self.min_window_samples + 1, self.step_size):
                end_idx = min(start_idx + self.window_size, len(engine_data))
                window_data = engine_data.iloc[start_idx:end_idx][sensor_cols].values

                # Pad window if too small (only if we can reach it)
                if window_data.shape[0] < self.window_size:
                    padding = np.zeros(
                        (self.window_size - window_data.shape[0], window_data.shape[1])
                    )
                    window_data = np.vstack([padding, window_data])

                # Get RUL label from last cycle in window
                last_cycle_idx = end_idx - 1
                rul = engine_data.iloc[last_cycle_idx].get("RUL", engine_data.iloc[last_cycle_idx].get("rul", -1))

                windows_list.append(window_data)
                engine_ids_list.append(engine_id)
                rul_labels_list.append(rul)

        X = np.array(windows_list)
        engine_ids = np.array(engine_ids_list)
        rul_labels = np.array(rul_labels_list)

        logger.info(f"Generated {len(windows_list)} windows from {df[engine_col].nunique()} engines")
        logger.info(f"Window shape: {X.shape} (num_windows, window_size, features)")

        return X, engine_ids, rul_labels

    def flatten_windows(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten 3D windows to 2D for ML models.

        Parameters
        ----------
        X : np.ndarray
            Windows array of shape (num_windows, window_size, num_features)

        Returns
        -------
        X_flat : np.ndarray
            Flattened array of shape (num_windows, window_size * num_features)
        """
        num_windows, window_size, num_features = X.shape
        X_flat = X.reshape(num_windows, window_size * num_features)
        logger.debug(f"Flattened windows: {X.shape} -> {X_flat.shape}")
        return X_flat

    def create_sequences_dict(
        self, X: np.ndarray, engine_ids: np.ndarray, rul_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Package windows into a dictionary for easy access.

        Parameters
        ----------
        X : np.ndarray
            Windows array
        engine_ids : np.ndarray
            Engine IDs for each window
        rul_labels : np.ndarray
            RUL labels for each window

        Returns
        -------
        sequences : dict
            Dictionary with 'windows', 'engine_ids', 'rul_labels', 'flattened'
        """
        return {
            "windows": X,
            "engine_ids": engine_ids,
            "rul_labels": rul_labels,
            "flattened": self.flatten_windows(X),
            "num_sequences": X.shape[0],
            "sequence_shape": X.shape,
        }
