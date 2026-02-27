"""
Reproducible Feature Engineering Pipeline

Integrates all feature engineering steps into a single reproducible pipeline.
Can be saved/loaded for consistent preprocessing across train/validation/test sets.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.sliding_windows import SlidingWindowGenerator
from src.features.health_indicators import HealthIndicatorCalculator
from src.features.feature_selection import FeatureSelector
from src.features.engineering import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline with reproducibility."""

    def __init__(
        self,
        window_size: int = 30,
        window_step: int = 1,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize feature engineering pipeline.

        Parameters
        ----------
        window_size : int, default=30
            Size of sliding windows
        window_step : int, default=1
            Step size for window generation
        scale_features : bool, default=True
            Whether to standardize features
        random_state : int, default=42
            Random state for reproducibility
        """
        self.window_size = window_size
        self.window_step = window_step
        self.scale_features = scale_features
        self.random_state = random_state

        # Initialize components
        self.window_generator = SlidingWindowGenerator(window_size=window_size, step_size=window_step)
        self.health_calculator = HealthIndicatorCalculator()
        self.feature_selector = FeatureSelector(random_state=random_state)
        self.time_series_engineer = TimeSeriesFeatureEngineer()
        self.scaler = StandardScaler() if scale_features else None

        # Store configuration
        self.config = {
            "window_size": window_size,
            "window_step": window_step,
            "scale_features": scale_features,
            "random_state": random_state,
        }

        # Track fitted state
        self.is_fitted = False
        self.sensor_cols: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

    def fit(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        target_col: str = "RUL",
        feature_selection_method: str = "combined",
        selection_k: int = 20,
    ) -> "FeatureEngineeringPipeline":
        """
        Fit pipeline on training data.

        Process:
        1. Generate sliding windows
        2. Calculate health indicators
        3. Apply time-series feature engineering
        4. Select features (optional)
        5. Fit scaler (optional)

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe
        sensor_cols : List[str]
            List of sensor column names
        target_col : str, default='RUL'
            Target column name
        feature_selection_method : str, default='combined'
            Feature selection method: 'variance', 'correlation', 'tree', 'pca', 'combined', or None
        selection_k : int, default=20
            Number of features to select

        Returns
        -------
        self : FeatureEngineeringPipeline
            Fitted pipeline
        """
        logger.info("Fitting feature engineering pipeline...")
        self.sensor_cols = sensor_cols

        # Step 1: Generate sliding windows
        logger.info("Step 1: Generating sliding windows")
        X_windows, engine_ids, rul_labels = self.window_generator.generate_windows(df)
        X_flat = self.window_generator.flatten_windows(X_windows)

        # Step 2: Calculate health indicators
        logger.info("Step 2: Calculating health indicators")
        # Fit health calculator to store reference baseline for transform()
        self.health_calculator.fit(df, sensor_cols)
        df_health = self._add_health_indicators(df, sensor_cols)

        # Step 3: Apply time-series feature engineering
        logger.info("Step 3: Engineering time-series features")
        df_engineered = self.time_series_engineer.add_rolling_statistics(
            df_health, sensor_cols, window_sizes=[5, 10, 20]
        )
        df_engineered = self.time_series_engineer.add_ewma_features(df_engineered, sensor_cols)
        df_engineered = self.time_series_engineer.add_difference_features(df_engineered, sensor_cols)
        # Fix: add_fourier_features expects time_col, not sensor_cols
        # Also it returns a dataframe, pass it correctly
        df_engineered = self.time_series_engineer.add_fourier_features(df_engineered, time_col='cycle')

        # Extract engineered features as 2D array
        # Exclude metadata columns to just get features
        metadata_cols = ["engine_id", "cycle", "RUL", "rul", "op_setting_1", "op_setting_2", "op_setting_3"]
        engineered_cols = [col for col in df_engineered.columns if col not in metadata_cols]
        
        # Ensure only numeric columns are selected
        engineered_cols = [col for col in engineered_cols if pd.api.types.is_numeric_dtype(df_engineered[col])]

        # Create the feature matrix
        X_engineered = df_engineered[engineered_cols].values

        logger.info(f"Engineered features: {X_engineered.shape[1]} features")

        # Handle NaNs
        # Rolling features and differencing introduce NaNs at the beginning of each engine's data
        # We need to handle them before feature selection/scaling
        # Option 1: Fill with 0 (simplest)
        # Option 2: Drop rows (might lose data if windows are large)
        # Option 3: Backfill/Forwardfill
        
        # Here we'll use fillna(0) for robustness, but dropping is also valid if we have enough data
        # df_engineered = df_engineered.dropna(subset=engineered_cols) # Would require re-aligning y
        
        # Let's fill with 0 to be safe and keep all rows aligned with y
        X_engineered = np.nan_to_num(X_engineered, nan=0.0)

        # Step 4: Feature selection
        if feature_selection_method:
            logger.info(f"Step 4: Selecting features using {feature_selection_method} method")
            X_selected, self.selected_features = self._select_features(
                pd.DataFrame(X_engineered, columns=engineered_cols),
                df[target_col],
                method=feature_selection_method,
                k=selection_k,
            )
            X_final = X_selected.values
        else:
            logger.info("Step 4: Skipping feature selection")
            self.selected_features = engineered_cols
            X_final = X_engineered

        # Step 5: Fit scaler
        if self.scaler:
            logger.info("Step 5: Fitting feature scaler")
            self.scaler.fit(X_final)

        self.is_fitted = True
        logger.info("Pipeline fitting complete ✓")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform

        Returns
        -------
        X : np.ndarray
            Transformed features
        y : np.ndarray
            RUL labels
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        if self.sensor_cols is None:
            raise RuntimeError("Sensor columns not defined")

        logger.info(f"Transforming {len(df)} samples")

        # Health indicators
        df_health = self._add_health_indicators(df, self.sensor_cols)

        # Time-series feature engineering
        df_engineered = self.time_series_engineer.add_rolling_statistics(
            df_health, self.sensor_cols, window_sizes=[5, 10, 20]
        )
        df_engineered = self.time_series_engineer.add_ewma_features(df_engineered, self.sensor_cols)
        df_engineered = self.time_series_engineer.add_difference_features(df_engineered, self.sensor_cols)
        df_engineered = self.time_series_engineer.add_fourier_features(df_engineered, time_col='cycle')

        # Extract features
        # Also fill NaNs here for consistency with fit
        X = df_engineered[self.selected_features].values
        X = np.nan_to_num(X, nan=0.0)

        # Scale if fitted
        if self.scaler:
            X = self.scaler.transform(X)

        # Get target
        if "RUL" in df.columns:
            y = df["RUL"].values
        elif "rul" in df.columns:
            y = df["rul"].values
        else:
            y = np.zeros(len(df))

        logger.info(f"Transform complete: X={X.shape}")
        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        target_col: str = "RUL",
        feature_selection_method: str = "combined",
        selection_k: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Training data
        sensor_cols : List[str]
            Sensor column names
        target_col : str, default='RUL'
            Target column name
        feature_selection_method : str, default='combined'
            Feature selection method
        selection_k : int, default=20
            Number of features to select

        Returns
        -------
        X : np.ndarray
            Transformed features
        y : np.ndarray
            RUL labels
        """
        self.fit(df, sensor_cols, target_col, feature_selection_method, selection_k)
        return self.transform(df)

    def save(self, output_dir: str) -> None:
        """
        Save pipeline to disk for reproducibility.

        Saves:
        - Pipeline configuration (JSON)
        - Fitted components (pickle)
        - Feature information (CSV)

        Parameters
        ----------
        output_dir : str
            Directory to save pipeline
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_file = output_path / "pipeline_config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config: {config_file}")

        # Save fitted components
        components_file = output_path / "pipeline_components.pkl"
        components = {
            "scaler": self.scaler,
            "sensor_cols": self.sensor_cols,
            "selected_features": self.selected_features,
            "time_series_engineer": self.time_series_engineer,
        }
        with open(components_file, "wb") as f:
            pickle.dump(components, f)
        logger.info(f"Saved components: {components_file}")

        # Save feature list
        if self.selected_features:
            features_df = pd.DataFrame({"feature": self.selected_features})
            features_file = output_path / "selected_features.csv"
            features_df.to_csv(features_file, index=False)
            logger.info(f"Saved features: {features_file}")

    @classmethod
    def load(cls, output_dir: str) -> "FeatureEngineeringPipeline":
        """
        Load pipeline from disk.

        Parameters
        ----------
        output_dir : str
            Directory containing saved pipeline

        Returns
        -------
        pipeline : FeatureEngineeringPipeline
            Loaded pipeline
        """
        output_path = Path(output_dir)

        # Load configuration
        config_file = output_path / "pipeline_config.json"
        with open(config_file, "r") as f:
            config = json.load(f)

        # Initialize pipeline with saved config
        pipeline = cls(**config)

        # Load fitted components
        components_file = output_path / "pipeline_components.pkl"
        with open(components_file, "rb") as f:
            components = pickle.load(f)

        pipeline.scaler = components["scaler"]
        pipeline.sensor_cols = components["sensor_cols"]
        pipeline.selected_features = components["selected_features"]
        pipeline.time_series_engineer = components["time_series_engineer"]
        pipeline.is_fitted = True

        logger.info(f"Loaded pipeline from {output_dir}")
        return pipeline

    def get_config(self) -> Dict:
        """Get pipeline configuration."""
        return self.config.copy()

    def get_feature_info(self) -> Dict:
        """Get information about selected features."""
        return {
            "num_features": len(self.selected_features) if self.selected_features else 0,
            "selected_features": self.selected_features,
            "sensor_cols": self.sensor_cols,
            "is_fitted": self.is_fitted,
        }

    # ==================== Private Methods ====================

    def _add_health_indicators(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Add health indicators to dataframe using the stored reference baseline (no re-fitting)."""
        # Use transform() which relies on self.reference_baseline_ set during fit().
        # Never call .fit() here — that would recompute baselines from the current data
        # and give different results after joblib serialization/deserialization.
        if hasattr(self.health_calculator, 'transform'):
            return self.health_calculator.transform(df, sensor_cols)

        # Legacy fallback for older HealthIndicatorCalculator without transform()
        df_health = df.copy()
        df_health = self.health_calculator.calculate_sensor_drift(df_health, sensor_cols)
        drift_cols = [f"{s}_drift" for s in sensor_cols]
        health_index = self.health_calculator.calculate_combined_health_index(
            df_health, drift_cols, method="mean"
        )
        df_health["health_index"] = health_index
        return df_health

    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "combined",
        k: int = 20,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply feature selection."""
        if method == "variance":
            return self.feature_selector.select_by_variance(X, threshold=0.01)
        elif method == "correlation":
            return self.feature_selector.select_by_correlation(X, y, k=k)
        elif method == "tree":
            X_sel, feat, _ = self.feature_selector.select_by_tree_importance(X, y, k=k)
            return X_sel, feat
        elif method == "pca":
            X_sel, _ = self.feature_selector.select_by_pca(X, n_components=k)
            cols = [f"PC{i+1}" for i in range(X_sel.shape[1])]
            return pd.DataFrame(X_sel, columns=cols), cols
        elif method == "combined":
            return self.feature_selector.select_combined(X, y)
        else:
            raise ValueError(f"Unknown selection method: {method}")
