"""
Time-Series Feature Engineering Module

Implements various feature engineering techniques for time-series data:
- Rolling statistics (mean, std, min, max)
- Exponential weighted moving average (EWMA)
- Fourier features for cyclical patterns
- Difference features (delta values)
- Trend estimation
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    """
    Feature engineering for time-series sensor data.
    """
    
    def __init__(self, window_sizes: List[int] = None, ewma_spans: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: Sliding window sizes for rolling statistics
            ewma_spans: Exponential weighted moving average spans
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        self.ewma_spans = ewma_spans or [5, 10, 20]
    
    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        window_sizes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add rolling mean, std, min, max features
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor column names
            window_sizes: Window sizes to use (default: self.window_sizes)
            
        Returns:
            DataFrame with new rolling features
        """
        df = df.copy()
        window_sizes = window_sizes or self.window_sizes
        
        # Group by engine to avoid leakage across engines
        for col in sensor_cols:
            for window in window_sizes: # Fix loop order: sensors first, then windows
        
                grouped = df.groupby('engine_id')[col].rolling(
                    window=window,
                    min_periods=1
                ).agg(['mean', 'std', 'min', 'max']).reset_index(drop=True) # Use agg list for efficiency
                
                # Check column names after agg
                # If agg returns a DataFrame with MultiIndex columns, flatten them
                # But rolling(...).agg(...) usually returns DataFrame with columns matching agg function names
                
                df[f'{col}_roll{window}_mean'] = grouped['mean']
                df[f'{col}_roll{window}_std'] = grouped['std'].fillna(0)
                df[f'{col}_roll{window}_min'] = grouped['min']
                df[f'{col}_roll{window}_max'] = grouped['max']
        
        logger.info(f"Added rolling statistics for windows: {window_sizes}")
        return df

    def add_ewma_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        ewma_spans: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add exponential weighted moving average features.
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor column names
            ewma_spans: EWMA spans to use (default: self.ewma_spans)
            
        Returns:
            DataFrame with new EWMA features
        """
        df = df.copy()
        ewma_spans = ewma_spans or self.ewma_spans
        
        # Group by engine to avoid leakage
        for col in sensor_cols:
            for span in ewma_spans:
                df[f'{col}_ewma{span}'] = df.groupby('engine_id')[col].ewm(
                    span=span,
                    adjust=False
                ).mean().reset_index(drop=True)
        
        logger.info(f"Added EWMA features for spans: {ewma_spans}")
        return df
    
    def add_difference_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add difference (delta) features.
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor column names
            lags: Lag values for differences (default: [1, 5, 10])
            
        Returns:
            DataFrame with new difference features
        """
        df = df.copy()
        lags = lags or [1, 5, 10]
        
        # Group by engine
        for col in sensor_cols:
            for lag in lags:
                df[f'{col}_diff{lag}'] = df.groupby('engine_id')[col].diff(periods=lag)
        
        logger.info(f"Added difference features for lags: {lags}")
        return df
    
    def add_fourier_features(
        self,
        df: pd.DataFrame,
        time_col: str = 'cycle',
        num_fourier_features: int = 5,
        period: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Add Fourier features to capture cyclical patterns.
        
        Args:
            df: Dataframe with time column
            time_col: Name of the time/cycle column
            num_fourier_features: Number of sin/cos pairs
            period: Period of the cycle (default: max value in time_col)
            
        Returns:
            DataFrame with new Fourier features
        """
        df = df.copy()
        
        if period is None:
            period = df[time_col].max()
            
        # Ensure time_col exists
        if time_col not in df.columns:
            logger.warning(f"Time column '{time_col}' not found. Skipping Fourier features.")
            return df
        
        for i in range(1, num_fourier_features + 1):
            df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df[time_col] / period)
            df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df[time_col] / period)
        
        logger.info(f"Added {num_fourier_features} Fourier feature pairs")
        return df
    
    @staticmethod
    def add_trend_features(
        df: pd.DataFrame,
        sensor_cols: List[str],
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Estimate trend (slope) for sensors using linear regression over window.
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor column names
            window_size: Window for trend estimation
            
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        for col in sensor_cols:
            def calc_trend(series):
                if len(series) < 2:
                    return np.nan
                # Calculate slope using simple linear regression
                x = np.arange(len(series))
                z = np.polyfit(x, series, 1)
                return z[0]  # Return slope
            
            df[f'{col}_trend'] = df.groupby('engine_id')[col].rolling(
                window=window_size,
                min_periods=2
            ).apply(calc_trend, raw=True).reset_index(drop=True)
        
        logger.info(f"Added trend features (window size: {window_size})")
        return df
    
    @staticmethod
    def add_statistical_features(
        df: pd.DataFrame,
        sensor_cols: List[str],
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Add statistical features (skewness, kurtosis).
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor column names
            window_size: Window for calculations
            
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        for col in sensor_cols:
            df[f'{col}_skew'] = df.groupby('engine_id')[col].rolling(
                window=window_size,
                min_periods=2
            ).skew().reset_index(drop=True)
            
            df[f'{col}_kurt'] = df.groupby('engine_id')[col].rolling(
                window=window_size,
                min_periods=2
            ).kurt().reset_index(drop=True)
        
        logger.info(f"Added statistical features (skewness, kurtosis)")
        return df


class ChangePointDetector:
    """
    Detect change points in sensor data (degradation onset).
    """
    
    @staticmethod
    def detect_pelt(
        series: np.ndarray,
        penalty: float = 10
    ) -> List[int]:
        """
        Detect change points using PELT algorithm.
        
        Args:
            series: 1D time series
            penalty: Penalty for adding change points
            
        Returns:
            List of change point indices
        """
        try:
            from ruptures import Pelt
            algo = Pelt(model='l2').fit(series)
            return algo.predict(pen=penalty)
        except Exception as e:
            logger.warning(f"PELT detection failed: {e}")
            return []
    
    @staticmethod
    def detect_binary_segmentation(
        series: np.ndarray,
        n_splits: int = 3
    ) -> List[int]:
        """
        Detect change points using binary segmentation.
        
        Args:
            series: 1D time series
            n_splits: Number of splits
            
        Returns:
            List of change point indices
        """
        try:
            from ruptures import Binseg
            algo = Binseg(model='l2', jump=1, min_size=5, max_size=None)
            algo.fit(series)
            return algo.predict(n_bkps=n_splits)
        except Exception as e:
            logger.warning(f"Binseg detection failed: {e}")
            return []
    
    @staticmethod
    def detect_degradation_onset(
        df: pd.DataFrame,
        sensor_cols: List[str],
        method: str = 'pelt'
    ) -> dict:
        """
        Detect degradation onset for each engine.
        
        Args:
            df: Dataframe with engine_id and sensor columns
            sensor_cols: Sensor columns to analyze
            method: 'pelt' or 'binseg'
            
        Returns:
            Dictionary mapping engine_id to degradation onset cycle
        """
        degradation_onsets = {}
        
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id]
            
            # Use first sensor for change point detection
            series = engine_data[sensor_cols[0]].values
            
            if method == 'pelt':
                change_points = ChangePointDetector.detect_pelt(series)
            else:
                change_points = ChangePointDetector.detect_binary_segmentation(series)
            
            # Use first change point as degradation onset
            if change_points:
                onset_idx = change_points[0]
                onset_cycle = engine_data.iloc[onset_idx]['cycle']
                degradation_onsets[engine_id] = onset_cycle
        
        logger.info(f"Detected degradation onset for {len(degradation_onsets)} engines")
        return degradation_onsets


def create_engineered_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    include_rolling: bool = True,
    include_ewma: bool = True,
    include_diff: bool = True,
    include_fourier: bool = True,
    include_trend: bool = True,
    include_stats: bool = True
) -> pd.DataFrame:
    """
    Convenience function to apply all feature engineering transformations.
    
    Args:
        df: Input dataframe
        sensor_cols: Sensor column names
        include_rolling: Include rolling statistics
        include_ewma: Include EWMA features
        include_diff: Include difference features
        include_fourier: Include Fourier features
        include_trend: Include trend features
        include_stats: Include statistical features
        
    Returns:
        Dataframe with engineered features
    """
    engineer = TimeSeriesFeatureEngineer()
    result = df.copy()
    
    if include_rolling:
        result = engineer.add_rolling_statistics(result, sensor_cols)
    
    if include_ewma:
        result = engineer.add_ewma_features(result, sensor_cols)
    
    if include_diff:
        result = engineer.add_difference_features(result, sensor_cols)
    
    if include_fourier:
        result = engineer.add_fourier_features(result)
    
    if include_trend:
        result = engineer.add_trend_features(result, sensor_cols)
    
    if include_stats:
        result = engineer.add_statistical_features(result, sensor_cols)
    
    logger.info(f"Created engineered features. Total columns: {result.shape[1]}")
    
    return result
