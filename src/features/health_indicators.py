"""
Health Indicators for Engine Degradation Detection

Computes domain-specific health indicators from sensor readings.
These indicate degradation stages and are more interpretable than raw sensor values.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal, stats

logger = logging.getLogger(__name__)


class HealthIndicatorCalculator:
    """Calculate health indicators from sensor readings."""

    def __init__(self, reference_threshold: float = 2.0):
        """
        Initialize health indicator calculator.

        Parameters
        ----------
        reference_threshold : float, default=2.0
            Number of standard deviations to define drift threshold
        """
        self.reference_threshold = reference_threshold

    def fit(self, df: pd.DataFrame, sensor_cols: List[str], window_size: int = 10):
        """
        Fit the reference baseline (mean and std) for each sensor using the first
        window_size cycles per engine. Stores result in self.reference_baseline_.
        """
        self.reference_baseline_ = {}
        for sensor in sensor_cols:
            means = df.groupby("engine_id")[sensor].apply(
                lambda x: x.iloc[:window_size].mean() if len(x) >= window_size else x.mean()
            )
            stds = df.groupby("engine_id")[sensor].apply(
                lambda x: x.iloc[:window_size].std() if len(x) >= window_size else x.std()
            )
            self.reference_baseline_[sensor] = {
                'mean': means,
                'std': stds.replace(0, 1.0)
            }
        return self

    def transform(self, df: pd.DataFrame, sensor_cols: List[str], window_size: int = 10) -> pd.DataFrame:
        """
        Use the stored reference_baseline_ to compute drift features and health index.
        Does NOT recompute the baseline; uses what was stored in fit().
        """
        if not hasattr(self, 'reference_baseline_'):
            raise RuntimeError("HealthIndicatorCalculator must be fit before transform.")
        df_drift = df.copy()
        for sensor in sensor_cols:
            mean_map = df_drift['engine_id'].map(self.reference_baseline_[sensor]['mean'])
            std_map = df_drift['engine_id'].map(self.reference_baseline_[sensor]['std']).replace(0, 1.0)
            df_drift[f"{sensor}_drift"] = np.abs((df_drift[sensor] - mean_map) / (std_map + 1e-8))
        drift_cols = [f"{s}_drift" for s in sensor_cols]
        health_index = self.calculate_combined_health_index(df_drift, drift_cols, method="mean")
        df_drift["health_index"] = health_index
        return df_drift

    def calculate_sensor_drift(
        self, df: pd.DataFrame, sensor_cols: List[str], window_size: int = 10
    ) -> pd.DataFrame:
        """
        Calculate drift magnitude for each sensor.

        Drift = |current_value - baseline_value| / baseline_std

        Parameters
        ----------
        df : pd.DataFrame
            Time-series dataframe
        sensor_cols : List[str]
            Sensor column names
        window_size : int, default=10
            Number of cycles for baseline calculation

        Returns
        -------
        df_drift : pd.DataFrame
            Original dataframe with added drift columns
        """
        df_drift = df.copy()

        for sensor in sensor_cols:
            # Calculate rolling baseline (mean and std of first window_size cycles per engine)
            baseline_mean = df.groupby("engine_id")[sensor].transform(
                lambda x: x.iloc[:window_size].mean() if len(x) >= window_size else x.mean()
            )
            baseline_std = df.groupby("engine_id")[sensor].transform(
                lambda x: x.iloc[:window_size].std() if len(x) >= window_size else x.std()
            )

            # Avoid division by zero
            baseline_std = baseline_std.replace(0, 1.0)

            # Calculate drift as z-score
            df_drift[f"{sensor}_drift"] = np.abs(
                (df[sensor] - baseline_mean) / (baseline_std + 1e-8)
            )

        logger.info(f"Calculated sensor drift for {len(sensor_cols)} sensors")
        return df_drift

    def calculate_combined_health_index(
        self, df: pd.DataFrame, drift_cols: List[str], method: str = "mean"
    ) -> pd.Series:
        """
        Combine drift values into single health index.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with drift columns
        drift_cols : List[str]
            List of drift column names
        method : str, default='mean'
            Aggregation method: 'mean', 'max', 'weighted_max'

        Returns
        -------
        health_index : pd.Series
            Combined health index (lower = healthier, higher = more degraded)
        """
        if method == "mean":
            health_index = df[drift_cols].mean(axis=1)
        elif method == "max":
            health_index = df[drift_cols].max(axis=1)
        elif method == "weighted_max":
            # Weight max drift more heavily
            health_index = df[drift_cols].max(axis=1) * 0.7 + df[drift_cols].mean(axis=1) * 0.3
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Calculated combined health index using {method} method")
        return health_index

    def calculate_trend_acceleration(
        self, df: pd.DataFrame, sensor: str, window_size: int = 10
    ) -> pd.Series:
        """
        Calculate second derivative (acceleration) of sensor trend.

        High acceleration = rapid degradation.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series dataframe
        sensor : str
            Sensor column name
        window_size : int, default=10
            Window size for trend calculation

        Returns
        -------
        acceleration : pd.Series
            Trend acceleration values
        """
        # Calculate first derivative (velocity)
        velocity = df.groupby("engine_id")[sensor].diff().fillna(0)

        # Calculate second derivative (acceleration)
        acceleration = velocity.groupby(df["engine_id"]).rolling(window=window_size).std().reset_index(level=0, drop=True)

        logger.debug(f"Calculated trend acceleration for {sensor}")
        return acceleration

    def calculate_degradation_rate(
        self, df: pd.DataFrame, sensor: str, window_size: int = 10
    ) -> pd.Series:
        """
        Calculate rate of sensor value change (degradation speed).

        Parameters
        ----------
        df : pd.DataFrame
            Time-series dataframe
        sensor : str
            Sensor column name
        window_size : int, default=10
            Window size for rate calculation

        Returns
        -------
        degradation_rate : pd.Series
            Rate of change values
        """
        # Calculate rolling slope using linear regression
        def rolling_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope, _, _, _, _ = stats.linregress(x, series)
            return slope

        degradation_rate = df.groupby("engine_id")[sensor].rolling(window=window_size).apply(
            rolling_slope, raw=False
        ).reset_index(level=0, drop=True)

        logger.debug(f"Calculated degradation rate for {sensor}")
        return degradation_rate

    def calculate_sensor_anomaly_score(
        self, df: pd.DataFrame, sensor: str, window_size: int = 10, sigma: float = 3.0
    ) -> pd.Series:
        """
        Calculate anomaly score using modified z-score within rolling window.

        Based on Median Absolute Deviation (MAD) for robustness to outliers.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series dataframe
        sensor : str
            Sensor column name
        window_size : int, default=10
            Rolling window size
        sigma : float, default=3.0
            Threshold for anomaly detection

        Returns
        -------
        anomaly_score : pd.Series
            Anomaly score (0 = normal, higher = more anomalous)
        """
        def rolling_mad(series):
            if len(series) < 2:
                return 0
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            if mad == 0:
                return 0
            return np.abs(series.iloc[-1] - median) / (mad * 1.4826)

        anomaly_score = df.groupby("engine_id")[sensor].rolling(window=window_size).apply(
            rolling_mad, raw=False
        ).reset_index(level=0, drop=True)

        return anomaly_score

    def calculate_multivariate_health_index(
        self, df: pd.DataFrame, sensor_cols: List[str], weights: Optional[List[float]] = None
    ) -> pd.Series:
        """
        Calculate health index from multiple sensors with optional weighting.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with drift columns
        sensor_cols : List[str]
            List of sensor column names (will use drift versions)
        weights : List[float], optional
            Weight for each sensor (default: equal weights)

        Returns
        -------
        health_index : pd.Series
            Weighted health index
        """
        drift_cols = [f"{s}_drift" for s in sensor_cols]

        # Verify drift columns exist
        existing_cols = [col for col in drift_cols if col in df.columns]
        if not existing_cols:
            raise ValueError("No drift columns found. Run calculate_sensor_drift first.")

        if weights is None:
            weights = [1.0 / len(existing_cols)] * len(existing_cols)
        else:
            assert len(weights) == len(existing_cols)
            weights = np.array(weights) / np.sum(weights)  # Normalize

        # Calculate weighted health index
        health_index = np.zeros(len(df))
        for col, weight in zip(existing_cols, weights):
            health_index += weight * df[col].values

        logger.info(f"Calculated multivariate health index from {len(existing_cols)} sensors")
        return pd.Series(health_index, index=df.index)

    def identify_degradation_phases(
        self, df: pd.DataFrame, health_index: pd.Series, percentiles: List[float] = [33, 67]
    ) -> pd.DataFrame:
        """
        Classify degradation phases based on health index percentiles.

        Phases: Healthy (0-33%), Degrading (33-67%), Failed (67-100%)

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        health_index : pd.Series
            Health index values
        percentiles : List[float], default=[33, 67]
            Percentile thresholds for phase boundaries

        Returns
        -------
        df_phases : pd.DataFrame
            Dataframe with phase labels
        """
        df_phases = df.copy()
        df_phases["health_index"] = health_index

        # Calculate percentile thresholds per engine
        quantiles = [p / 100.0 for p in percentiles]
        thresholds = df_phases.groupby("engine_id")["health_index"].quantile(quantiles).unstack()
        
        # Rename columns to 0, 1, 2... for easy integer access
        thresholds.columns = range(len(quantiles))

        # Assign phases
        phases = []
        # Pre-fetch thresholds to dictionary for faster lookup than loc inside loop
        threshold_dict = thresholds.to_dict('index')
        
        for idx, row in df_phases.iterrows():
            engine_id = row["engine_id"]
            hi = row["health_index"]

            if engine_id in threshold_dict:
                engine_thresholds = threshold_dict[engine_id]
                # Use integer keys 0, 1 etc matching the renamed columns
                if hi <= engine_thresholds[0]:
                    phase = "Healthy"
                elif hi <= engine_thresholds[1]:
                    phase = "Degrading"
                else:
                    phase = "Failed"
            else:
                phase = "Unknown"

            phases.append(phase)

        df_phases["degradation_phase"] = phases
        logger.info(f"Identified degradation phases: {df_phases['degradation_phase'].value_counts().to_dict()}")
        return df_phases
