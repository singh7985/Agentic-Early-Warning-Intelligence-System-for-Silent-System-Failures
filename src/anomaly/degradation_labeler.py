"""
Degradation Labeling

Labels periods of silent degradation by combining anomaly detection,
change-point detection, and RUL predictions to identify when systems
begin degrading before visible failure symptoms appear.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class DegradationLabeler:
    """
    Labels silent degradation periods using multiple signals.
    
    Combines:
    - Anomaly scores
    - Change points
    - RUL predictions
    - Sensor trends
    
    to identify when equipment enters degradation phase before failure.
    """
    
    def __init__(
        self,
        rul_threshold: float = 100.0,
        anomaly_window: int = 10,
        anomaly_rate_threshold: float = 0.3,
        change_point_proximity: int = 20,
        min_degradation_length: int = 5
    ):
        """
        Initialize degradation labeler.
        
        Args:
            rul_threshold: RUL below which to consider degradation (default: 100 cycles)
            anomaly_window: Window for calculating rolling anomaly rate
            anomaly_rate_threshold: Minimum anomaly rate to flag degradation (default: 0.3)
            change_point_proximity: Cycles after change point to consider degradation
            min_degradation_length: Minimum length of degradation period
        """
        self.rul_threshold = rul_threshold
        self.anomaly_window = anomaly_window
        self.anomaly_rate_threshold = anomaly_rate_threshold
        self.change_point_proximity = change_point_proximity
        self.min_degradation_length = min_degradation_length
        
        logger.info(f"Initialized DegradationLabeler with RUL threshold={rul_threshold}")
    
    def label_degradation(
        self,
        rul_values: np.ndarray,
        anomaly_flags: Optional[np.ndarray] = None,
        anomaly_scores: Optional[np.ndarray] = None,
        change_points: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Label degradation periods using multiple signals.
        
        Args:
            rul_values: RUL predictions for each time step
            anomaly_flags: Binary anomaly indicators (optional)
            anomaly_scores: Continuous anomaly scores (optional)
            change_points: Change point indices (optional)
            timestamps: Time/cycle numbers (optional)
        
        Returns:
            DataFrame with degradation labels and scores
        """
        n = len(rul_values)
        
        if timestamps is None:
            timestamps = np.arange(n)
        
        # Initialize degradation scores
        degradation_scores = np.zeros(n)
        
        # 1. RUL-based scoring (normalized)
        rul_score = self._score_rul(rul_values)
        degradation_scores += rul_score * 0.4  # 40% weight
        
        # 2. Anomaly-based scoring
        if anomaly_flags is not None:
            anomaly_score = self._score_anomalies(anomaly_flags, anomaly_scores)
            degradation_scores += anomaly_score * 0.3  # 30% weight
        
        # 3. Change-point-based scoring
        if change_points is not None:
            change_score = self._score_change_points(n, change_points)
            degradation_scores += change_score * 0.3  # 30% weight
        
        # Normalize to [0, 1]
        if degradation_scores.max() > 0:
            degradation_scores = degradation_scores / degradation_scores.max()
        
        # Binary degradation labels (threshold at 0.5)
        degradation_labels = (degradation_scores >= 0.5).astype(int)
        
        # Filter short degradation periods
        degradation_labels = self._filter_short_periods(degradation_labels)
        
        # Identify degradation phases
        phases = self._identify_phases(degradation_labels)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rul': rul_values,
            'degradation_score': degradation_scores,
            'is_degrading': degradation_labels,
            'phase': phases
        })
        
        if anomaly_flags is not None:
            df['is_anomaly'] = anomaly_flags
        
        if anomaly_scores is not None:
            df['anomaly_score'] = anomaly_scores
        
        # Add change point markers
        df['is_change_point'] = 0
        if change_points is not None:
            df.loc[change_points, 'is_change_point'] = 1
        
        # Summary statistics
        n_degrading = np.sum(degradation_labels)
        n_phases = len(np.unique(phases[phases > 0]))
        
        logger.info(f"Labeled {n_degrading} degrading samples ({n_degrading/n:.1%}) in {n_phases} phases")
        
        return df
    
    def _score_rul(self, rul_values: np.ndarray) -> np.ndarray:
        """
        Score degradation based on RUL values.
        
        Lower RUL = higher degradation score
        """
        rul_values = np.asarray(rul_values)
        
        # Inverse sigmoid scoring
        scores = 1 / (1 + np.exp((rul_values - self.rul_threshold) / 20))
        
        return scores
    
    def _score_anomalies(
        self,
        anomaly_flags: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Score degradation based on anomaly patterns.
        
        High anomaly rate in recent window = higher score
        """
        anomaly_flags = np.asarray(anomaly_flags)
        n = len(anomaly_flags)
        
        # Rolling anomaly rate
        rolling_rate = pd.Series(anomaly_flags).rolling(
            window=self.anomaly_window, min_periods=1
        ).mean().values
        
        # Combine with scores if available
        if anomaly_scores is not None:
            # Normalize scores
            scores_norm = anomaly_scores / (anomaly_scores.max() + 1e-8)
            # Weighted combination
            combined = 0.6 * rolling_rate + 0.4 * scores_norm
        else:
            combined = rolling_rate
        
        return combined
    
    def _score_change_points(self, n: int, change_points: np.ndarray) -> np.ndarray:
        """
        Score degradation based on proximity to change points.
        
        Points after change points get higher scores (exponential decay)
        """
        scores = np.zeros(n)
        
        for cp in change_points:
            # Exponential decay from change point
            for i in range(cp, min(cp + self.change_point_proximity, n)):
                distance = i - cp
                score = np.exp(-distance / (self.change_point_proximity / 3))
                scores[i] = max(scores[i], score)
        
        return scores
    
    def _filter_short_periods(self, labels: np.ndarray) -> np.ndarray:
        """
        Remove degradation periods shorter than minimum length.
        """
        labels = labels.copy()
        
        # Find runs of 1s
        runs = []
        current_run_start = None
        
        for i, val in enumerate(labels):
            if val == 1 and current_run_start is None:
                current_run_start = i
            elif val == 0 and current_run_start is not None:
                runs.append((current_run_start, i))
                current_run_start = None
        
        # Handle run at end
        if current_run_start is not None:
            runs.append((current_run_start, len(labels)))
        
        # Filter short runs
        for start, end in runs:
            if end - start < self.min_degradation_length:
                labels[start:end] = 0
        
        return labels
    
    def _identify_phases(self, labels: np.ndarray) -> np.ndarray:
        """
        Identify distinct degradation phases.
        
        Returns:
            Phase numbers (0 = normal, 1+ = degradation phase number)
        """
        phases = np.zeros(len(labels), dtype=int)
        
        phase_num = 0
        in_phase = False
        
        for i, label in enumerate(labels):
            if label == 1:
                if not in_phase:
                    phase_num += 1
                    in_phase = True
                phases[i] = phase_num
            else:
                in_phase = False
        
        return phases
    
    def get_degradation_periods(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract degradation period information.
        
        Args:
            df: DataFrame from label_degradation()
        
        Returns:
            List of degradation period dictionaries
        """
        periods = []
        
        unique_phases = sorted(df[df['phase'] > 0]['phase'].unique())
        
        for phase in unique_phases:
            phase_data = df[df['phase'] == phase]
            
            period = {
                'phase': int(phase),
                'start_idx': int(phase_data.index[0]),
                'end_idx': int(phase_data.index[-1]),
                'start_timestamp': phase_data['timestamp'].iloc[0],
                'end_timestamp': phase_data['timestamp'].iloc[-1],
                'duration': int(len(phase_data)),
                'start_rul': float(phase_data['rul'].iloc[0]),
                'end_rul': float(phase_data['rul'].iloc[-1]),
                'rul_drop': float(phase_data['rul'].iloc[0] - phase_data['rul'].iloc[-1]),
                'mean_degradation_score': float(phase_data['degradation_score'].mean()),
                'max_degradation_score': float(phase_data['degradation_score'].max())
            }
            
            if 'is_anomaly' in phase_data.columns:
                period['anomaly_rate'] = float(phase_data['is_anomaly'].mean())
                period['n_anomalies'] = int(phase_data['is_anomaly'].sum())
            
            if 'is_change_point' in phase_data.columns:
                period['n_change_points'] = int(phase_data['is_change_point'].sum())
            
            periods.append(period)
        
        logger.info(f"Extracted {len(periods)} degradation periods")
        
        return periods
    
    def plot_degradation_labels(
        self,
        df: pd.DataFrame,
        title: str = "Degradation Labeling",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Visualize degradation labeling results.
        
        Args:
            df: DataFrame from label_degradation()
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        timestamps = df['timestamp'].values
        
        # 1. RUL with degradation highlighting
        ax = axes[0]
        ax.plot(timestamps, df['rul'], 'b-', alpha=0.7, label='RUL')
        
        # Highlight degradation periods
        for phase in df[df['phase'] > 0]['phase'].unique():
            phase_data = df[df['phase'] == phase]
            ax.axvspan(
                phase_data['timestamp'].iloc[0],
                phase_data['timestamp'].iloc[-1],
                alpha=0.3, color='red', label='Degradation' if phase == 1 else ''
            )
        
        ax.axhline(y=self.rul_threshold, color='orange', linestyle='--',
                  label=f'RUL Threshold ({self.rul_threshold})')
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('RUL')
        ax.set_title('RUL with Degradation Periods')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Degradation scores
        ax = axes[1]
        ax.plot(timestamps, df['degradation_score'], 'purple', alpha=0.7,
               label='Degradation Score')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax.fill_between(timestamps, 0, df['degradation_score'],
                       where=df['is_degrading']==1, alpha=0.3, color='red')
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Score')
        ax.set_title('Degradation Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Anomalies (if available)
        if 'is_anomaly' in df.columns:
            ax = axes[2]
            
            # Rolling anomaly rate
            rolling_rate = df['is_anomaly'].rolling(
                window=self.anomaly_window, min_periods=1
            ).mean()
            
            ax.plot(timestamps, rolling_rate, 'orange', alpha=0.7,
                   label=f'Rolling Anomaly Rate (w={self.anomaly_window})')
            ax.axhline(y=self.anomaly_rate_threshold, color='r', linestyle='--',
                      label=f'Threshold ({self.anomaly_rate_threshold})')
            
            # Scatter anomalies
            anomaly_times = timestamps[df['is_anomaly'] == 1]
            ax.scatter(anomaly_times, [0.05]*len(anomaly_times),
                      c='red', s=10, alpha=0.5, label='Anomalies')
            
            ax.set_xlabel('Time/Cycle')
            ax.set_ylabel('Anomaly Rate')
            ax.set_title('Anomaly Patterns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Phase diagram
        ax = axes[3]
        
        # Create color map for phases
        unique_phases = df['phase'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_phases)))
        
        for phase, color in zip(sorted(unique_phases), colors):
            phase_mask = df['phase'] == phase
            if phase == 0:
                label = 'Normal'
            else:
                phase_data = df[phase_mask]
                label = f'Phase {phase} (RUL: {phase_data["rul"].iloc[0]:.0f}→{phase_data["rul"].iloc[-1]:.0f})'
            
            ax.scatter(
                timestamps[phase_mask],
                [phase]*np.sum(phase_mask),
                c=[color], s=20, alpha=0.7, label=label
            )
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Phase')
        ax.set_title('Degradation Phases')
        ax.set_yticks(sorted(unique_phases))
        if len(unique_phases) <= 10:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get degradation statistics.
        
        Args:
            df: DataFrame from label_degradation()
        
        Returns:
            Dictionary of statistics
        """
        periods = self.get_degradation_periods(df)
        
        stats = {
            'total_samples': len(df),
            'degrading_samples': int(df['is_degrading'].sum()),
            'degradation_rate': float(df['is_degrading'].mean()),
            'n_phases': len(periods),
            'mean_phase_duration': float(np.mean([p['duration'] for p in periods])) if periods else 0,
            'total_degradation_duration': int(df['is_degrading'].sum()),
            'mean_degradation_score': float(df[df['is_degrading']==1]['degradation_score'].mean()) if any(df['is_degrading']) else 0
        }
        
        if 'is_anomaly' in df.columns:
            stats['overall_anomaly_rate'] = float(df['is_anomaly'].mean())
            stats['anomaly_rate_in_degradation'] = float(
                df[df['is_degrading']==1]['is_anomaly'].mean()
            ) if any(df['is_degrading']) else 0
        
        if 'is_change_point' in df.columns:
            stats['n_change_points'] = int(df['is_change_point'].sum())
        
        return stats
    
    def save(self, filepath: str):
        """Save labeler to disk"""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Saved labeler to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DegradationLabeler':
        """Load labeler from disk"""
        import joblib
        labeler = joblib.load(filepath)
        logger.info(f"Loaded labeler from {filepath}")
        return labeler
