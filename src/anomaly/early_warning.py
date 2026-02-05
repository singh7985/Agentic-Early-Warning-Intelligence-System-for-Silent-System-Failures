"""
Early Warning System

Combines anomaly detection, change-point detection, and degradation labeling
to generate early warnings before system failures. Calculates lead-time and
provides actionable alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class EarlyWarningSystem:
    """
    Early warning system for silent system failures.
    
    Generates warnings by combining:
    - RUL predictions
    - Anomaly detection
    - Change-point detection
    - Degradation labeling
    
    Calculates lead-time before failure and provides actionable alerts.
    """
    
    def __init__(
        self,
        critical_rul: float = 50.0,
        warning_rul: float = 100.0,
        anomaly_threshold: float = 0.5,
        degradation_threshold: float = 0.5,
        min_warning_gap: int = 10,
        alert_levels: Optional[Dict[str, float]] = None
    ):
        """
        Initialize early warning system.
        
        Args:
            critical_rul: RUL threshold for critical alerts (default: 50)
            warning_rul: RUL threshold for warnings (default: 100)
            anomaly_threshold: Anomaly score threshold for alerts
            degradation_threshold: Degradation score threshold for alerts
            min_warning_gap: Minimum cycles between warnings (avoid spam)
            alert_levels: Custom alert level thresholds
        """
        self.critical_rul = critical_rul
        self.warning_rul = warning_rul
        self.anomaly_threshold = anomaly_threshold
        self.degradation_threshold = degradation_threshold
        self.min_warning_gap = min_warning_gap
        
        # Alert levels
        if alert_levels is None:
            self.alert_levels = {
                'critical': 0.8,    # High risk
                'high': 0.6,        # Elevated risk
                'medium': 0.4,      # Moderate risk
                'low': 0.2,         # Low risk
                'info': 0.0         # Informational
            }
        else:
            self.alert_levels = alert_levels
        
        logger.info(f"Initialized EarlyWarningSystem with critical_rul={critical_rul}, "
                   f"warning_rul={warning_rul}")
    
    def generate_warnings(
        self,
        rul_values: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
        degradation_scores: Optional[np.ndarray] = None,
        change_points: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        engine_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate early warnings for all time points.
        
        Args:
            rul_values: RUL predictions
            anomaly_scores: Anomaly scores (optional)
            degradation_scores: Degradation scores (optional)
            change_points: Change point indices (optional)
            timestamps: Time/cycle numbers (optional)
            engine_ids: Engine identifiers (optional)
        
        Returns:
            DataFrame with warning information
        """
        n = len(rul_values)
        
        if timestamps is None:
            timestamps = np.arange(n)
        
        # Calculate composite risk scores
        risk_scores = self._calculate_risk_scores(
            rul_values, anomaly_scores, degradation_scores, change_points, n
        )
        
        # Determine alert levels
        alert_levels = self._determine_alert_levels(risk_scores)
        
        # Generate warnings (only at significant points)
        warnings = self._generate_warning_flags(
            risk_scores, alert_levels, rul_values
        )
        
        # Calculate lead-time (cycles until RUL=0)
        lead_times = rul_values.copy()
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rul': rul_values,
            'risk_score': risk_scores,
            'alert_level': alert_levels,
            'warning_triggered': warnings,
            'lead_time': lead_times
        })
        
        if engine_ids is not None:
            df['engine_id'] = engine_ids
        
        if anomaly_scores is not None:
            df['anomaly_score'] = anomaly_scores
        
        if degradation_scores is not None:
            df['degradation_score'] = degradation_scores
        
        # Mark change points
        df['is_change_point'] = 0
        if change_points is not None:
            df.loc[change_points, 'is_change_point'] = 1
        
        # Summary
        n_warnings = warnings.sum()
        logger.info(f"Generated {n_warnings} warnings ({n_warnings/n:.1%} of data)")
        
        return df
    
    def _calculate_risk_scores(
        self,
        rul_values: np.ndarray,
        anomaly_scores: Optional[np.ndarray],
        degradation_scores: Optional[np.ndarray],
        change_points: Optional[np.ndarray],
        n: int
    ) -> np.ndarray:
        """
        Calculate composite risk scores.
        
        Combines multiple signals into single risk metric.
        """
        risk_scores = np.zeros(n)
        
        # 1. RUL-based risk (50% weight)
        rul_risk = 1 / (1 + np.exp((rul_values - self.warning_rul) / 20))
        risk_scores += rul_risk * 0.5
        
        # 2. Anomaly-based risk (25% weight)
        if anomaly_scores is not None:
            # Normalize
            anomaly_risk = anomaly_scores / (anomaly_scores.max() + 1e-8)
            risk_scores += anomaly_risk * 0.25
        
        # 3. Degradation-based risk (20% weight)
        if degradation_scores is not None:
            risk_scores += degradation_scores * 0.2
        
        # 4. Change-point-based risk (5% weight)
        if change_points is not None:
            change_risk = np.zeros(n)
            for cp in change_points:
                # Spike at change point
                proximity = 10
                for i in range(max(0, cp-proximity), min(n, cp+proximity)):
                    distance = abs(i - cp)
                    change_risk[i] = max(change_risk[i], 1 - distance / proximity)
            risk_scores += change_risk * 0.05
        
        # Clip to [0, 1]
        risk_scores = np.clip(risk_scores, 0, 1)
        
        return risk_scores
    
    def _determine_alert_levels(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Map risk scores to alert levels.
        """
        alert_levels = np.array(['info'] * len(risk_scores), dtype=object)
        
        for level, threshold in sorted(self.alert_levels.items(), key=lambda x: x[1]):
            alert_levels[risk_scores >= threshold] = level
        
        return alert_levels
    
    def _generate_warning_flags(
        self,
        risk_scores: np.ndarray,
        alert_levels: np.ndarray,
        rul_values: np.ndarray
    ) -> np.ndarray:
        """
        Generate binary warning flags.
        
        Only trigger warnings at significant risk increases or critical thresholds.
        """
        warnings = np.zeros(len(risk_scores), dtype=int)
        
        last_warning_idx = -self.min_warning_gap - 1
        
        for i in range(len(risk_scores)):
            # Skip if too soon after last warning
            if i - last_warning_idx < self.min_warning_gap:
                continue
            
            # Trigger conditions
            trigger = False
            
            # 1. Critical RUL threshold
            if rul_values[i] <= self.critical_rul:
                trigger = True
            
            # 2. High or critical alert level
            if alert_levels[i] in ['high', 'critical']:
                trigger = True
            
            # 3. Sudden risk increase
            if i > 0:
                risk_increase = risk_scores[i] - risk_scores[i-1]
                if risk_increase > 0.2:  # 20% jump
                    trigger = True
            
            if trigger:
                warnings[i] = 1
                last_warning_idx = i
        
        return warnings
    
    def get_warning_events(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract warning event information.
        
        Args:
            df: DataFrame from generate_warnings()
        
        Returns:
            List of warning event dictionaries
        """
        events = []
        
        warning_indices = df[df['warning_triggered'] == 1].index
        
        for idx in warning_indices:
            row = df.loc[idx]
            
            event = {
                'index': int(idx),
                'timestamp': row['timestamp'],
                'rul': float(row['rul']),
                'risk_score': float(row['risk_score']),
                'alert_level': str(row['alert_level']),
                'lead_time': float(row['lead_time'])
            }
            
            if 'engine_id' in row:
                event['engine_id'] = row['engine_id']
            
            if 'anomaly_score' in row:
                event['anomaly_score'] = float(row['anomaly_score'])
            
            if 'degradation_score' in row:
                event['degradation_score'] = float(row['degradation_score'])
            
            if 'is_change_point' in row:
                event['is_change_point'] = bool(row['is_change_point'])
            
            events.append(event)
        
        logger.info(f"Extracted {len(events)} warning events")
        
        return events
    
    def calculate_lead_time_statistics(
        self,
        df: pd.DataFrame,
        actual_failure_time: Optional[float] = None
    ) -> Dict:
        """
        Calculate lead-time statistics for warnings.
        
        Args:
            df: DataFrame from generate_warnings()
            actual_failure_time: Actual failure timestamp (if known)
        
        Returns:
            Dictionary of lead-time statistics
        """
        warning_events = self.get_warning_events(df)
        
        if not warning_events:
            logger.warning("No warnings to calculate lead-time")
            return {
                'n_warnings': 0,
                'first_warning_timestamp': None,
                'first_warning_rul': None,
                'first_warning_lead_time': None
            }
        
        # First warning
        first_warning = warning_events[0]
        
        stats = {
            'n_warnings': len(warning_events),
            'first_warning_timestamp': first_warning['timestamp'],
            'first_warning_rul': first_warning['rul'],
            'first_warning_lead_time': first_warning['lead_time'],
            'first_warning_risk_score': first_warning['risk_score'],
            'first_warning_alert_level': first_warning['alert_level']
        }
        
        # If actual failure time is known
        if actual_failure_time is not None:
            stats['actual_failure_time'] = actual_failure_time
            stats['warning_to_failure_time'] = actual_failure_time - first_warning['timestamp']
            stats['warning_accuracy'] = abs(
                first_warning['lead_time'] - stats['warning_to_failure_time']
            ) / stats['warning_to_failure_time'] if stats['warning_to_failure_time'] > 0 else 0
        
        # Statistics across all warnings
        all_lead_times = [e['lead_time'] for e in warning_events]
        stats['mean_lead_time'] = float(np.mean(all_lead_times))
        stats['median_lead_time'] = float(np.median(all_lead_times))
        stats['min_lead_time'] = float(np.min(all_lead_times))
        stats['max_lead_time'] = float(np.max(all_lead_times))
        
        # Alert level distribution
        alert_counts = {}
        for event in warning_events:
            level = event['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        stats['alert_level_distribution'] = alert_counts
        
        return stats
    
    def plot_warnings(
        self,
        df: pd.DataFrame,
        actual_failure_time: Optional[float] = None,
        title: str = "Early Warning System",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Visualize early warning system results.
        
        Args:
            df: DataFrame from generate_warnings()
            actual_failure_time: Actual failure timestamp (if known)
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        timestamps = df['timestamp'].values
        
        # 1. RUL with warnings
        ax = axes[0]
        ax.plot(timestamps, df['rul'], 'b-', alpha=0.7, label='RUL')
        
        # Warning points
        warning_mask = df['warning_triggered'] == 1
        if warning_mask.any():
            ax.scatter(
                timestamps[warning_mask],
                df['rul'][warning_mask],
                c='red', s=100, alpha=0.8, label='Warnings',
                edgecolor='black', linewidths=2, zorder=5, marker='^'
            )
        
        # Thresholds
        ax.axhline(y=self.critical_rul, color='red', linestyle='--',
                  label=f'Critical ({self.critical_rul})', linewidth=2)
        ax.axhline(y=self.warning_rul, color='orange', linestyle='--',
                  label=f'Warning ({self.warning_rul})')
        
        # Actual failure
        if actual_failure_time is not None:
            ax.axvline(x=actual_failure_time, color='black', linestyle='-',
                      linewidth=2, label='Actual Failure')
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('RUL')
        ax.set_title('RUL with Early Warnings')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Risk scores
        ax = axes[1]
        
        # Color by alert level
        for level, color in zip(
            ['info', 'low', 'medium', 'high', 'critical'],
            ['green', 'blue', 'yellow', 'orange', 'red']
        ):
            mask = df['alert_level'] == level
            if mask.any():
                ax.scatter(
                    timestamps[mask],
                    df['risk_score'][mask],
                    c=color, s=20, alpha=0.6, label=level.capitalize()
                )
        
        # Warnings
        if warning_mask.any():
            ax.scatter(
                timestamps[warning_mask],
                df['risk_score'][warning_mask],
                c='red', s=100, alpha=0.8, edgecolor='black',
                linewidths=2, zorder=5, marker='^'
            )
        
        # Threshold lines
        for level, threshold in sorted(self.alert_levels.items(), key=lambda x: x[1], reverse=True):
            if level != 'info':
                ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Scores and Alert Levels')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Lead-time
        ax = axes[2]
        ax.plot(timestamps, df['lead_time'], 'purple', alpha=0.7, label='Lead-Time')
        
        if warning_mask.any():
            ax.scatter(
                timestamps[warning_mask],
                df['lead_time'][warning_mask],
                c='red', s=100, alpha=0.8, edgecolor='black',
                linewidths=2, zorder=5, marker='^', label='Warning Points'
            )
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Lead-Time (cycles)')
        ax.set_title('Lead-Time Before Failure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Timeline of events
        ax = axes[3]
        
        # RUL baseline
        ax.plot(timestamps, df['rul'], 'gray', alpha=0.3, linewidth=1)
        
        # Warning events
        events = self.get_warning_events(df)
        if events:
            event_times = [e['timestamp'] for e in events]
            event_risks = [e['risk_score'] for e in events]
            event_levels = [e['alert_level'] for e in events]
            
            level_colors = {
                'critical': 'red', 'high': 'orange', 'medium': 'yellow',
                'low': 'blue', 'info': 'green'
            }
            
            for i, (t, risk, level) in enumerate(zip(event_times, event_risks, event_levels)):
                ax.axvline(x=t, color=level_colors.get(level, 'gray'),
                          alpha=0.7, linestyle='--')
                ax.text(t, ax.get_ylim()[1]*0.9, f'W{i+1}',
                       rotation=90, verticalalignment='top', fontsize=8)
        
        # Actual failure
        if actual_failure_time is not None:
            ax.axvline(x=actual_failure_time, color='black',
                      linewidth=2, label='Failure')
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('RUL')
        ax.set_title('Warning Timeline')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def export_warnings(
        self,
        df: pd.DataFrame,
        filepath: str,
        format: str = 'csv'
    ):
        """
        Export warnings to file.
        
        Args:
            df: DataFrame from generate_warnings()
            filepath: Output file path
            format: Export format ('csv', 'json', 'html')
        """
        events = self.get_warning_events(df)
        
        if format == 'csv':
            events_df = pd.DataFrame(events)
            events_df.to_csv(filepath, index=False)
        elif format == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(events, f, indent=2)
        elif format == 'html':
            events_df = pd.DataFrame(events)
            events_df.to_html(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(events)} warnings to {filepath}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get early warning system statistics.
        
        Args:
            df: DataFrame from generate_warnings()
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(df),
            'n_warnings': int(df['warning_triggered'].sum()),
            'warning_rate': float(df['warning_triggered'].mean()),
            'mean_risk_score': float(df['risk_score'].mean()),
            'max_risk_score': float(df['risk_score'].max())
        }
        
        # Alert level distribution
        alert_counts = df['alert_level'].value_counts().to_dict()
        stats['alert_level_distribution'] = alert_counts
        
        # Lead-time statistics
        lead_time_stats = self.calculate_lead_time_statistics(df)
        stats.update(lead_time_stats)
        
        return stats
    
    def save(self, filepath: str):
        """Save warning system to disk"""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Saved warning system to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'EarlyWarningSystem':
        """Load warning system from disk"""
        import joblib
        system = joblib.load(filepath)
        logger.info(f"Loaded warning system from {filepath}")
        return system
