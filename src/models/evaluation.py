"""
Model Evaluation and Visualization

Provides comprehensive evaluation metrics and visualization utilities
for RUL prediction models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class RULEvaluator:
    """Comprehensive evaluation for RUL prediction models."""

    def __init__(self, model_name: str = "Model"):
        """
        Initialize evaluator.

        Parameters
        ----------
        model_name : str, default="Model"
            Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics_history = []

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True RUL values
        y_pred : np.ndarray
            Predicted RUL values
        prefix : str, optional
            Prefix for metric names (e.g., 'train_', 'test_')

        Returns
        -------
        metrics : dict
            Dictionary of metrics
        """
        # Basic regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # NASA Scoring Function (Asymmetric penalty)
        # d = y_pred - y_true
        # if d < 0 (early prediction): penalty = exp(-d/13) - 1
        # if d >= 0 (late prediction): penalty = exp(d/10) - 1
        d = y_pred - y_true
        nasa_score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
        
        # Residuals
        residuals = y_true - y_pred
        
        # Additional metrics
        max_error = np.max(np.abs(residuals))
        median_error = np.median(np.abs(residuals))
        std_error = np.std(residuals)
        
        # Early/late prediction metrics
        early_predictions = np.sum(y_pred > y_true)
        late_predictions = np.sum(y_pred < y_true)
        early_pct = (early_predictions / len(y_true)) * 100
        late_pct = (late_predictions / len(y_true)) * 100
        
        metrics = {
            f'{prefix}rmse': rmse,
            f'{prefix}mae': mae,
            f'{prefix}r2': r2,
            f'{prefix}nasa_score': nasa_score,
            f'{prefix}max_error': max_error,
            f'{prefix}median_error': median_error,
            f'{prefix}std_error': std_error,
            f'{prefix}early_pct': early_pct,
            f'{prefix}late_pct': late_pct,
        }
        
        self.metrics_history.append(metrics)
        
        logger.info(f"{self.model_name} {prefix}metrics:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  NASA Score: {nasa_score:.2f}")
        
        return metrics

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Create comprehensive prediction visualization.

        Parameters
        ----------
        y_true : np.ndarray
            True RUL values
        y_pred : np.ndarray
            Predicted RUL values
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : tuple, default=(14, 10)
            Figure size

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        if title is None:
            title = f"{self.model_name} - RUL Prediction Analysis"
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Predicted vs Actual (scatter)
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True RUL')
        axes[0, 0].set_ylabel('Predicted RUL')
        axes[0, 0].set_title('Predicted vs Actual RUL', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20, color='coral')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted RUL')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual histogram
        axes[0, 2].hist(residuals, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('Residual')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residual Distribution', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Time series (first 500 samples)
        n_samples = min(500, len(y_true))
        axes[1, 0].plot(y_true[:n_samples], label='True RUL', linewidth=2, alpha=0.7, color='blue')
        axes[1, 0].plot(y_pred[:n_samples], label='Predicted RUL', linewidth=2, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('RUL')
        axes[1, 0].set_title(f'Time Series Comparison (First {n_samples} Samples)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Error distribution by RUL range
        rul_bins = np.linspace(y_true.min(), y_true.max(), 10)
        bin_indices = np.digitize(y_true, rul_bins)
        errors_by_bin = [np.abs(residuals[bin_indices == i]).mean() 
                        for i in range(1, len(rul_bins))]
        bin_centers = (rul_bins[:-1] + rul_bins[1:]) / 2
        
        axes[1, 1].bar(bin_centers, errors_by_bin, width=np.diff(rul_bins)[0] * 0.8, 
                      color='gold', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('RUL Range')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by RUL Range', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Metrics summary
        metrics = self.calculate_metrics(y_true, y_pred)
        axes[1, 2].axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        {'=' * 30}
        RMSE:           {metrics['rmse']:.2f}
        MAE:            {metrics['mae']:.2f}
        R² Score:       {metrics['r2']:.4f}
        NASA Score:     {metrics['nasa_score']:.2f}
        
        Max Error:      {metrics['max_error']:.2f}
        Median Error:   {metrics['median_error']:.2f}
        Std Error:      {metrics['std_error']:.2f}
        
        Early Pred:     {metrics['early_pct']:.1f}%
        Late Pred:      {metrics['late_pct']:.1f}%
        
        Samples:        {len(y_true):,}
        """
        
        axes[1, 2].text(0.1, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training history.

        Parameters
        ----------
        train_losses : list
            Training losses
        val_losses : list, optional
            Validation losses
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        if title is None:
            title = f"{self.model_name} - Training History"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label='Training Loss', linewidth=2, marker='o', markersize=4)
        
        if val_losses:
            ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        return fig

    def create_comparison_table(
        self, models_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple models.

        Parameters
        ----------
        models_results : dict
            Dictionary mapping model names to their metrics

        Returns
        -------
        comparison_df : pd.DataFrame
            Comparison table
        """
        comparison_df = pd.DataFrame(models_results).T
        
        # Sort by RMSE (lower is better)
        if 'rmse' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('rmse')
        
        logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")
        
        return comparison_df

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['rmse', 'mae', 'r2'],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison of multiple models.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Comparison dataframe
        metrics : list, default=['rmse', 'mae', 'r2']
            Metrics to plot
        save_path : str, optional
            Path to save figure

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
                axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_xlabel('Model')
                axes[i].grid(True, alpha=0.3, axis='y')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig
