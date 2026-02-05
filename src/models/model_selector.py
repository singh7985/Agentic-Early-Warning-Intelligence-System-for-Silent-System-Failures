"""
Model Selection and Comparison

Provides utilities for comparing multiple models and selecting the best one
based on performance and stability criteria.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelSelector:
    """Compare and select best model based on multiple criteria."""

    def __init__(self, primary_metric: str = 'rmse', lower_is_better: bool = True):
        """
        Initialize model selector.

        Parameters
        ----------
        primary_metric : str, default='rmse'
            Primary metric for ranking
        lower_is_better : bool, default=True
            Whether lower metric values are better
        """
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better
        self.models_results = {}
        self.stability_scores = {}

    def add_model_results(
        self,
        model_name: str,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        inference_time: float,
        training_time: float,
        model_size_mb: Optional[float] = None,
    ) -> None:
        """
        Add model results.

        Parameters
        ----------
        model_name : str
            Name of the model
        train_metrics : dict
            Training metrics
        test_metrics : dict
            Test metrics
        inference_time : float
            Average inference time (seconds)
        training_time : float
            Total training time (seconds)
        model_size_mb : float, optional
            Model size in MB
        """
        self.models_results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'inference_time': inference_time,
            'training_time': training_time,
            'model_size_mb': model_size_mb,
        }
        
        # Calculate stability (train-test gap)
        self._calculate_stability(model_name)
        
        logger.info(f"Added results for {model_name}")

    def _calculate_stability(self, model_name: str) -> None:
        """Calculate stability score (train-test performance gap)."""
        results = self.models_results[model_name]
        train_metrics = results['train']
        test_metrics = results['test']
        
        # Calculate gaps for each metric
        gaps = {}
        for metric in train_metrics.keys():
            if metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]
                
                # For R2, calculate absolute difference
                if metric == 'r2':
                    gap = abs(train_val - test_val)
                else:
                    # For error metrics, calculate percentage increase
                    gap = abs((test_val - train_val) / (train_val + 1e-8)) * 100
                
                gaps[metric] = gap
        
        # Overall stability score (lower is better)
        stability_score = np.mean(list(gaps.values()))
        self.stability_scores[model_name] = {
            'overall': stability_score,
            'gaps': gaps,
        }

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all models.

        Returns
        -------
        comparison_df : pd.DataFrame
            Comparison table with all metrics
        """
        rows = []
        
        for model_name, results in self.models_results.items():
            row = {'model': model_name}
            
            # Test metrics (primary)
            for metric, value in results['test'].items():
                row[f'test_{metric}'] = value
            
            # Train metrics
            for metric, value in results['train'].items():
                row[f'train_{metric}'] = value
            
            # Stability
            row['stability_score'] = self.stability_scores[model_name]['overall']
            
            # Performance
            row['inference_time'] = results['inference_time']
            row['training_time'] = results['training_time']
            
            if results['model_size_mb'] is not None:
                row['model_size_mb'] = results['model_size_mb']
            
            rows.append(row)
        
        comparison_df = pd.DataFrame(rows)
        
        # Sort by primary metric
        test_metric_col = f'test_{self.primary_metric}'
        if test_metric_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                test_metric_col, 
                ascending=self.lower_is_better
            )
        
        return comparison_df

    def select_best_model(
        self,
        stability_weight: float = 0.3,
        performance_weight: float = 0.7,
        max_inference_time: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        """
        Select best model based on weighted criteria.

        Parameters
        ----------
        stability_weight : float, default=0.3
            Weight for stability score
        performance_weight : float, default=0.7
            Weight for performance
        max_inference_time : float, optional
            Maximum allowed inference time (filter)

        Returns
        -------
        best_model : str
            Name of best model
        scores : dict
            Detailed scores for all models
        """
        if not self.models_results:
            raise ValueError("No models added yet")
        
        scores = {}
        
        # Get all test metrics for the primary metric
        test_metrics = {
            name: results['test'][self.primary_metric]
            for name, results in self.models_results.items()
        }
        
        # Normalize metrics (0-1 scale)
        metric_values = np.array(list(test_metrics.values()))
        if self.lower_is_better:
            normalized_metrics = 1 - (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)
        else:
            normalized_metrics = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)
        
        # Normalize stability scores (lower is better)
        stability_values = np.array([
            self.stability_scores[name]['overall'] 
            for name in test_metrics.keys()
        ])
        normalized_stability = 1 - (stability_values - stability_values.min()) / (stability_values.max() - stability_values.min() + 1e-8)
        
        # Calculate combined scores
        for i, model_name in enumerate(test_metrics.keys()):
            # Filter by inference time if specified
            if max_inference_time is not None:
                inference_time = self.models_results[model_name]['inference_time']
                if inference_time > max_inference_time:
                    logger.info(f"Excluding {model_name} due to inference time: {inference_time:.4f}s > {max_inference_time}s")
                    continue
            
            performance_score = normalized_metrics[i]
            stability_score = normalized_stability[i]
            
            combined_score = (
                performance_weight * performance_score +
                stability_weight * stability_score
            )
            
            scores[model_name] = {
                'performance_score': performance_score,
                'stability_score': stability_score,
                'combined_score': combined_score,
                'test_metric': test_metrics[model_name],
                'stability_gap': stability_values[i],
            }
        
        # Select best model
        best_model = max(scores.keys(), key=lambda k: scores[k]['combined_score'])
        
        logger.info(f"\nModel Selection Results:")
        logger.info(f"  Best Model: {best_model}")
        logger.info(f"  Combined Score: {scores[best_model]['combined_score']:.4f}")
        logger.info(f"  Test {self.primary_metric}: {scores[best_model]['test_metric']:.4f}")
        logger.info(f"  Stability Gap: {scores[best_model]['stability_gap']:.2f}%")
        
        return best_model, scores

    def plot_comparison(
        self,
        metrics: List[str] = ['rmse', 'mae', 'r2'],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Create comprehensive comparison plot.

        Parameters
        ----------
        metrics : list, default=['rmse', 'mae', 'r2']
            Metrics to plot
        save_path : str, optional
            Path to save figure
        figsize : tuple, default=(16, 10)
            Figure size

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        comparison_df = self.get_comparison_table()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Primary metric comparison
        test_col = f'test_{self.primary_metric}'
        train_col = f'train_{self.primary_metric}'
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, comparison_df[train_col], width, 
                      label='Train', color='steelblue', edgecolor='black')
        axes[0, 0].bar(x + width/2, comparison_df[test_col], width,
                      label='Test', color='coral', edgecolor='black')
        axes[0, 0].set_ylabel(self.primary_metric.upper())
        axes[0, 0].set_title(f'{self.primary_metric.upper()} Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Stability scores
        axes[0, 1].bar(comparison_df['model'], comparison_df['stability_score'],
                      color='lightgreen', edgecolor='black')
        axes[0, 1].set_ylabel('Stability Gap (%)')
        axes[0, 1].set_title('Model Stability (Lower is Better)', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Inference time
        axes[0, 2].bar(comparison_df['model'], comparison_df['inference_time'] * 1000,
                      color='gold', edgecolor='black')
        axes[0, 2].set_ylabel('Inference Time (ms)')
        axes[0, 2].set_title('Inference Speed', fontweight='bold')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training time
        axes[1, 0].bar(comparison_df['model'], comparison_df['training_time'],
                      color='purple', edgecolor='black', alpha=0.7)
        axes[1, 0].set_ylabel('Training Time (s)')
        axes[1, 0].set_title('Training Time', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Multiple metrics radar (if 3+ metrics)
        if len(metrics) >= 3:
            # Prepare data for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            ax = plt.subplot(2, 3, 5, projection='polar')
            
            for model_name in comparison_df['model']:
                values = []
                for metric in metrics:
                    test_col = f'test_{metric}'
                    if test_col in comparison_df.columns:
                        val = comparison_df[comparison_df['model'] == model_name][test_col].values[0]
                        # Normalize (simple min-max)
                        col_vals = comparison_df[test_col].values
                        normalized = (val - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-8)
                        if metric != 'r2':  # For error metrics, invert
                            normalized = 1 - normalized
                        values.append(normalized)
                
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.upper() for m in metrics])
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Metric Performance\n(Normalized)', fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
        
        # Plot 6: Summary table
        axes[1, 2].axis('off')
        
        # Get best model
        best_model, scores = self.select_best_model()
        
        summary_text = f"""
        MODEL COMPARISON SUMMARY
        {'=' * 40}
        
        Best Model: {best_model}
        Combined Score: {scores[best_model]['combined_score']:.4f}
        
        Test {self.primary_metric.upper()}: {scores[best_model]['test_metric']:.4f}
        Stability Gap: {scores[best_model]['stability_gap']:.2f}%
        
        Ranking:
        """
        
        for i, (name, score) in enumerate(
            sorted(scores.items(), key=lambda x: x[1]['combined_score'], reverse=True), 1
        ):
            summary_text += f"\n  {i}. {name} ({score['combined_score']:.3f})"
        
        axes[1, 2].text(0.1, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig

    def export_results(self, filepath: str) -> None:
        """
        Export comparison results to CSV.

        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        comparison_df = self.get_comparison_table()
        comparison_df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
