"""
MLflow Integration for Model Tracking

Provides utilities for logging experiments, models, and metrics to MLflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking wrapper."""

    def __init__(
        self,
        experiment_name: str = "RUL_Prediction",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Parameters
        ----------
        experiment_name : str, default="RUL_Prediction"
            Name of MLflow experiment
        tracking_uri : str, optional
            MLflow tracking URI (default: local ./mlruns)
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"Experiment ID: {self.experiment.experiment_id}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name for this run
        tags : dict, optional
            Tags for this run

        Returns
        -------
        run : ActiveRun
            MLflow active run context manager
        """
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.

        Parameters
        ----------
        params : dict
            Parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Parameters
        ----------
        metrics : dict
            Metrics to log
        step : int, optional
            Step number (for time series metrics)
        """
        for key, value in metrics.items():
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
        logger.info(f"Logged {len(metrics)} metrics")

    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
    ) -> None:
        """
        Log scikit-learn model.

        Parameters
        ----------
        model : sklearn model
            Trained model
        artifact_path : str, default="model"
            Path within run artifacts
        signature : ModelSignature, optional
            Model signature
        """
        mlflow.sklearn.log_model(model, artifact_path, signature=signature)
        logger.info(f"Logged sklearn model to {artifact_path}")

    def log_pytorch_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
    ) -> None:
        """
        Log PyTorch model.

        Parameters
        ----------
        model : nn.Module
            Trained model
        artifact_path : str, default="model"
            Path within run artifacts
        signature : ModelSignature, optional
            Model signature
        """
        mlflow.pytorch.log_model(model, artifact_path, signature=signature)
        logger.info(f"Logged PyTorch model to {artifact_path}")

    def log_figure(self, fig: plt.Figure, artifact_file: str) -> None:
        """
        Log matplotlib figure.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure
        artifact_file : str
            Filename for artifact (e.g., 'predictions.png')
        """
        mlflow.log_figure(fig, artifact_file)
        logger.info(f"Logged figure: {artifact_file}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact file.

        Parameters
        ----------
        local_path : str
            Path to local file
        artifact_path : str, optional
            Path within run artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        """
        Log dictionary as JSON artifact.

        Parameters
        ----------
        dictionary : dict
            Dictionary to log
        artifact_file : str
            Filename for artifact (e.g., 'config.json')
        """
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dictionary: {artifact_file}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for current run.

        Parameters
        ----------
        tags : dict
            Tags to set
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        logger.info(f"Set {len(tags)} tags")

    def get_run_id(self) -> str:
        """Get current run ID."""
        run = mlflow.active_run()
        return run.info.run_id if run else None

    def end_run(self) -> None:
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")


def log_xgboost_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    run_name: str = "XGBoost",
    tracker: Optional[MLflowTracker] = None,
) -> str:
    """
    Log XGBoost model with all artifacts.

    Parameters
    ----------
    model : XGBoostRULPredictor
        Trained model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    params : dict
        Model parameters
    metrics : dict
        Evaluation metrics
    run_name : str, default="XGBoost"
        Run name
    tracker : MLflowTracker, optional
        MLflow tracker instance

    Returns
    -------
    run_id : str
        MLflow run ID
    """
    if tracker is None:
        tracker = MLflowTracker()
    
    with tracker.start_run(run_name=run_name):
        # Log parameters
        tracker.log_params(params)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_sklearn_model(model.model, artifact_path="model")
        
        # Log feature importance
        if hasattr(model, 'feature_importance_'):
            importance_dict = {
                f"feature_{i}_importance": float(imp) 
                for i, imp in enumerate(model.feature_importance_)
            }
            tracker.log_dict(importance_dict, "feature_importance.json")
        
        # Set tags
        tracker.set_tags({
            "model_type": "XGBoost",
            "framework": "sklearn",
            "task": "regression",
            "target": "RUL",
        })
        
        run_id = tracker.get_run_id()
        logger.info(f"XGBoost model logged with run_id: {run_id}")
        
        return run_id


def log_deep_learning_model(
    trainer: Any,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    training_history: Dict,
    run_name: Optional[str] = None,
    tracker: Optional[MLflowTracker] = None,
) -> str:
    """
    Log deep learning model with all artifacts.

    Parameters
    ----------
    trainer : DeepLearningTrainer
        Trained model trainer
    model_type : str
        Model type ('LSTM', 'TCN', etc.)
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    params : dict
        Model parameters
    metrics : dict
        Evaluation metrics
    training_history : dict
        Training history
    run_name : str, optional
        Run name
    tracker : MLflowTracker, optional
        MLflow tracker instance

    Returns
    -------
    run_id : str
        MLflow run ID
    """
    if tracker is None:
        tracker = MLflowTracker()
    
    if run_name is None:
        run_name = model_type
    
    with tracker.start_run(run_name=run_name):
        # Log parameters
        tracker.log_params(params)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log training losses
        for epoch, (train_loss, val_loss) in enumerate(zip(
            training_history['train_losses'], 
            training_history.get('val_losses', [])
        )):
            tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, step=epoch)
        
        # Log model
        tracker.log_pytorch_model(trainer.model, artifact_path="model")
        
        # Set tags
        tracker.set_tags({
            "model_type": model_type,
            "framework": "pytorch",
            "task": "regression",
            "target": "RUL",
        })
        
        run_id = tracker.get_run_id()
        logger.info(f"{model_type} model logged with run_id: {run_id}")
        
        return run_id
