"""
MLflow Experiment Tracking: Log models, metrics, and parameters

Purpose:
- Track all experiments with parameters and metrics
- Version and register models
- Compare experiment results
- Enable reproducibility
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment"""
    experiment_name: str
    run_name: str
    model_type: str  # 'sklearn', 'pytorch', 'rag', 'agent'
    system_variant: str  # 'ml_only', 'ml_rag', 'ml_rag_agents'
    description: str = ""


@dataclass
class ModelMetrics:
    """Metrics for a model"""
    mae: float
    rmse: float
    r_squared: float
    mape: float
    f1_score: float
    precision: float
    recall: float
    auc: float


class MLflowTracker:
    """
    Track experiments, models, and metrics with MLflow.
    
    Features:
    - Log parameters and metrics
    - Register models
    - Compare experiments
    - Track artifacts
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize MLflow tracker."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"MLflow tracker initialized: {tracking_uri}")

    def create_experiment(self, experiment_name: str) -> str:
        """Create new experiment."""
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e:
            logger.warning(f"Experiment already exists: {experiment_name}")
            return mlflow.get_experiment_by_name(experiment_name).experiment_id

    def start_run(self, config: ExperimentConfig) -> str:
        """Start new MLflow run."""
        experiment_id = self.create_experiment(config.experiment_name)
        
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=config.run_name,
        )
        
        # Log configuration
        mlflow.log_params({
            'model_type': config.model_type,
            'system_variant': config.system_variant,
            'description': config.description,
            'timestamp': datetime.now().isoformat(),
        })
        
        logger.info(f"Started run: {config.run_name}")
        return run.info.run_id

    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics at step {step}")

    def log_model_metrics(self, metrics: ModelMetrics):
        """Log model evaluation metrics."""
        metrics_dict = {
            'mae': metrics.mae,
            'rmse': metrics.rmse,
            'r_squared': metrics.r_squared,
            'mape': metrics.mape,
            'f1_score': metrics.f1_score,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'auc': metrics.auc,
        }
        self.log_metrics(metrics_dict)
        logger.info(f"Logged model metrics: MAE={metrics.mae:.2f}, R²={metrics.r_squared:.3f}")

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact (file or directory)."""
        mlflow.log_artifact(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")

    def log_artifacts_dict(self, artifacts: Dict[str, Any]):
        """Log dictionary of artifacts as JSON."""
        artifact_json = json.dumps(artifacts, indent=2, default=str)
        with open('/tmp/artifacts.json', 'w') as f:
            f.write(artifact_json)
        mlflow.log_artifact('/tmp/artifacts.json')
        logger.info(f"Logged {len(artifacts)} artifacts")

    def log_model_sklearn(self, model, model_name: str):
        """Log scikit-learn model."""
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged sklearn model: {model_name}")

    def register_model(self, run_id: str, model_uri: str, model_name: str, 
                      version: int, description: str = ""):
        """Register model in MLflow registry."""
        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Registered model: {model_name} (version {result.version})")
            return result
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """Transition model to new stage (Staging, Production, Archived)."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get information about a run."""
        run = mlflow.get_run(run_id)
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags,
        }

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        results = {}
        
        for exp_id in experiment_ids:
            runs = mlflow.search_runs(experiment_ids=[exp_id])
            
            if len(runs) > 0:
                best_run = runs.sort_values('metrics.f1_score', ascending=False).iloc[0]
                results[exp_id] = {
                    'best_run_id': best_run['run_id'],
                    'best_f1': best_run['metrics.f1_score'],
                    'mae': best_run.get('metrics.mae', None),
                    'rmse': best_run.get('metrics.rmse', None),
                    'num_runs': len(runs),
                }
        
        logger.info(f"Compared {len(experiment_ids)} experiments")
        return results

    def end_run(self, status: str = "FINISHED"):
        """End current run."""
        mlflow.end_run()
        logger.info(f"Ended run with status: {status}")

    def get_best_model(self, experiment_name: str, metric: str = "f1_score") -> Dict[str, Any]:
        """Get best model from experiment."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        if len(runs) == 0:
            logger.warning(f"No runs found for experiment: {experiment_name}")
            return None
        
        best_run = runs.iloc[0]
        
        return {
            'run_id': best_run['run_id'],
            'model_uri': f"runs:/{best_run['run_id']}/model",
            'metrics': {
                col.replace('metrics.', ''): best_run[col]
                for col in best_run.index if col.startswith('metrics.')
            },
            'params': {
                col.replace('params.', ''): best_run[col]
                for col in best_run.index if col.startswith('params.')
            },
        }

    def log_system_config(self, config: Dict[str, Any]):
        """Log system configuration."""
        mlflow.set_tags({
            'system_config': json.dumps(config, indent=2, default=str),
        })
        logger.info("Logged system configuration")

    def log_comparison_results(self, comparison_name: str, results: Dict[str, Any]):
        """Log comparison results."""
        mlflow.log_dict(results, f"{comparison_name}.json")
        logger.info(f"Logged comparison results: {comparison_name}")

    def create_run_link(self, run_id: str) -> str:
        """Create link to run in MLflow UI."""
        return f"{self.tracking_uri}/#/experiments/0/runs/{run_id}"

    def get_run_artifacts(self, run_id: str) -> List[str]:
        """Get artifacts from a run."""
        artifacts = self.client.list_artifacts(run_id)
        return [a.path for a in artifacts]

    def download_run_artifact(self, run_id: str, artifact_path: str, 
                             local_path: str):
        """Download artifact from run."""
        self.client.download_artifacts(run_id, artifact_path, local_path)
        logger.info(f"Downloaded artifact: {artifact_path} to {local_path}")
