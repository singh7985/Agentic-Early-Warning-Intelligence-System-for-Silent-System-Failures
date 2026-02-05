import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    
    # API
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: str = None
    
    # Data paths
    data_raw_path: str = "./data/raw"
    data_processed_path: str = "./data/processed"
    cmapss_dataset_url: str = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
    
    # Model paths
    model_checkpoint_path: str = "./models/checkpoints"
    vector_db_path: str = "./data/vector_db"
    
    # Database
    pgvector_connection_string: str = "postgresql://localhost:5432/ewis"
    faiss_index_path: str = "./data/faiss_index"
    
    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "agentic-ewis"
    
    # System
    random_seed: int = 42
    num_workers: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def ensure_paths(self):
        """Create necessary directories if they don't exist."""
        for path_attr in [
            "data_raw_path",
            "data_processed_path",
            "model_checkpoint_path",
            "vector_db_path"
        ]:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_paths()
