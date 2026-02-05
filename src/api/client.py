"""
API Client for Early Warning System

Usage example:
    client = EarlyWarningClient("http://localhost:8000")
    
    sensor_data = {
        "engine_id": 1,
        "cycle": 150,
        "operational_setting_1": 0.0023,
        ...
    }
    
    result = client.predict(sensor_data, use_rag=True, use_agents=True)
    print(f"RUL: {result['predicted_rul']}")
"""

import requests
from typing import Dict, List, Optional, Any


class EarlyWarningClient:
    """Client for interacting with Early Warning System API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def predict(
        self,
        sensor_data: Dict[str, Any],
        use_rag: bool = True,
        use_agents: bool = True,
        return_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict RUL for an engine.
        
        Args:
            sensor_data: Dictionary containing sensor readings
            use_rag: Use RAG for context-aware prediction
            use_agents: Use agent orchestration
            return_explanation: Include explanation in response
            
        Returns:
            Prediction result with RUL, confidence, and explanation
        """
        payload = {
            "sensor_data": sensor_data,
            "use_rag": use_rag,
            "use_agents": use_agents,
            "return_explanation": return_explanation,
        }

        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def explain(
        self,
        engine_id: int,
        cycle: int,
        predicted_rul: float,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction.
        
        Args:
            engine_id: Engine ID
            cycle: Cycle number
            predicted_rul: Predicted RUL value
            sensor_data: Optional sensor data
            
        Returns:
            Detailed explanation with key factors and similar cases
        """
        payload = {
            "engine_id": engine_id,
            "cycle": cycle,
            "predicted_rul": predicted_rul,
            "sensor_data": sensor_data,
        }

        response = self.session.post(
            f"{self.base_url}/explain",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status including models, agents, and drift detection
        """
        response = self.session.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics.
        
        Returns:
            Metrics including latency, confidence, and token usage
        """
        response = self.session.get(
            f"{self.base_url}/metrics",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def check_drift(
        self,
        reference_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]],
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Check for data drift.
        
        Args:
            reference_data: Historical reference data
            current_data: Current data to compare
            threshold: Drift detection threshold
            
        Returns:
            Drift detection results
        """
        payload = {
            "reference_data": reference_data,
            "current_data": current_data,
            "threshold": threshold,
        }

        response = self.session.post(
            f"{self.base_url}/drift",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the session."""
        self.session.close()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = EarlyWarningClient("http://localhost:8000")

    # Check health
    print("Checking API health...")
    health = client.health()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")

    # Make prediction
    print("\nMaking prediction...")
    sensor_data = {
        "engine_id": 1,
        "cycle": 150,
        "operational_setting_1": 0.0023,
        "operational_setting_2": 0.0003,
        "operational_setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.51,
        "sensor_3": 1589.09,
        "sensor_4": 1400.51,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 553.62,
        "sensor_8": 2388.05,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.39,
        "sensor_12": 521.48,
        "sensor_13": 2388.04,
        "sensor_14": 8133.77,
        "sensor_15": 8.4052,
        "sensor_16": 0.03,
        "sensor_17": 392,
    }

    result = client.predict(sensor_data, use_rag=True, use_agents=True)
    print(f"Predicted RUL: {result['predicted_rul']:.1f} cycles")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Risk Level: {result['risk_level']}")
    
    if result.get('explanation'):
        print(f"\nExplanation: {result['explanation']}")

    # Get metrics
    print("\nFetching system metrics...")
    metrics = client.metrics()
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")

    # Close client
    client.close()
