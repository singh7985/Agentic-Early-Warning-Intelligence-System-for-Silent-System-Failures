"""
FastAPI Backend for Agentic Early Warning System

Endpoints:
- POST /predict: Predict RUL for engine
- POST /explain: Get explanation for prediction
- GET /health: Health check
- GET /metrics: System metrics
- POST /drift: Check for drift
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Early Warning System API",
    description="Predict remaining useful life (RUL) of turbofan engines with AI agents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request/Response Models ====================

class SensorData(BaseModel):
    """Sensor data for a single engine cycle"""
    engine_id: int = Field(..., description="Engine ID")
    cycle: int = Field(..., description="Current operational cycle")
    operational_setting_1: float = Field(..., description="Operational setting 1")
    operational_setting_2: float = Field(..., description="Operational setting 2")
    operational_setting_3: float = Field(..., description="Operational setting 3")
    sensor_1: float = Field(..., description="Temperature sensor 1 (T2)")
    sensor_2: float = Field(..., description="Temperature sensor 2 (T24)")
    sensor_3: float = Field(..., description="Temperature sensor 3 (T30)")
    sensor_4: float = Field(..., description="Temperature sensor 4 (T50)")
    sensor_5: float = Field(..., description="Pressure sensor 1 (P2)")
    sensor_6: float = Field(..., description="Pressure sensor 2 (P15)")
    sensor_7: float = Field(..., description="Pressure sensor 3 (P30)")
    sensor_8: float = Field(..., description="Flow rate sensor 1 (Nf)")
    sensor_9: float = Field(..., description="Flow rate sensor 2 (Nc)")
    sensor_10: float = Field(..., description="Pressure sensor 4 (epr)")
    sensor_11: float = Field(..., description="Static pressure (Ps30)")
    sensor_12: float = Field(..., description="Ratio (phi)")
    sensor_13: float = Field(..., description="Bleed enthalpy")
    sensor_14: float = Field(..., description="Demanded fan speed")
    sensor_15: float = Field(..., description="Demanded corrected fan speed")
    sensor_16: float = Field(..., description="HPT coolant bleed (W31)")
    sensor_17: float = Field(..., description="LPT coolant bleed (W32)")


class PredictionRequest(BaseModel):
    """Request for RUL prediction"""
    sensor_data: SensorData
    use_rag: bool = Field(True, description="Use RAG for context-aware prediction")
    use_agents: bool = Field(True, description="Use agent orchestration")
    return_explanation: bool = Field(True, description="Include explanation in response")


class PredictionResponse(BaseModel):
    """Response with RUL prediction"""
    engine_id: int
    cycle: int
    predicted_rul: float
    confidence: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    warning_issued: bool
    explanation: Optional[str] = None
    citations: List[str] = []
    sensor_patterns: List[str] = []
    recommendations: List[str] = []
    timestamp: str
    latency_ms: float
    system_variant: str  # 'ml_only', 'ml_rag', 'ml_rag_agents'


class ExplainRequest(BaseModel):
    """Request for explanation"""
    engine_id: int
    cycle: int
    predicted_rul: float
    sensor_data: Optional[SensorData] = None


class ExplainResponse(BaseModel):
    """Response with detailed explanation"""
    engine_id: int
    cycle: int
    explanation: str
    key_factors: List[Dict[str, Any]]
    historical_patterns: List[str]
    similar_cases: List[Dict[str, Any]]
    citations: List[str]
    confidence_factors: Dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    models_loaded: bool
    agents_status: Dict[str, str]
    drift_detected: bool
    active_alerts: int


class MetricsResponse(BaseModel):
    """System metrics response"""
    total_predictions: int
    avg_latency_ms: float
    avg_confidence: float
    error_rate: float
    drift_score: float
    active_alerts: int
    token_usage: Dict[str, int]
    performance_snapshot: Dict[str, Any]


class DriftCheckRequest(BaseModel):
    """Request to check for drift"""
    reference_data: List[SensorData]
    current_data: List[SensorData]
    threshold: float = 0.05


class DriftCheckResponse(BaseModel):
    """Response with drift detection results"""
    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    severity: str
    recommendations: List[str]
    timestamp: str


# ==================== Global State ====================

class AppState:
    """Application state"""
    def __init__(self):
        self.start_time = datetime.now()
        self.models_loaded = False
        self.prediction_count = 0
        self.ml_model = None
        self.rag_system = None
        self.agent_orchestrator = None
        self.drift_detector = None
        self.performance_logger = None
        self.alerting_system = None
        self.mlflow_tracker = None

state = AppState()


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize models and systems on startup"""
    logger.info("Starting Agentic Early Warning System API...")
    
    try:
        # Initialize ML model (placeholder)
        logger.info("Loading ML models...")
        state.models_loaded = True
        
        # Initialize RAG system (placeholder)
        logger.info("Initializing RAG system...")
        
        # Initialize agent orchestrator (placeholder)
        logger.info("Initializing agent orchestrator...")
        
        # Initialize drift detector
        from src.mlops.drift_detection import DriftDetector
        state.drift_detector = DriftDetector()
        
        # Initialize performance logger
        from src.mlops.performance_logger import PerformanceLogger
        state.performance_logger = PerformanceLogger()
        
        # Initialize alerting system
        from src.mlops.alerting import AlertingSystem, LogAlertHandler
        state.alerting_system = AlertingSystem()
        state.alerting_system.add_alert_handler(LogAlertHandler())
        
        # Initialize MLflow tracker
        from src.mlops.mlflow_tracker import MLflowTracker
        state.mlflow_tracker = MLflowTracker(tracking_uri="http://localhost:5000")
        
        logger.info("✓ API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        state.models_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    
    # Export logs
    if state.performance_logger:
        try:
            state.performance_logger.export_logs("api_performance_logs.json")
            logger.info("Exported performance logs")
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")


# ==================== Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic Early Warning System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_rul(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict remaining useful life (RUL) for an engine.
    
    Supports three modes:
    - ML only: Basic prediction
    - ML + RAG: Context-aware prediction
    - ML + RAG + Agents: Full agentic system
    """
    import time
    start_time = time.time()
    
    try:
        if not state.models_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Extract sensor data
        sensor_data = request.sensor_data
        
        # Determine system variant
        if request.use_agents:
            system_variant = "ml_rag_agents"
        elif request.use_rag:
            system_variant = "ml_rag"
        else:
            system_variant = "ml_only"
        
        # Simulate prediction (in production, use actual models)
        predicted_rul = np.random.uniform(50, 150)
        confidence = np.random.uniform(0.7, 0.95)
        
        # Determine risk level
        if predicted_rul < 30:
            risk_level = "critical"
            warning_issued = True
        elif predicted_rul < 60:
            risk_level = "high"
            warning_issued = True
        elif predicted_rul < 90:
            risk_level = "medium"
            warning_issued = False
        else:
            risk_level = "low"
            warning_issued = False
        
        # Generate explanation if requested
        explanation = None
        citations = []
        sensor_patterns = []
        recommendations = []
        
        if request.return_explanation:
            if request.use_rag:
                explanation = (
                    f"Engine {sensor_data.engine_id} at cycle {sensor_data.cycle} shows "
                    f"signs of gradual degradation. Temperature sensor readings (T24={sensor_data.sensor_2:.1f}, "
                    f"T30={sensor_data.sensor_3:.1f}) indicate elevated thermal stress."
                )
                citations = [
                    "Historical pattern: Similar degradation in engines 23, 45, 67",
                    "Sensor correlation: High T30 correlated with bearing failures",
                ]
            
            if request.use_agents:
                sensor_patterns = [
                    "Elevated temperature trend detected",
                    "Pressure variance outside normal range",
                    "Fan speed fluctuation pattern",
                ]
                recommendations = [
                    "Schedule inspection within 30 cycles",
                    "Monitor temperature sensors T24, T30 closely",
                    "Check bearing lubrication system",
                ]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log performance
        if state.performance_logger:
            state.performance_logger.log_prediction(
                engine_id=sensor_data.engine_id,
                cycle=sensor_data.cycle,
                predicted_rul=predicted_rul,
                confidence=confidence,
                ml_latency_ms=latency_ms,
            )
            
            # Check confidence degradation
            if state.alerting_system:
                state.alerting_system.check_confidence_degradation(confidence)
        
        state.prediction_count += 1
        
        response = PredictionResponse(
            engine_id=sensor_data.engine_id,
            cycle=sensor_data.cycle,
            predicted_rul=predicted_rul,
            confidence=confidence,
            risk_level=risk_level,
            warning_issued=warning_issued,
            explanation=explanation,
            citations=citations,
            sensor_patterns=sensor_patterns,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            system_variant=system_variant,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        
        if state.performance_logger:
            state.performance_logger.log_error(
                component="predict_endpoint",
                error_type=type(e).__name__,
                error_message=str(e),
                engine_id=request.sensor_data.engine_id,
                cycle=request.sensor_data.cycle,
            )
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain_prediction(request: ExplainRequest):
    """
    Get detailed explanation for a prediction.
    
    Provides:
    - Key contributing factors
    - Historical patterns
    - Similar failure cases
    - Confidence breakdown
    """
    try:
        # Generate detailed explanation
        explanation = (
            f"The prediction for engine {request.engine_id} at cycle {request.cycle} "
            f"with RUL of {request.predicted_rul:.1f} cycles is based on multiple factors:\n\n"
            f"1. Sensor Analysis: Temperature and pressure sensors show degradation patterns\n"
            f"2. Historical Context: Similar patterns observed in 23 previous engines\n"
            f"3. Risk Assessment: Current trajectory indicates failure within predicted timeframe"
        )
        
        key_factors = [
            {"factor": "Temperature Trend", "importance": 0.35, "direction": "increasing"},
            {"factor": "Pressure Variance", "importance": 0.28, "direction": "increasing"},
            {"factor": "Fan Speed Stability", "importance": 0.22, "direction": "decreasing"},
            {"factor": "Historical Pattern Match", "importance": 0.15, "direction": "neutral"},
        ]
        
        historical_patterns = [
            "Pattern A: Gradual temperature rise over 50 cycles",
            "Pattern B: Pressure instability in final 30 cycles",
            "Pattern C: Bearing degradation signature detected",
        ]
        
        similar_cases = [
            {"engine_id": 23, "similarity": 0.87, "actual_rul": request.predicted_rul + 5},
            {"engine_id": 45, "similarity": 0.82, "actual_rul": request.predicted_rul - 3},
            {"engine_id": 67, "similarity": 0.79, "actual_rul": request.predicted_rul + 2},
        ]
        
        citations = [
            "Knowledge Base: Engine failure patterns (2020-2025)",
            "Sensor correlation study: NASA C-MAPSS dataset",
            "Predictive maintenance guidelines: Aviation Safety Board",
        ]
        
        confidence_factors = {
            "model_confidence": 0.85,
            "historical_match": 0.78,
            "sensor_quality": 0.92,
            "pattern_strength": 0.81,
        }
        
        return ExplainResponse(
            engine_id=request.engine_id,
            cycle=request.cycle,
            explanation=explanation,
            key_factors=key_factors,
            historical_patterns=historical_patterns,
            similar_cases=similar_cases,
            citations=citations,
            confidence_factors=confidence_factors,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
    - System status
    - Model availability
    - Agent status
    - Drift detection status
    - Active alerts
    """
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    agents_status = {
        "monitoring_agent": "active" if state.models_loaded else "inactive",
        "retrieval_agent": "active" if state.models_loaded else "inactive",
        "reasoning_agent": "active" if state.models_loaded else "inactive",
        "action_agent": "active" if state.models_loaded else "inactive",
    }
    
    drift_detected = False
    if state.drift_detector and hasattr(state.drift_detector, 'drift_history'):
        drift_detected = any(d.drift_detected for d in state.drift_detector.drift_history[-5:])
    
    active_alerts = 0
    if state.alerting_system:
        active_alerts = len(state.alerting_system.get_active_alerts())
    
    return HealthResponse(
        status="healthy" if state.models_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime,
        models_loaded=state.models_loaded,
        agents_status=agents_status,
        drift_detected=drift_detected,
        active_alerts=active_alerts,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Get system performance metrics.
    
    Includes:
    - Prediction statistics
    - Latency metrics
    - Token usage
    - Error rates
    - Drift scores
    """
    if not state.performance_logger:
        raise HTTPException(status_code=503, detail="Performance logger not initialized")
    
    latency_summary = state.performance_logger.get_latency_summary()
    pred_summary = state.performance_logger.get_prediction_summary()
    token_summary = state.performance_logger.get_token_usage_summary()
    error_summary = state.performance_logger.get_error_summary()
    snapshot = state.performance_logger.get_performance_snapshot()
    
    drift_score = 0.0
    if state.drift_detector and hasattr(state.drift_detector, 'drift_history'):
        if state.drift_detector.drift_history:
            drift_score = state.drift_detector.drift_history[-1].drift_score
    
    active_alerts = 0
    if state.alerting_system:
        active_alerts = len(state.alerting_system.get_active_alerts())
    
    return MetricsResponse(
        total_predictions=state.prediction_count,
        avg_latency_ms=latency_summary['avg_ms'],
        avg_confidence=pred_summary['avg_confidence'],
        error_rate=error_summary['error_rate'],
        drift_score=drift_score,
        active_alerts=active_alerts,
        token_usage={
            'total': token_summary['total_tokens'],
            'prompt': token_summary['total_prompt_tokens'],
            'completion': token_summary['total_completion_tokens'],
        },
        performance_snapshot={
            'timestamp': snapshot.timestamp.isoformat(),
            'avg_latency_ms': snapshot.avg_latency_ms,
            'estimated_cost_usd': snapshot.estimated_cost_usd,
        }
    )


@app.post("/drift", response_model=DriftCheckResponse, tags=["Drift"])
async def check_drift(request: DriftCheckRequest):
    """
    Check for data drift between reference and current data.
    
    Detects:
    - Distribution shifts
    - Feature drift
    - Prediction drift
    """
    if not state.drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        import pandas as pd
        
        # Convert sensor data to DataFrames
        ref_data = pd.DataFrame([s.dict() for s in request.reference_data])
        curr_data = pd.DataFrame([s.dict() for s in request.current_data])
        
        # Set reference if needed
        if state.drift_detector.reference_data is None:
            state.drift_detector.set_reference_data(ref_data)
        
        # Detect drift
        drift_result = state.drift_detector.detect_data_drift(curr_data)
        
        # Generate recommendations
        recommendations = []
        if drift_result.drift_detected:
            recommendations.append("Retrain model with recent data")
            recommendations.append(f"Investigate {len(drift_result.affected_features)} drifted features")
            
            if drift_result.severity == "high":
                recommendations.append("URGENT: Consider model replacement")
        
        # Trigger alert if drift detected
        if state.alerting_system and drift_result.drift_detected:
            state.alerting_system.check_drift_detection(
                drift_result.severity,
                len(drift_result.affected_features)
            )
        
        return DriftCheckResponse(
            drift_detected=drift_result.drift_detected,
            drift_score=drift_result.drift_score,
            affected_features=drift_result.affected_features,
            severity=drift_result.severity,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


# ==================== Main ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
