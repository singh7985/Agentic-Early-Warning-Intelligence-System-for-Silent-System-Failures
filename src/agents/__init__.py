"""
Agentic AI Architecture for Early Warning System

This module provides autonomous agents for:
- Monitoring: Runs ML inference and detects anomalies/drift
- Retrieval: Queries historical data from VectorDB
- Reasoning: Explains risk using retrieved evidence
- Action: Suggests interventions and escalates

Agents are orchestrated using LangGraph for collaborative decision-making.
"""

__version__ = "1.0.0"

from .monitoring_agent import MonitoringAgent
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .action_agent import ActionAgent
from .orchestrator import AgentOrchestrator, AgentMessage, AgentResult

__all__ = [
    'MonitoringAgent',
    'RetrievalAgent',
    'ReasoningAgent',
    'ActionAgent',
    'AgentOrchestrator',
    'AgentMessage',
    'AgentResult',
]
