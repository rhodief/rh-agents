"""
RH-Agents - Doctrine-Driven AI Actors Orchestration Framework

A Python framework for building AI agent orchestration systems with:
- Actor-based architecture (Agents, Tools, LLMs)
- State recovery and replay
- Parallel execution
- Event streaming
"""

# Core actors
from rh_agents.core.actors import Agent, Tool, LLM, ToolSet, BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.result_types import LLM_Result, Tool_Result

# State management
from rh_agents.state_backends import (
    FileSystemStateBackend, 
    FileSystemArtifactBackend
)
from rh_agents.core.state_backend import StateBackend, ArtifactBackend
from rh_agents.core.state_recovery import (
    StateSnapshot,
    StateStatus,
    ReplayMode,
    StateMetadata
)

# Event system
from rh_agents.bus_handlers import EventPrinter, EventStreamer
from rh_agents.core.types import EventType, ExecutionStatus

# Data models
from rh_agents.models import Message, AuthorType, ArtifactRef

# Parallel execution
from rh_agents.core.parallel import ErrorStrategy, ParallelEventGroup

__version__ = "1.1.0"

__all__ = [
    # Core actors
    "Agent",
    "Tool",
    "LLM",
    "ToolSet",
    "BaseActor",
    # Execution
    "ExecutionState",
    "ExecutionEvent",
    "ExecutionResult",
    # Results
    "LLM_Result",
    "Tool_Result",
    # State backends
    "StateBackend",
    "ArtifactBackend",
    "FileSystemStateBackend",
    "FileSystemArtifactBackend",
    # State recovery
    "StateSnapshot",
    "StateStatus",
    "ReplayMode",
    "StateMetadata",
    # Event system
    "EventPrinter",
    "EventStreamer",
    "EventType",
    "ExecutionStatus",
    # Data models
    "Message",
    "AuthorType",
    "ArtifactRef",
    # Parallel
    "ErrorStrategy",
    "ParallelEventGroup",
]
