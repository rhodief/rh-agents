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
from rh_agents.core.types import EventType, ExecutionStatus, LogEvent, LogSeverity

# Data models
from rh_agents.models import Message, AuthorType, ArtifactRef

# Parallel execution
from rh_agents.core.parallel import ErrorStrategy as ParallelErrorStrategy, ParallelEventGroup

# Builder types
from rh_agents.core.types import ErrorStrategy as BuilderErrorStrategy, AggregationStrategy
from rh_agents.core.result_types import ToolExecutionResult

# Additional helpers (Phase 2)
from rh_agents.decorators import tool as tool_decorator, agent as agent_decorator
from rh_agents.validation import (
    validate_actor,
    validate_state,
    validate_handler_signature,
    ActorValidationError,
    StateValidationError
)
from rh_agents.builders import (
    BuilderAgent,
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)

__version__ = "2.0.0"

__all__ = [
    # Core actors
    "Agent",
    "Tool",
    "LLM",
    "ToolSet",
    "BaseActor",
    # Builders
    "BuilderAgent",
    "StructuredAgent",
    "CompletionAgent",
    "ToolExecutorAgent",
    "DirectToolAgent",
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
    "ParallelErrorStrategy",
    "ParallelEventGroup",
    # Builder types
    "BuilderErrorStrategy",
    "AggregationStrategy",
    "ToolExecutionResult",
    # Decorators
    "tool_decorator",
    "agent_decorator",
    # Validation
    "validate_actor",
    "validate_state",
    "validate_handler_signature",
    "ActorValidationError",
    "StateValidationError",
]
