"""
Phase 1 Unit Tests: Public API Imports

Validates that all public API components can be imported from top-level rh_agents package.
"""
import pytest


def test_core_actors_import():
    """Test importing core actor classes from top-level."""
    from rh_agents import Agent, Tool, LLM, ToolSet, BaseActor
    
    assert Agent is not None
    assert Tool is not None
    assert LLM is not None
    assert ToolSet is not None
    assert BaseActor is not None


def test_execution_import():
    """Test importing execution components from top-level."""
    from rh_agents import ExecutionState, ExecutionEvent, ExecutionResult
    
    assert ExecutionState is not None
    assert ExecutionEvent is not None
    assert ExecutionResult is not None


def test_result_types_import():
    """Test importing result types from top-level."""
    from rh_agents import LLM_Result, Tool_Result
    
    assert LLM_Result is not None
    assert Tool_Result is not None


def test_state_backends_import():
    """Test importing state backend classes from top-level."""
    from rh_agents import (
        StateBackend,
        ArtifactBackend,
        FileSystemStateBackend,
        FileSystemArtifactBackend
    )
    
    assert StateBackend is not None
    assert ArtifactBackend is not None
    assert FileSystemStateBackend is not None
    assert FileSystemArtifactBackend is not None


def test_state_recovery_import():
    """Test importing state recovery components from top-level."""
    from rh_agents import StateSnapshot, StateStatus, ReplayMode, StateMetadata
    
    assert StateSnapshot is not None
    assert StateStatus is not None
    assert ReplayMode is not None
    assert StateMetadata is not None


def test_event_system_import():
    """Test importing event system components from top-level."""
    from rh_agents import EventPrinter, EventStreamer, EventType, ExecutionStatus
    
    assert EventPrinter is not None
    assert EventStreamer is not None
    assert EventType is not None
    assert ExecutionStatus is not None


def test_data_models_import():
    """Test importing data models from top-level."""
    from rh_agents import Message, AuthorType, ArtifactRef
    
    assert Message is not None
    assert AuthorType is not None
    assert ArtifactRef is not None


def test_parallel_import():
    """Test importing parallel execution components from top-level."""
    from rh_agents import ErrorStrategy, ParallelEventGroup
    
    assert ErrorStrategy is not None
    assert ParallelEventGroup is not None


def test_version_available():
    """Test that version is available."""
    from rh_agents import __version__
    
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ == "2.0.0"


def test_all_exports():
    """Test that __all__ contains all expected exports."""
    from rh_agents import __all__
    
    expected_exports = [
        "Agent", "Tool", "LLM", "ToolSet", "BaseActor",
        "ExecutionState", "ExecutionEvent", "ExecutionResult",
        "LLM_Result", "Tool_Result",
        "StateBackend", "ArtifactBackend",
        "FileSystemStateBackend", "FileSystemArtifactBackend",
        "StateSnapshot", "StateStatus", "ReplayMode", "StateMetadata",
        "EventPrinter", "EventStreamer", "EventType", "ExecutionStatus",
        "Message", "AuthorType", "ArtifactRef",
        "ErrorStrategy", "ParallelEventGroup",
    ]
    
    for export in expected_exports:
        assert export in __all__, f"{export} not in __all__"


def test_backward_compat_imports():
    """Test that old import paths still work (backward compatibility)."""
    # Old style imports should still work
    from rh_agents.core.actors import Agent, Tool, LLM
    from rh_agents.core.execution import ExecutionState
    from rh_agents.bus_handlers import EventPrinter
    
    assert Agent is not None
    assert Tool is not None
    assert LLM is not None
    assert ExecutionState is not None
    assert EventPrinter is not None
