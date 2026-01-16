"""
State Recovery System - Core Models

This module contains the core data models for the state recovery system,
which enables stateless pipeline execution with checkpoint/restore capabilities.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class StateStatus(str, Enum):
    """Execution state status for tracking pipeline progress"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplayMode(str, Enum):
    """
    Defines how to handle replay of already-executed events.
    
    - NORMAL: Standard replay - skip completed events, execute only new ones
    - VALIDATION: Re-execute all events and compare results (for testing determinism)
    - REPUBLISH_ALL: Republish all events to bus (for debugging/event stream reconstruction)
    """
    NORMAL = "normal"
    VALIDATION = "validation"
    REPUBLISH_ALL = "republish_all"


class StateMetadata(BaseModel):
    """
    Rich metadata for state organization, querying, and lineage tracking.
    
    Enables:
    - Tagging states for easy retrieval
    - Tracking pipeline name and description
    - State lineage (parent-child relationships)
    - Custom user-defined metadata
    """
    tags: list[str] = Field(default_factory=list, description="Tags for categorization (e.g., ['production', 'user_123'])")
    description: str = Field(default="", description="Human-readable description of this state")
    pipeline_name: str = Field(default="", description="Name of the pipeline that created this state")
    parent_state_id: Optional[str] = Field(default=None, description="ID of parent state (for lineage tracking)")
    custom: dict[str, Any] = Field(default_factory=dict, description="User-defined custom metadata")


class StateDiff(BaseModel):
    """
    Represents differences between two state snapshots.
    
    Used for:
    - Debugging (what changed between checkpoints?)
    - Audit trail (track execution evolution)
    - Testing (assert expected state changes)
    """
    new_events: list[dict] = Field(default_factory=list, description="Events present in snapshot2 but not in snapshot1")
    removed_events: list[dict] = Field(default_factory=list, description="Events present in snapshot1 but not in snapshot2")
    changed_storage: dict[str, tuple[Any, Any]] = Field(default_factory=dict, description="Storage keys with changed values (old, new)")
    new_artifacts: list[str] = Field(default_factory=list, description="Artifact keys added in snapshot2")
    removed_artifacts: list[str] = Field(default_factory=list, description="Artifact keys removed in snapshot2")


class StateSnapshot(BaseModel):
    """
    Serialized execution state snapshot for persistence and recovery.
    
    Contains:
    - State identity (UUID)
    - Timestamps and status
    - Complete execution state (history, storage, stack)
    - Artifact references (actual artifacts stored separately)
    - Rich metadata for querying
    - Schema version for migrations
    """
    state_id: str = Field(..., description="Unique identifier for this execution state (UUID)")
    created_at: str = Field(..., description="ISO timestamp when state was first created")
    updated_at: str = Field(..., description="ISO timestamp of last update")
    version: str = Field(default="1.0.0", description="Schema version for migration compatibility")
    status: StateStatus = Field(..., description="Current status of the execution")
    metadata: StateMetadata = Field(default_factory=StateMetadata, description="Rich metadata for organization")
    
    # Core state data
    execution_state: dict[str, Any] = Field(..., description="Serialized ExecutionState (excludes runtime components)")
    artifact_refs: dict[str, str] = Field(default_factory=dict, description="Mapping of artifact keys to artifact IDs")
    
    def is_compatible(self, current_version: str) -> bool:
        """
        Check if snapshot version is compatible with current schema version.
        
        Uses semantic versioning rules:
        - Major version change: Breaking, incompatible
        - Minor/patch version change: Compatible
        
        Args:
            current_version: Current schema version (e.g., "1.0.0")
            
        Returns:
            True if compatible, False if migration needed
        """
        def parse_version(v: str) -> tuple[int, int, int]:
            parts = v.split('.')
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0, int(parts[2]) if len(parts) > 2 else 0)
        
        snapshot_major, _, _ = parse_version(self.version)
        current_major, _, _ = parse_version(current_version)
        
        # Same major version = compatible
        return snapshot_major == current_major
    
    @staticmethod
    def diff(snapshot1: "StateSnapshot", snapshot2: "StateSnapshot") -> StateDiff:
        """
        Compare two snapshots and return their differences.
        
        Args:
            snapshot1: First snapshot (typically earlier/baseline)
            snapshot2: Second snapshot (typically later/changed)
            
        Returns:
            StateDiff object containing all differences
        """
        diff = StateDiff()
        
        # Compare events in history
        state1 = snapshot1.execution_state
        state2 = snapshot2.execution_state
        
        # Get event lists (handling different possible structures)
        events1_list = state1.get('history', {}).get('events', [])
        events2_list = state2.get('history', {}).get('events', [])
        
        # Convert to sets of addresses for comparison
        events1_addrs = {e.get('address', '') for e in events1_list if isinstance(e, dict)}
        events2_addrs = {e.get('address', '') for e in events2_list if isinstance(e, dict)}
        
        # Find new and removed events
        new_addrs = events2_addrs - events1_addrs
        removed_addrs = events1_addrs - events2_addrs
        
        diff.new_events = [e for e in events2_list if isinstance(e, dict) and e.get('address') in new_addrs]
        diff.removed_events = [e for e in events1_list if isinstance(e, dict) and e.get('address') in removed_addrs]
        
        # Compare storage data
        storage1 = state1.get('storage', {}).get('data', {})
        storage2 = state2.get('storage', {}).get('data', {})
        
        for key in set(storage1.keys()) | set(storage2.keys()):
            val1 = storage1.get(key)
            val2 = storage2.get(key)
            if val1 != val2:
                diff.changed_storage[key] = (val1, val2)
        
        # Compare artifacts
        artifacts1 = set(snapshot1.artifact_refs.keys())
        artifacts2 = set(snapshot2.artifact_refs.keys())
        
        diff.new_artifacts = list(artifacts2 - artifacts1)
        diff.removed_artifacts = list(artifacts1 - artifacts2)
        
        return diff
    
    def __str__(self) -> str:
        """Human-readable representation of snapshot"""
        return (f"StateSnapshot(id={self.state_id[:8]}..., "
                f"status={self.status.value}, "
                f"events={len(self.execution_state.get('history', {}).get('events', []))}, "
                f"artifacts={len(self.artifact_refs)})")


# Schema version for this module
CURRENT_SCHEMA_VERSION = "1.0.0"
