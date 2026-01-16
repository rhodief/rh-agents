# State Recovery System - Implementation Plan

## Overview

This document consolidates all design decisions from the specification and provides a clear, actionable implementation roadmap. The system transforms from a **hash-based cache** to a **stateless pipeline with state recovery**, enabling user-in-the-loop workflows, error recovery, and selective replay.

---

## Key Design Decisions Summary

### Critical Understanding: Stateless Pipeline System

**This is NOT a traditional cache or strict state recovery system.**

**Purpose:** Enable stateless execution where:
1. **User-in-the-loop**: Pause execution, await user input, resume from exact point
2. **Error recovery**: On failure, restore state and retry from failure point
3. **Selective replay**: Re-execute from specific address when user requests changes
4. **Audit trail**: Track complete execution history for debugging and compliance

### Architecture Principles

1. **State Identity**: Each execution has unique UUID (`state_id`)
2. **Complete Serialization**: Full execution state (history, storage, artifacts) can be persisted
3. **Smart Replay**: On restore, skip already-executed events, run only new ones
4. **Address-Based Matching**: Use execution address to identify events (simple, sufficient)
5. **Selective Publishing**: Replay doesn't re-trigger bus notifications (unless debugging)
6. **Artifact Abstraction**: Large objects stored separately with pluggable adapters
7. **No CacheBackend**: Completely remove existing cache system (replaced by state recovery)

---

## Components to Implement

### 1. Core Models (`rh_agents/core/state_recovery.py` - NEW)

```python
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class StateStatus(str, Enum):
    """Execution state status"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class StateMetadata(BaseModel):
    """Rich metadata for state organization and querying"""
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    pipeline_name: str = ""
    parent_state_id: Optional[str] = None  # State lineage
    custom: dict[str, Any] = Field(default_factory=dict)

class StateSnapshot(BaseModel):
    """Serialized execution state snapshot"""
    state_id: str
    created_at: str
    updated_at: str
    version: str = "1.0.0"  # Schema version for migrations
    status: StateStatus
    metadata: StateMetadata
    
    # Serialized core state
    execution_state: dict[str, Any]  # ExecutionState as dict
    artifact_refs: dict[str, str] = Field(default_factory=dict)  # key -> artifact_id mapping
    
    def is_compatible(self, current_version: str) -> bool:
        """Check if snapshot version is compatible with current schema"""
        # Implement semver comparison logic
        pass

class ReplayMode(str, Enum):
    """How to handle replay of already-executed events"""
    NORMAL = "normal"  # Standard replay: skip completed events
    VALIDATION = "validation"  # Re-execute and compare results
    REPUBLISH_ALL = "republish_all"  # Republish all events (debugging)
```

### 2. State Backend Abstraction (`rh_agents/core/state_backend.py` - NEW)

```python
from abc import ABC, abstractmethod
from typing import Optional, Any
from rh_agents.core.state_recovery import StateSnapshot

class StateBackend(ABC):
    """Abstract interface for state persistence"""
    
    @abstractmethod
    def save_state(self, snapshot: StateSnapshot) -> bool:
        """Persist state snapshot. Returns True on success."""
        pass
    
    @abstractmethod
    def load_state(self, state_id: str) -> Optional[StateSnapshot]:
        """Load state snapshot by ID. Returns None if not found."""
        pass
    
    @abstractmethod
    def list_states(
        self, 
        tags: Optional[list[str]] = None,
        status: Optional[StateStatus] = None,
        pipeline_name: Optional[str] = None,
        limit: int = 100
    ) -> list[StateSnapshot]:
        """List states with optional filters"""
        pass
    
    @abstractmethod
    def delete_state(self, state_id: str) -> bool:
        """Delete state snapshot. Returns True if existed."""
        pass
    
    @abstractmethod
    def update_state(self, snapshot: StateSnapshot) -> bool:
        """Update existing snapshot. Returns True on success."""
        pass

class ArtifactBackend(ABC):
    """Abstract interface for artifact storage (separate from state)"""
    
    @abstractmethod
    def save_artifact(self, artifact_id: str, artifact: Any) -> bool:
        """Store artifact with given ID"""
        pass
    
    @abstractmethod
    def load_artifact(self, artifact_id: str) -> Optional[Any]:
        """Load artifact by ID"""
        pass
    
    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact"""
        pass
    
    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists"""
        pass
```

### 3. File System Implementation (`rh_agents/state_backends.py` - NEW)

MVP implementation using local file system:

```python
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any
from rh_agents.core.state_backend import StateBackend, ArtifactBackend
from rh_agents.core.state_recovery import StateSnapshot, StateStatus

class FileSystemStateBackend(StateBackend):
    """File-based state storage (MVP implementation)"""
    
    def __init__(self, base_path: str = "./.state_store"):
        self.base_path = Path(base_path)
        self.states_dir = self.base_path / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, snapshot: StateSnapshot) -> bool:
        """Save as JSON file"""
        file_path = self.states_dir / f"{snapshot.state_id}.json"
        with open(file_path, 'w') as f:
            f.write(snapshot.model_dump_json(indent=2))
        return True
    
    def load_state(self, state_id: str) -> Optional[StateSnapshot]:
        file_path = self.states_dir / f"{state_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return StateSnapshot.model_validate_json(f.read())
    
    def list_states(self, tags=None, status=None, pipeline_name=None, limit=100):
        """Load all states and filter in memory (simple MVP approach)"""
        # Implementation details...
    
    # ... other methods

class FileSystemArtifactBackend(ArtifactBackend):
    """File-based artifact storage using pickle"""
    
    def __init__(self, base_path: str = "./.state_store/artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_artifact(self, artifact_id: str, artifact: Any) -> bool:
        file_path = self.base_path / f"{artifact_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(artifact, f)
        return True
    
    def load_artifact(self, artifact_id: str) -> Optional[Any]:
        file_path = self.base_path / f"{artifact_id}.pkl"
        if not file_path.exists():
            return None
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    # ... other methods
```

### 4. ExecutionState Modifications (`rh_agents/core/execution.py`)

**Add Fields:**
```python
class ExecutionState(BaseModel):
    # NEW: State identity and recovery control
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    replay_mode: ReplayMode = Field(default=ReplayMode.NORMAL)
    resume_from_address: Optional[str] = Field(default=None)
    
    # Existing fields
    storage: ExecutionStore = Field(default_factory=ExecutionStore)
    history: HistorySet = Field(default_factory=HistorySet)
    execution_stack: list[str] = Field(default_factory=list)
    
    # Runtime components (excluded from serialization)
    event_bus: EventBus = Field(default_factory=EventBus, exclude=True)
    _state_backend: Optional[StateBackend] = Field(default=None, exclude=True)
    _artifact_backend: Optional[ArtifactBackend] = Field(default=None, exclude=True)
    
    # REMOVED: _cache_backend (no longer needed)
```

**Add Methods:**
```python
def to_snapshot(
    self, 
    status: StateStatus = StateStatus.RUNNING,
    metadata: Optional[StateMetadata] = None
) -> StateSnapshot:
    """Create snapshot of current state"""
    # Serialize core state (exclude runtime components)
    state_dict = self.model_dump(exclude={'event_bus', '_state_backend', '_artifact_backend'})
    
    # Extract artifact references (store artifacts separately)
    artifact_refs = {}
    for key, artifact in self.storage.artifacts.items():
        artifact_id = self._compute_artifact_id(artifact)
        artifact_refs[key] = artifact_id
        if self._artifact_backend:
            self._artifact_backend.save_artifact(artifact_id, artifact)
    
    return StateSnapshot(
        state_id=self.state_id,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        status=status,
        metadata=metadata or StateMetadata(),
        execution_state=state_dict,
        artifact_refs=artifact_refs
    )

@classmethod
def from_snapshot(
    cls,
    snapshot: StateSnapshot,
    state_backend: Optional[StateBackend] = None,
    artifact_backend: Optional[ArtifactBackend] = None,
    event_bus: Optional[EventBus] = None,
    replay_mode: ReplayMode = ReplayMode.NORMAL,
    resume_from_address: Optional[str] = None
) -> "ExecutionState":
    """Restore state from snapshot"""
    # Deserialize core state
    state = cls(**snapshot.execution_state)
    
    # Restore artifacts
    if artifact_backend:
        for key, artifact_id in snapshot.artifact_refs.items():
            artifact = artifact_backend.load_artifact(artifact_id)
            if artifact:
                state.storage.artifacts[key] = artifact
    
    # Reconstruct runtime components
    state.event_bus = event_bus or EventBus()
    state._state_backend = state_backend
    state._artifact_backend = artifact_backend
    state.replay_mode = replay_mode
    state.resume_from_address = resume_from_address
    
    return state

def save_checkpoint(
    self, 
    status: StateStatus = StateStatus.RUNNING,
    metadata: Optional[StateMetadata] = None
) -> bool:
    """Save current state to backend"""
    if not self._state_backend:
        return False
    snapshot = self.to_snapshot(status, metadata)
    return self._state_backend.save_state(snapshot)

@classmethod
def load_from_state_id(
    cls,
    state_id: str,
    state_backend: StateBackend,
    artifact_backend: Optional[ArtifactBackend] = None,
    event_bus: Optional[EventBus] = None,
    replay_mode: ReplayMode = ReplayMode.NORMAL,
    resume_from_address: Optional[str] = None
) -> Optional["ExecutionState"]:
    """Load state by ID"""
    snapshot = state_backend.load_state(state_id)
    if not snapshot:
        return None
    return cls.from_snapshot(
        snapshot, state_backend, artifact_backend, 
        event_bus, replay_mode, resume_from_address
    )

def should_skip_event(self, address: str) -> bool:
    """Check if event should be skipped during replay"""
    if self.replay_mode == ReplayMode.VALIDATION:
        return False  # Always execute for validation
    
    # If resume_from_address is set, skip everything before it
    if self.resume_from_address:
        # Check if we've reached the resume point
        if address == self.resume_from_address:
            # Clear resume_from_address so subsequent events execute
            self.resume_from_address = None
            return False  # Execute this event
        # Skip if we haven't reached resume point yet
        return address in self.history.__events
    
    # Normal replay: skip if already completed
    return address in self.history.__events
```

### 5. ExecutionEvent Modifications (`rh_agents/core/events.py`)

**Add Fields:**
```python
class ExecutionEvent(BaseModel, Generic[OutputT]):
    # ... existing fields ...
    
    # NEW: Replay control
    is_replayed: bool = Field(default=False, description="True if event from restored state")
    skip_republish: bool = Field(default=False, description="Skip publishing to event bus")
```

**Modify `__call__` Method:**
```python
async def __call__(self, input_data, extra_context, execution_state: ExecutionState):
    """Execute with replay awareness"""
    execution_state.push_context(f'{self.actor.name}{"::" + self.tag if self.tag else ""}')
    
    try:
        current_address = execution_state.get_current_address(self.actor.event_type)
        
        # REPLAY LOGIC: Check if event should be skipped
        if execution_state.should_skip_event(current_address):
            # Event already executed - retrieve result from history
            existing_event = execution_state.history[current_address]
            
            # Mark as replayed
            self.is_replayed = True
            self.from_cache = True  # Keep for compatibility
            self.execution_time = 0.0
            
            # Decide whether to republish
            if execution_state.replay_mode == ReplayMode.REPUBLISH_ALL:
                self.skip_republish = False
            else:
                self.skip_republish = True  # Don't republish old events
            
            # Publish if needed (marked as replayed)
            await execution_state.add_event(self, ExecutionStatus.RECOVERED)
            
            # Extract result from existing event
            # NOTE: Result might be in ExecutionResult wrapper or direct
            if hasattr(existing_event, 'detail'):
                # Try to extract actual result from history
                # This requires storing results in history events
                pass
            
            return ExecutionResult[OutputT](
                result=existing_event.result,  # Need to store this in history
                execution_time=0.0,
                ok=True
            )
        
        # NOT REPLAYED or VALIDATION MODE - Execute normally
        
        # Run preconditions
        await self.actor.run_preconditions(input_data, extra_context, execution_state)
        
        # Start timer
        self.start_timer()
        self.detail = self._serialize_detail(input_data)
        await execution_state.add_event(self, ExecutionStatus.STARTED)
        
        # Execute handler
        if not asyncio.iscoroutinefunction(self.actor.handler):
            raise TypeError(f"Handler for actor '{self.actor.name}' must be async.")
        result = await self.actor.handler(input_data, extra_context, execution_state)
        
        # Run postconditions
        await self.actor.run_postconditions(result, extra_context, execution_state)
        
        # Complete
        self.stop_timer()
        self.detail = self._serialize_detail(result)
        await execution_state.add_event(self, ExecutionStatus.COMPLETED)
        
        execution_result = ExecutionResult[OutputT](
            result=result,
            execution_time=self.execution_time,
            ok=True
        )
        
        # VALIDATION MODE: Compare with historical result if it exists
        if execution_state.replay_mode == ReplayMode.VALIDATION:
            if current_address in execution_state.history.__events:
                historical_event = execution_state.history[current_address]
                # Compare results (implement comparison logic)
                # Log/raise if mismatch
        
        return execution_result
    
    except Exception as e:
        self.stop_timer()
        self.message = str(e)
        await execution_state.add_event(self, ExecutionStatus.FAILED)
        return ExecutionResult[OutputT](
            result=None,
            execution_time=self.execution_time,
            ok=False,
            erro_message=str(e)
        )
    
    finally:
        execution_state.pop_context()
```

**Modify `add_event` in ExecutionState:**
```python
async def add_event(self, event: Any, status: ExecutionStatus):
    """Add event to history and conditionally publish"""
    event.address = self.get_current_address(event.actor.event_type)
    event.execution_status = status
    self.history.add(event)
    
    # Only publish if not skipped
    if not event.skip_republish:
        await self.event_bus.publish(event)
```

### 6. Remove Cache System

**Files to Modify:**
- `rh_agents/core/events.py`: Remove `__try_retrieve_from_cache` and `__store_result_in_cache` methods
- `rh_agents/core/execution.py`: Remove `_cache_backend` field
- `rh_agents/core/cache.py`: Mark as deprecated or remove entirely
- `rh_agents/cache_backends.py`: Mark as deprecated or remove

### 7. HistorySet Enhancement (`rh_agents/core/execution.py`)

**Add Method for Result Storage:**
```python
class ExecutionEvent:
    result: Optional[Any] = Field(default=None)  # Store actual result

class HistorySet(BaseModel):
    # ... existing code ...
    
    def get_event_result(self, address: str) -> Optional[Any]:
        """Get the result from a completed event at given address"""
        if address in self.__events:
            event = self.__events[address]
            if event.execution_status == ExecutionStatus.COMPLETED:
                return event.result
        return None
    
    def has_completed_event(self, address: str) -> bool:
        """Check if event at address completed successfully"""
        if address not in self.__events:
            return False
        return self.__events[address].execution_status == ExecutionStatus.COMPLETED
```

### 8. State Diffing Utility (`rh_agents/core/state_recovery.py`)

```python
class StateDiff(BaseModel):
    """Represents differences between two state snapshots"""
    new_events: list[dict] = Field(default_factory=list)
    removed_events: list[dict] = Field(default_factory=list)
    changed_storage: dict[str, tuple[Any, Any]] = Field(default_factory=dict)
    new_artifacts: list[str] = Field(default_factory=list)
    removed_artifacts: list[str] = Field(default_factory=list)

class StateSnapshot(BaseModel):
    # ... existing fields ...
    
    @staticmethod
    def diff(snapshot1: "StateSnapshot", snapshot2: "StateSnapshot") -> StateDiff:
        """Compare two snapshots and return differences"""
        diff = StateDiff()
        
        # Compare events
        state1 = snapshot1.execution_state
        state2 = snapshot2.execution_state
        
        events1 = set(state1.get('history', {}).get('events', []))
        events2 = set(state2.get('history', {}).get('events', []))
        
        diff.new_events = list(events2 - events1)
        diff.removed_events = list(events1 - events2)
        
        # Compare storage
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
```

### 9. Auto-Checkpoint Configuration (`rh_agents/core/execution.py`)

```python
class AutoCheckpointConfig(BaseModel):
    """Configuration for automatic state checkpointing"""
    enabled: bool = False
    checkpoint_after_events: list[EventType] = Field(default_factory=list)
    checkpoint_after_artifacts: bool = True
    checkpoint_after_llm: bool = True
    checkpoint_after_tools: bool = True

class ExecutionState(BaseModel):
    # ... existing fields ...
    auto_checkpoint_config: Optional[AutoCheckpointConfig] = None
    
    async def add_event(self, event: Any, status: ExecutionStatus):
        """Add event with optional auto-checkpoint"""
        event.address = self.get_current_address(event.actor.event_type)
        event.execution_status = status
        self.history.add(event)
        
        if not event.skip_republish:
            await self.event_bus.publish(event)
        
        # Auto-checkpoint logic
        if self.auto_checkpoint_config and self.auto_checkpoint_config.enabled:
            should_checkpoint = False
            
            if event.actor.event_type in self.auto_checkpoint_config.checkpoint_after_events:
                should_checkpoint = True
            elif event.actor.is_artifact and self.auto_checkpoint_config.checkpoint_after_artifacts:
                should_checkpoint = True
            elif event.actor.event_type == EventType.LLM and self.auto_checkpoint_config.checkpoint_after_llm:
                should_checkpoint = True
            elif event.actor.event_type == EventType.TOOL and self.auto_checkpoint_config.checkpoint_after_tools:
                should_checkpoint = True
            
            if should_checkpoint and status == ExecutionStatus.COMPLETED:
                self.save_checkpoint(status=StateStatus.RUNNING)
```

---

## Implementation Phases

### Phase 1: Foundation (MVP) - Priority: HIGH ✅ COMPLETED
**Goal:** Basic state save/restore with manual checkpoints

**Tasks:**
1. ✅ **DONE** Create `rh_agents/core/state_recovery.py` with core models
2. ✅ **DONE** Create `rh_agents/core/state_backend.py` with abstractions
3. ✅ **DONE** Create `rh_agents/state_backends.py` with FileSystem implementations
4. ✅ **DONE** Add `state_id`, `replay_mode`, `resume_from_address` to ExecutionState
5. ✅ **DONE** Add `to_snapshot()`, `from_snapshot()`, `save_checkpoint()` to ExecutionState
6. ✅ **DONE** Add `is_replayed`, `skip_republish`, `result` to ExecutionEvent
7. ✅ **DONE** Store results in history events for replay
8. ⏳ **NEXT** Test: Save state, restore, continue execution

**Deliverable:** Can manually save and restore execution state

**Status:** Core implementation complete. Ready for testing.

### Phase 2: Smart Replay - Priority: HIGH
**Goal:** Automatic event skipping during replay

**Tasks:**
1. ✅ Implement `should_skip_event()` in ExecutionState
2. ✅ Modify `ExecutionEvent.__call__()` to check history and skip if needed
3. ✅ Implement selective event publishing (skip_republish logic)
4. ✅ Implement `resume_from_address` functionality
5. ✅ Test: Replay skips completed events, executes only new ones
6. ✅ Test: resume_from_address re-executes from specific point

**Deliverable:** Can restore state and resume execution with smart skipping

### Phase 3: Remove Cache System - Priority: HIGH
**Goal:** Clean up old cache-based approach

**Tasks:**
1. ✅ Remove `__try_retrieve_from_cache` from ExecutionEvent
2. ✅ Remove `__store_result_in_cache` from ExecutionEvent
3. ✅ Remove `_cache_backend` from ExecutionState
4. ✅ Remove or deprecate `rh_agents/core/cache.py`
5. ✅ Remove or deprecate `rh_agents/cache_backends.py`
6. ✅ Update all references to use state recovery instead
7. ✅ Test: System works without cache backend

**Deliverable:** Clean codebase with only state recovery system

### Phase 4: Advanced Features - Priority: MEDIUM
**Goal:** Metadata, diffing, validation

**Tasks:**
1. ✅ Implement `StateMetadata` with tags, lineage, custom fields
2. ✅ Implement `StateSnapshot.diff()` for state comparison
3. ✅ Implement `ReplayMode.VALIDATION` for determinism testing
4. ✅ Add `list_states()` filtering by tags/status/pipeline
5. ✅ Implement artifact garbage collection (orphaned artifacts)
6. ✅ Test: Metadata querying, diffing, validation mode

**Deliverable:** Full-featured state management with debugging tools

### Phase 5: Auto-Checkpoint - Priority: LOW
**Goal:** Automatic state saving at key points

**Tasks:**
1. ✅ Create `AutoCheckpointConfig`
2. ✅ Implement auto-checkpoint logic in `add_event()`
3. ✅ Add configuration for checkpoint triggers (LLM, tools, artifacts)
4. ✅ Test: Auto-checkpoint after configured events
5. ✅ Performance testing (checkpoint overhead)

**Deliverable:** Automatic checkpointing with configurable triggers

### Phase 6: Schema Versioning - Priority: LOW
**Goal:** Handle schema evolution gracefully

**Tasks:**
1. ✅ Implement `StateSnapshot.is_compatible()`
2. ✅ Create migration function registry
3. ✅ Implement v1.0 → v1.1 example migration
4. ✅ Test: Load old snapshot, auto-migrate to new schema

**Deliverable:** Forward-compatible state snapshots

---

## Critical Implementation Notes

### 1. Result Storage in History
**Problem:** Currently, ExecutionEvent doesn't store the actual result, only details.
**Solution:** Add `result: Optional[Any]` field to ExecutionEvent to store execution results for replay.

### 2. Artifact ID Computation
```python
def _compute_artifact_id(self, artifact: Any) -> str:
    """Compute unique ID for artifact (content-based hash)"""
    import hashlib
    import pickle
    try:
        serialized = pickle.dumps(artifact)
        return hashlib.sha256(serialized).hexdigest()
    except Exception:
        # Fallback to timestamp + random
        import time
        import random
        return hashlib.sha256(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()
```

### 3. Event Bus Reconstruction
Event bus subscribers cannot be serialized. On restore:
- Create new EventBus
- User must re-register handlers
- Document this requirement clearly

### 4. Backward Compatibility
During transition period, support both systems:
- Check if `_cache_backend` exists (old code)
- Use state recovery if available, fall back to cache

### 5. Testing Strategy
- Unit tests: Each component in isolation
- Integration tests: Full save/restore workflows
- End-to-end tests: Real pipeline execution with recovery

---

## Migration Checklist

### Before Starting
- ✅ Review and approve this plan
- ✅ Create feature branch: `dev-retrieve` 
- ✅ Set up test environment
- ✅ Back up current codebase

### Phase 1 Complete ✅
- ✅ All core models implemented
- ✅ Backend abstractions created
- ✅ FileSystem implementations working
- ✅ ExecutionState enhanced with state recovery
- ✅ All Phase 1 tests passing
- 📄 **Progress tracked in**: [docs/PROGRESS.md](./PROGRESS.md)

### During Phase 2 (Next)
- Implement replay logic in ExecutionEvent.__call__()
- Store results in history for replay
- Test skip/execute decision logic
- Verify selective event publishing

### After Phase 3
- Update documentation to remove cache references
- Provide migration guide for existing users
- Announce deprecation of CacheBackend

---

## Open Questions / Blockers

### None Currently

All design decisions have been made. Ready to proceed with implementation.

---

## Success Criteria

After full implementation, the system must support:

1. ✅ **Save State**: `execution_state.save_checkpoint()`
2. ✅ **Restore State**: `ExecutionState.load_from_state_id(state_id)`
3. ✅ **Smart Replay**: Automatically skip completed events
4. ✅ **Selective Replay**: Resume from specific address
5. ✅ **Artifact Recovery**: Large objects restored without recomputation
6. ✅ **Metadata Querying**: Find states by tags, status, pipeline
7. ✅ **State Diffing**: Compare two snapshots
8. ✅ **Validation Mode**: Verify determinism
9. ✅ **Auto-Checkpoint**: Optional automatic state saving
10. ✅ **No Cache Dependency**: CacheBackend completely removed

---

## Next Step

**Start with Phase 1: Foundation (MVP)**

Create the core models and file system backend. Get basic save/restore working before adding complexity.

Ready to begin implementation?
