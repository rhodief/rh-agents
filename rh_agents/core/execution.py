from __future__ import annotations
import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Callable, Union, Optional, Awaitable, AsyncGenerator
from pydantic import BaseModel, Field, field_serializer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rh_agents.core.events import ExecutionEvent
    from rh_agents.core.state_backend import StateBackend, ArtifactBackend
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.state_recovery import ReplayMode
import inspect

class ExecutionStore(BaseModel):
    data: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    
    def get(self, key: str) -> str | None:
        return self.data.get(key, None)
    
    def set(self, key: str, value: str):
        self.data[key] = value
    
    def get_artifact(self, key: str) -> Any | None:
        """Retrieve an artifact by key."""
        return self.artifacts.get(key, None)
    
    def set_artifact(self, key: str, value: Any):
        """Store an artifact by key."""
        self.artifacts[key] = value
    
    def has_artifact(self, key: str) -> bool:
        """Check if an artifact exists."""
        return key in self.artifacts


class HistorySet(BaseModel):
    events: list[Any] = Field(default_factory=list)  # Use Any instead of ExecutionEvent
    __events: dict[str, Any] = {}

    def __init__(self, events: list[Any] | None = None, **data):
        if events is None:
            events = []
        super().__init__(events=events, **data)
        self.__events = {event.address: event for event in self.events}

    def __iter__(self):
        for address, event in self.__events.items():
            yield address, event

    def get_event_list(self) -> list[Any]:
        return list(self.__events.values())
    
    def __getitem__(self, key: str) -> Any:
        return self.__events[key]
    
    def __setitem__(self, key: str, value: Any):
        """Store event and update latest event lookup for the address"""
        self.__events[key] = value
        self.events.append(value)
        
    def add(self, event: Any):
        """Add event to history, keeping all events while updating latest lookup"""
        self.__setitem__(event.address, event)
    
    def get_event_result(self, address: str) -> Optional[Any]:
        """Get the result from a completed event at given address."""
        if address in self.__events:
            event = self.__events[address]
            if hasattr(event, 'execution_status') and event.execution_status == ExecutionStatus.COMPLETED:
                return getattr(event, 'result', None)
        return None
    
    def has_completed_event(self, address: str) -> bool:
        """Check if event at address completed successfully."""
        if address not in self.__events:
            return False
        event = self.__events[address]
        return hasattr(event, 'execution_status') and event.execution_status == ExecutionStatus.COMPLETED


class EventBus(BaseModel):
    subscribers: list[Callable] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)
    queue: asyncio.Queue = Field(default_factory=asyncio.Queue)

    class Config:
        arbitrary_types_allowed = True

    def subscribe(self, handler: Callable):
        self.subscribers.append(handler)

    async def publish(self, event: ExecutionEvent):
        self.events.append(event)

        for handler in self.subscribers:
            event_copy = (
                event.model_copy()
                if hasattr(event, "model_copy")
                else event
            )

            result = handler(event_copy)
            if asyncio.iscoroutine(result):
                await result
        
        # Yield control to allow other tasks (like stream generators) to process the event
        await asyncio.sleep(0)

    async def stream(self) -> AsyncGenerator[Any, None]:
        while True:
            event = await self.queue.get()
            yield event



class ExecutionState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    # State recovery fields
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this execution state")
    replay_mode: ReplayMode = Field(default=ReplayMode.NORMAL, description="How to handle replay of already-executed events")
    resume_from_address: Optional[str] = Field(default=None, description="Address to resume execution from (replay all events after this)")
    
    # Core state components
    storage: ExecutionStore = Field(default_factory=lambda: ExecutionStore())
    current_execution: Union[Any, None] = None  # Use Any instead of ExecutionEvent
    history: HistorySet = Field(default_factory=HistorySet)
    execution_stack: list[str] = Field(default_factory=list, description="Stack tracking current execution path (agent/tool names)")
    
    # Runtime components (not serialized, stored as plain attributes)
    event_bus: EventBus = Field(default_factory=EventBus, exclude=True)
    
    def __init__(
        self, 
        cache_backend: Optional[Any] = None,
        state_backend: Optional["StateBackend"] = None,
        artifact_backend: Optional["ArtifactBackend"] = None,
        **data
    ):
        """Initialize ExecutionState with optional backends."""
        super().__init__(**data)
        # Store backends as instance attributes (not Pydantic fields)
        self._cache_backend = cache_backend  # DEPRECATED
        self._state_backend = state_backend
        self._artifact_backend = artifact_backend
    
    @property
    def cache_backend(self) -> Optional[Any]:
        """Get the cache backend instance. DEPRECATED: Use state recovery instead."""
        return getattr(self, '_cache_backend', None)
    
    @cache_backend.setter
    def cache_backend(self, value: Optional[Any]):
        """Set the cache backend instance. DEPRECATED: Use state recovery instead."""
        self._cache_backend = value
    
    @property
    def state_backend(self) -> Optional["StateBackend"]:
        """Get the state backend instance."""
        return getattr(self, '_state_backend', None)
    
    @state_backend.setter
    def state_backend(self, value: Optional["StateBackend"]):
        """Set the state backend instance."""
        self._state_backend = value
    
    @property
    def artifact_backend(self) -> Optional["ArtifactBackend"]:
        """Get the artifact backend instance."""
        return getattr(self, '_artifact_backend', None)
    
    @artifact_backend.setter
    def artifact_backend(self, value: Optional["ArtifactBackend"]):
        """Set the artifact backend instance."""
        self._artifact_backend = value
    
    
    def get_current_address(self, event_type: EventType) -> str:
        """Build address from execution stack + event type, e.g. 'doctrine_agent::tool_call'"""
        if not self.execution_stack:
            return event_type.value
        return "::" .join(self.execution_stack) + "::" + event_type.value
    
    def push_context(self, name: str):
        """Enter a new execution level (agent, tool, llm)"""
        self.execution_stack.append(name)
    
    def pop_context(self) -> str | None:
        """Exit current execution level"""
        if self.execution_stack:
            return self.execution_stack.pop()
        return None
    
    async def add_event(self, event: Any, status: ExecutionStatus):
        """Add event to history and conditionally publish to event bus."""
        event.address = self.get_current_address(event.actor.event_type)
        event.execution_status = status
        self.history.add(event)
        
        # Only publish if not skipped (for replay control)
        if not getattr(event, 'skip_republish', False):
            await self.event_bus.publish(event)
        
    def add_step_result(self, step_index: int, value: Any):
        """Store step result in execution storage with key 'step_{index}'"""
        self.storage.set(f"step::{step_index}", str(value))
        
    def get_steps_result(self, step_index: list[int]) -> list[Any]:
        """Retrieve step result from execution storage with key 'step_{index}'"""
        results = []
        for index in step_index:
            item = self.storage.get(f"step::{index}")
            if item is not None:
                results.append(item)
        return results
    
    def get_last_step_result(self) -> Any | None:
        """Retrieve last step result from execution storage with key 'step_{index}'"""
        step_keys = [key for key in self.storage.data.keys() if key.startswith("step::")]
        if not step_keys:
            return None
        last_key = max(step_keys, key=lambda k: int(k.split("::")[1]))
        return self.storage.get(last_key)
            
    def get_all_steps_results(self) -> dict[str, str]:
        """Retrieve all stored execution results"""
        return self.storage.data.copy()
    
    def should_skip_event(self, address: str) -> bool:
        """
        Check if event should be skipped during replay.
        
        Logic:
        - VALIDATION mode: Never skip (re-execute everything)
        - With resume_from_address: Skip until we reach that address
        - Normal replay: Skip if event completed successfully in history
        
        Args:
            address: Event address to check
            
        Returns:
            True if event should be skipped, False if it should execute
        """
        if self.replay_mode == ReplayMode.VALIDATION:
            return False  # Always execute for validation
        
        # If resume_from_address is set, skip everything before it
        if self.resume_from_address:
            # Check if we've reached the resume point
            if address == self.resume_from_address:
                # Clear resume_from_address so subsequent events execute
                self.resume_from_address = None
                return False  # Execute this event
            # Skip if we haven't reached resume point yet AND event exists in history
            return self.history.has_completed_event(address)
        
        # Normal replay: skip if already completed
        return self.history.has_completed_event(address)
    
    def to_snapshot(
        self,
        status: Optional["StateStatus"] = None,
        metadata: Optional["StateMetadata"] = None
    ) -> "StateSnapshot":
        """
        Create a snapshot of current execution state for persistence.
        
        This includes:
        - Core state (history, storage, stack)
        - Artifact references (artifacts stored separately)
        - Metadata and timestamps
        
        Runtime components (event_bus, backends) are excluded.
        
        Args:
            status: Execution status (default: RUNNING)
            metadata: Optional metadata to attach
            
        Returns:
            StateSnapshot ready for persistence
        """
        from datetime import datetime
        from rh_agents.core.state_recovery import StateSnapshot, StateStatus, StateMetadata
        from rh_agents.state_backends import compute_artifact_id
        
        # Serialize core state (exclude runtime components)
        state_dict = self.model_dump(
            exclude={'event_bus'}
        )
        
        # Extract artifact references and save artifacts separately
        artifact_refs = {}
        for key, artifact in self.storage.artifacts.items():
            artifact_id = compute_artifact_id(artifact)
            artifact_refs[key] = artifact_id
            
            # Save artifact to backend if available
            if self.artifact_backend:
                self.artifact_backend.save_artifact(artifact_id, artifact)
        
        # Create snapshot
        return StateSnapshot(
            state_id=self.state_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status=status or StateStatus.RUNNING,
            metadata=metadata or StateMetadata(),
            execution_state=state_dict,
            artifact_refs=artifact_refs
        )
    
    @classmethod
    def from_snapshot(
        cls,
        snapshot: "StateSnapshot",
        state_backend: Optional["StateBackend"] = None,
        artifact_backend: Optional["ArtifactBackend"] = None,
        event_bus: Optional[EventBus] = None,
        replay_mode: ReplayMode = ReplayMode.NORMAL,
        resume_from_address: Optional[str] = None
    ) -> "ExecutionState":
        """
        Restore ExecutionState from a snapshot.
        
        Reconstructs:
        - Core state from serialized data
        - Artifacts from artifact backend
        - Runtime components (event bus, backends)
        
        Args:
            snapshot: StateSnapshot to restore from
            state_backend: Backend for state persistence
            artifact_backend: Backend for artifact storage
            event_bus: Event bus instance (creates new if None)
            replay_mode: How to handle replay
            resume_from_address: Optional address to resume from
            
        Returns:
            Restored ExecutionState ready for execution
        """
        # Deserialize core state
        state = cls(**snapshot.execution_state)
        
        # Restore artifacts from backend
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
        status: Optional["StateStatus"] = None,
        metadata: Optional["StateMetadata"] = None
    ) -> bool:
        """
        Save current state to backend as a checkpoint.
        
        Args:
            status: Execution status (default: RUNNING)
            metadata: Optional metadata to attach
            
        Returns:
            True if successful, False if no backend configured or save failed
        """
        if not self.state_backend:
            return False
        
        snapshot = self.to_snapshot(status, metadata)
        return self.state_backend.save_state(snapshot)
    
    @classmethod
    def load_from_state_id(
        cls,
        state_id: str,
        state_backend: "StateBackend",
        artifact_backend: Optional["ArtifactBackend"] = None,
        event_bus: Optional[EventBus] = None,
        replay_mode: ReplayMode = ReplayMode.NORMAL,
        resume_from_address: Optional[str] = None
    ) -> Optional["ExecutionState"]:
        """
        Load ExecutionState by its unique ID.
        
        Convenience method that loads snapshot and restores state.
        
        Args:
            state_id: Unique state identifier
            state_backend: Backend to load from
            artifact_backend: Backend for artifact loading
            event_bus: Event bus instance (creates new if None)
            replay_mode: How to handle replay
            resume_from_address: Optional address to resume from
            
        Returns:
            Restored ExecutionState if found, None otherwise
        """
        snapshot = state_backend.load_state(state_id)
        if not snapshot:
            return None
        
        return cls.from_snapshot(
            snapshot,
            state_backend,
            artifact_backend,
            event_bus,
            replay_mode,
            resume_from_address
        )
    