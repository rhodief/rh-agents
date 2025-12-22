from __future__ import annotations
from typing import Any, Callable, Union, Optional
from pydantic import BaseModel, Field, field_serializer
from rh_agents.core.types import EventType, ExecutionStatus


class ExecutionStore(BaseModel):
    data: dict[str, str] = Field(default_factory=dict)
    
    def get(self, key: str) -> str | None:
        return self.data.get(key, None)
    
    def set(self, key: str, value: str):
        self.data[key] = value


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


class EventBus(BaseModel):
    subscribers: list[Callable] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)  # Use Any instead of ExecutionEvent

    def subscribe(self, handler: Callable):
        """Subscribe a handler to all events."""
        self.subscribers.append(handler)

    def publish(self, event: Any):
        """Publish event to all subscribers and store in event log."""
        self.events.append(event)
        for handler in self.subscribers:
            handler(event)

    async def event_stream(self):
        """Async generator for all published events."""
        for event in self.events:
            yield event


class ExecutionState(BaseModel):
    storage: ExecutionStore = Field(default_factory=lambda: ExecutionStore())
    current_execution: Union[Any, None] = None  # Use Any instead of ExecutionEvent
    history: HistorySet = Field(default_factory=HistorySet)
    event_bus: EventBus = Field(default_factory=EventBus)    
    execution_stack: list[str] = Field(default_factory=list, description="Stack tracking current execution path (agent/tool names)")
    _cache_backend: Optional[Any] = None  # Private field to avoid serialization issues
    
    @property
    def cache_backend(self) -> Optional[Any]:
        """Get the cache backend instance."""
        return self._cache_backend
    
    @cache_backend.setter
    def cache_backend(self, value: Optional[Any]):
        """Set the cache backend instance."""
        self._cache_backend = value
    
    def __init__(self, cache_backend: Optional[Any] = None, **data):
        """Initialize ExecutionState with optional cache backend."""
        super().__init__(**data)
        self._cache_backend = cache_backend
    
    
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
    
    def add_event(self, event: Any, status: ExecutionStatus):
        """Add event to history and publish to event bus"""
        event.address = self.get_current_address(event.actor.event_type)
        event.execution_status = status
        self.history.add(event)
        self.event_bus.publish(event)
        
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
    