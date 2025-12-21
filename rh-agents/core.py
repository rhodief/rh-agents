import asyncio
import datetime
from enum import Enum
from time import time
from typing import Callable, Literal
from pydantic import BaseModel, Field


class EventType(str, Enum):
    AGENT_CALL = 'agent_call'
    TOOL_CALL = 'tool_call'
    LLM_CALL = 'llm_call'


class ExecutionStatus(str, Enum):
    STARTED = 'started'
    COMPLETED = 'completed'
    FAILED = 'failed'
    AWAITING = 'awaiting'
    HUMAN_INTERVENTION = 'human_intervention'

class BaseActor(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    handler: Callable
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType
    
    async def run_preconditions(self, input_data, orchestration_state: 'OrchestrationState'):
        """Run all precondition checks before execution (async)"""
        for precondition in self.preconditions:
            if asyncio.iscoroutinefunction(precondition):
                await precondition(input_data, orchestration_state)
            else:
                precondition(input_data, orchestration_state)

    async def run_postconditions(self, result, orchestration_state: 'OrchestrationState'):
        """Run all postcondition checks after execution (async)"""
        for postcondition in self.postconditions:
            if asyncio.iscoroutinefunction(postcondition):
                await postcondition(result, orchestration_state)
            else:
                postcondition(result, orchestration_state)

class ExecutionEvent(BaseModel):
    actor: BaseActor
    datetime: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="Timestamp of the event in milliseconds since epoch")
    address: str = Field(default="", description="Address of the agent triggering the event on exectution tree")
    execution_time: float | None = Field(None, description="Execution time in seconds")
    execution_status: ExecutionStatus = Field(ExecutionStatus.STARTED, description="Status of the execution event")
    message: str | None = Field(None, description="Optional message associated with the event")
    
    def start_timer(self):
        self._start_time = time()
    
    def stop_timer(self):
        if hasattr(self, '_start_time'):
            self.execution_time = time() - self._start_time
        else:
            self.execution_time = None
    
    async def __call__(self, input_data, orchestration_state: 'OrchestrationState'):
        """
        Execute the wrapped actor with full lifecycle management (async only).
        """
        orchestration_state.push_context(self.actor.name)
        try:
            # Run preconditions
            await self.actor.run_preconditions(input_data, orchestration_state)

            # Start timer and mark as started
            self.start_timer()
            orchestration_state.add_event(self, ExecutionStatus.STARTED)

            # Enforce async handler
            if not asyncio.iscoroutinefunction(self.actor.handler):
                raise TypeError(f"Handler for actor '{self.actor.name}' must be async.")
            result = await self.actor.handler(input_data, orchestration_state)

            # Run postconditions
            await self.actor.run_postconditions(result, orchestration_state)

            # Stop timer and mark as completed
            self.stop_timer()
            orchestration_state.add_event(self, ExecutionStatus.COMPLETED)

            return result

        except Exception as e:
            # Stop timer, mark as failed and capture error message
            self.stop_timer()
            self.message = str(e)
            orchestration_state.add_event(self, ExecutionStatus.FAILED)
            raise

        finally:
            orchestration_state.pop_context()

    def __call_sync__(self, input_data, orchestration_state: 'OrchestrationState'):
        raise NotImplementedError("Synchronous execution is not supported. Use 'await' on the event.")

class EventBus(BaseModel):
    subscribers: list[Callable] = Field(default_factory=list)
    events: list[ExecutionEvent] = Field(default_factory=list)

    def subscribe(self, handler: Callable):
        """Subscribe a handler to all events."""
        self.subscribers.append(handler)

    def publish(self, event: ExecutionEvent):
        """Publish event to all subscribers and store in event log."""
        self.events.append(event)
        for handler in self.subscribers:
            handler(event)

    def event_stream(self):
        """Generator for all published events (for async-style consumption)."""
        for event in self.events:
            yield event
    
class Tool(BaseActor):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    handler: Callable
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.TOOL_CALL
    
class ToolSet(BaseModel):
    tools: list[Tool] = Field(default_factory=list)
    __tools: dict[str, Tool] = {}

    def __init__(self, tools: list[Tool] | None = None, **data):
        if tools is None:
            tools = []
        super().__init__(tools=tools, **data)
        self.__tools = {tool.name: tool for tool in self.tools}

    def __iter__(self):
        for name, tool in self.__tools.items():
            yield name, tool

    def get_tool_list(self) -> list[Tool]:
        return list(self.__tools.values())
    
    def __getitem__(self, key: str) -> Tool:
        return self.__tools[key]


class LLM(BaseActor):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    handler: Callable
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.LLM_CALL

class Agent(BaseActor):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    handler: Callable
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.AGENT_CALL
    tools: ToolSet
    llm: LLM
    
    
class DoctrineStep(BaseModel):    
    index: int
    description: str
    feasible: bool
    required_steps: list[int] = Field(default_factory=list)
    
class Doctrine(BaseModel):
    goal: str
    constraints: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)
    steps: list[DoctrineStep] = Field(default_factory=list)
    
class ExecutionContext(BaseModel):
    current_step_index: int | None = None
    step_statuses: dict[int, Literal['pending', 'in_progress', 'completed', 'failed']] = {}
    step_results: dict[int, str] = {}

class HistorySet(BaseModel):
    events: list[ExecutionEvent] = Field(default_factory=list)
    __events: dict[str, ExecutionEvent] = {}

    def __init__(self, events: list[ExecutionEvent] | None = None, **data):
        if events is None:
            events = []
        super().__init__(events=events, **data)
        self.__events = {event.address: event for event in self.events}

    def __iter__(self):
        for address, event in self.__events.items():
            yield address, event

    def get_event_list(self) -> list[ExecutionEvent]:
        return list(self.__events.values())
    
    def __getitem__(self, key: str) -> ExecutionEvent:
        return self.__events[key]
    
    def __setitem__(self, key: str, value: ExecutionEvent):
        """Store only the latest event for each address (overwrites previous status)"""
        self.__events[key] = value
        self.events = list(self.__events.values())
        
    def add(self, event: ExecutionEvent):
        """Add event, keeping only the latest status per address"""
        self.__setitem__(event.address, event)
    
class OrchestrationState(BaseModel):
    doctrine: Doctrine
    execution_context: ExecutionContext | None = None
    history: HistorySet = Field(default_factory=HistorySet)
    event_bus: EventBus = Field(default_factory=EventBus)
    execution_stack: list[str] = Field(default_factory=list, description="Stack tracking current execution path (agent/tool names)")
    
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
    
    def add_event(self, event: ExecutionEvent, status: ExecutionStatus):
        """Add event to history and publish to event bus"""
        event.address = self.get_current_address(event.actor.event_type)
        event.execution_status = status
        self.history.add(event)
        self.event_bus.publish(event)
    
class Orchestrator(BaseActor):
    name: str
    description: str
    state: OrchestrationState | None = None
    
    async def __call__(self, input_data, orchestration_state: OrchestrationState):
        raise NotImplementedError("Orchestrator must implement its own __call__ method.")
    
    