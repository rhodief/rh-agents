# RH-Agents Refactoring Implementation Specification

**Version:** 1.0.0  
**Date:** January 24, 2026  
**Target Audience:** LLM Implementation Agent

---

## Overview

This document provides consolidated decisions and actionable implementation steps for refactoring the rh_agents package. All design decisions have been finalized and approved.

---

## Approved Decisions Summary

### D1: Generic Types - REMOVE
- Remove `Generic[T]` from `LLM`, `Generic[OutputT]` from `ExecutionEvent`, `Generic[T]` from `ExecutionResult`
- Create `.pyi` stub files to maintain type safety for type checkers
- Simplify instantiation: `ExecutionEvent(actor=llm)` instead of `ExecutionEvent[LLM_Result](actor=llm)`

### D2: Result Types - PROTOCOL + KEEP DOMAIN TYPES
- Remove unused `Agent_Result` class
- Keep `LLM_Result` and `Tool_Result` as domain-specific types
- Create `ActorOutput` protocol to tie them together for generic code
- Maintain type safety while allowing domain-specific fields

### D3: ToolSet - SIMPLIFY WITH COMPUTED DICT
- Keep as Pydantic model for serialization
- Primary storage: `list[Tool]`
- Add computed `@property` for dict lookup: `by_name`
- Remove dual storage complexity

### D4: Deprecated Cache Modules - REMOVE COMPLETELY
- Delete `rh_agents/core/cache.py`
- Delete `rh_agents/cache_backends.py`
- **IMPORTANT:** Keep `cacheable` field in actors - it's used by state recovery system, NOT deprecated cache
- Update all imports and references

### D5: Backend Management - PYDANTIC EXCLUDE
- Use `Field(exclude=True)` for `state_backend` and `artifact_backend`
- Set `model_config = {"arbitrary_types_allowed": True}`
- More idiomatic than private attributes

### D6: Field Redeclaration - REMOVE
- Remove redundant field declarations in `Tool`, `LLM`, `Agent` subclasses
- Only declare NEW fields or OVERRIDDEN fields (with different defaults)
- Use docstrings to document inherited fields

### D7: Public API - FLAT EXPORTS
- Populate `rh_agents/__init__.py` with public API
- Enable simple imports: `from rh_agents import Agent, Tool, LLM`
- Add `__version__` and `__all__`

### D8: Handler Types - TYPE ALIAS
- Use simple type alias: `AsyncHandler = Callable[[BaseModel, str, ExecutionState], Awaitable[Any]]`
- Add clear docstring with expected signature
- Maintain Awaitable (not Coroutine) for flexibility

### D9: Context Parameter - MAKE OPTIONAL
- Change signature: `async def handler(input_data, state, context: str = "") -> result`
- Backward compatible
- Document when to use context parameter

### D10: Event Address - KEEP CURRENT
- Maintain string-based hierarchical addresses
- No changes needed
- Already works well for replay and tracking

---

## Implementation Phases

### PHASE 1: Non-Breaking Improvements (v1.1.0)
**Estimated: 1-2 days**

#### 1.1: Public API Exports
**File:** `rh_agents/__init__.py`

```python
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
from rh_agents.core.execution import ExecutionState, ExecutionEvent
from rh_agents.core.events import ExecutionResult
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
```

#### 1.2: Update Examples
Update all examples to use new imports:
```python
# Old
from rh_agents.core.actors import Agent, Tool, LLM
from rh_agents.core.execution import ExecutionState

# New
from rh_agents import Agent, Tool, LLM, ExecutionState
```

**Files to update:**
- `examples/index.py`
- `examples/cached_index.py`
- `examples/parallel_*.py`
- All other example files

#### 1.3: Enhanced Documentation
Add/update docstrings in:
- `rh_agents/core/actors.py` - All actor classes
- `rh_agents/core/execution.py` - ExecutionState methods
- `rh_agents/core/events.py` - ExecutionEvent and ExecutionResult

---

### PHASE 2: Simplification (v1.5.0)
**Estimated: 3-4 days**

#### 2.1: Remove Generic Types

**File:** `rh_agents/core/actors.py`

**Changes:**
1. Remove `Generic[T]` from LLM class:
```python
# BEFORE
T = TypeVar('T', bound=Any)
class LLM(BaseActor, Generic[T]):
    handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]

# AFTER
class LLM(BaseActor):
    handler: Callable[[BaseModel, str, ExecutionState], Awaitable[LLM_Result]]
```

**File:** `rh_agents/core/events.py`

**Changes:**
2. Remove `Generic[OutputT]` from ExecutionEvent:
```python
# BEFORE
class ExecutionEvent(BaseModel, Generic[OutputT]):
    async def __call__(...) -> ExecutionResult[OutputT]:

# AFTER
class ExecutionEvent(BaseModel):
    async def __call__(...) -> ExecutionResult:
```

3. Remove `Generic[T]` from ExecutionResult:
```python
# BEFORE
class ExecutionResult(BaseModel, Generic[T]):
    result: Union[T, None] = ...

# AFTER
class ExecutionResult(BaseModel):
    result: Any = None
```

**Update all usages:**
- Search and replace: `ExecutionEvent[.*?]` → `ExecutionEvent`
- Search and replace: `ExecutionResult[.*?]` → `ExecutionResult`
- Search and replace: `LLM[.*?]` → `LLM`

**Files affected:**
- `rh_agents/agents.py`
- `rh_agents/llms.py`
- All example files
- All test files

#### 2.2: Create Type Stub Files

**File:** `rh_agents/core/actors.pyi`
```python
from typing import TypeVar, Generic, Awaitable, Callable
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')

class BaseActor:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    handler: Callable[..., Awaitable[Any]]
    # ... other fields

class LLM(BaseActor, Generic[T, R]):
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        **kwargs
    ) -> None: ...

class Tool(BaseActor, Generic[T]):
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        handler: Callable[[T, str, ExecutionState], Awaitable[Any]],
        **kwargs
    ) -> None: ...

class Agent(BaseActor, Generic[T, R]):
    tools: ToolSet
    llm: LLM | None
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        **kwargs
    ) -> None: ...
```

**File:** `rh_agents/core/events.pyi`
```python
from typing import TypeVar, Generic, Awaitable
from pydantic import BaseModel
from rh_agents.core.actors import BaseActor

T = TypeVar('T')

class ExecutionResult(Generic[T]):
    result: T | None
    execution_time: float | None
    ok: bool
    erro_message: str | None

class ExecutionEvent(Generic[T]):
    actor: BaseActor
    async def __call__(self, input_data: Any, context: str, state: ExecutionState) -> ExecutionResult[T]: ...
```

#### 2.3: Clean Up Field Redeclarations

**File:** `rh_agents/core/actors.py`

**Tool class:**
```python
# BEFORE
class Tool(BaseActor):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: Union[type[BaseModel], None] = None
    handler: AsyncHandler
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.TOOL_CALL
    cacheable: bool = Field(default=False, description="Tools are not cacheable by default")
    version: str = Field(default="1.0.0", description="Tool version")
    cache_ttl: Union[int, None] = Field(default=None, description="Cache TTL")
    is_artifact: bool = Field(default=False, description="Artifacts")

# AFTER
class Tool(BaseActor):
    """
    Tool actor for executing specific functions.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, cache_ttl, is_artifact
    """
    # Override defaults
    event_type: EventType = EventType.TOOL_CALL
    cacheable: bool = Field(default=False, description="Tools are not cacheable by default (side effects)")
    
    def model_post_init(self, __context) -> None:
        if self.output_model is None:
            from rh_agents.core.result_types import Tool_Result
            self.output_model = Tool_Result
```

**LLM class:**
```python
# BEFORE
class LLM(BaseActor, Generic[T]):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[LLM_Result] = LLM_Result
    handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.LLM_CALL
    cacheable: bool = Field(default=True, description="LLM cacheable")
    version: str = Field(default="1.0.0", description="LLM version")
    cache_ttl: Union[int, None] = Field(default=3600, description="1 hour TTL")
    is_artifact: bool = Field(default=False, description="Artifacts")

# AFTER
class LLM(BaseActor):
    """
    LLM actor for language model calls.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, is_artifact
    """
    # Override defaults
    output_model: type[LLM_Result] = LLM_Result
    event_type: EventType = EventType.LLM_CALL
    cacheable: bool = Field(default=True, description="LLM calls are cacheable by default")
    cache_ttl: int | None = Field(default=3600, description="Default 1 hour TTL for LLM results")
```

**Agent class:**
```python
# BEFORE
class Agent(BaseActor):
    name: str
    description: str
    handler: AsyncHandler
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.AGENT_CALL
    tools: ToolSet
    llm: LLM | None = None
    is_artifact: bool = Field(default=False, description="Artifacts")
    cacheable: bool = Field(default=False, description="Agents not cacheable")

# AFTER
class Agent(BaseActor):
    """
    Agent actor for orchestrating tools and LLMs.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, cache_ttl, is_artifact
        
    Additional fields:
        tools: Collection of tools available to the agent
        llm: Optional LLM for the agent to use
    """
    # Override defaults
    event_type: EventType = EventType.AGENT_CALL
    cacheable: bool = Field(default=False, description="Agents are not cacheable by default")
    
    # Agent-specific fields
    tools: ToolSet
    llm: LLM | None = None
```

#### 2.4: Simplify ToolSet

**File:** `rh_agents/core/actors.py`

**Replace ToolSet class:**
```python
# BEFORE
class ToolSet(BaseModel):
    tools: list[Tool] = Field(default_factory=list)
    __tools: dict[str, Tool] = {}

    def __init__(self, tools: Union[list[Tool], None] = None, **data):
        if tools is None:
            tools = []
        super().__init__(tools=tools, **data)
        self.__tools = {tool.name: tool for tool in self.tools}

    def __iter__(self):
        for name, tool in self.__tools.items():
            yield name, tool

    def get_tool_list(self) -> list[Tool]:
        return list(self.__tools.values())
    
    def __getitem__(self, key: str) -> Tool | None:
        return self.__tools.get(key)
    
    def get(self, key: str) -> Union[Tool, None]:
        return self.__tools.get(key, None)

# AFTER
class ToolSet(BaseModel):
    """Collection of tools with efficient name-based lookup."""
    tools: list[Tool] = Field(default_factory=list)
    
    @property
    def by_name(self) -> dict[str, Tool]:
        """Get tools as a name-indexed dictionary."""
        return {tool.name: tool for tool in self.tools}
    
    def __iter__(self):
        """Iterate over tools."""
        return iter(self.tools)
    
    def __getitem__(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self.by_name.get(name)
    
    def get(self, name: str) -> Tool | None:
        """Get tool by name, returns None if not found."""
        return self.by_name.get(name)
    
    def __len__(self) -> int:
        """Return number of tools."""
        return len(self.tools)
```

**Update usages:**
- Change `tool_set.get_tool_list()` → `list(tool_set)` or `tool_set.tools`

#### 2.5: Backend Management with Pydantic Exclude

**File:** `rh_agents/core/execution.py`

**Update ExecutionState:**
```python
# BEFORE
class ExecutionState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    # ... fields ...
    
    def __init__(
        self, 
        state_backend: Optional["StateBackend"] = None,
        artifact_backend: Optional["ArtifactBackend"] = None,
        **data
    ):
        super().__init__(**data)
        self._state_backend = state_backend
        self._artifact_backend = artifact_backend
    
    @property
    def state_backend(self) -> Optional["StateBackend"]:
        return getattr(self, '_state_backend', None)
    
    @state_backend.setter
    def state_backend(self, value: Optional["StateBackend"]):
        self._state_backend = value

# AFTER
class ExecutionState(BaseModel):
    """
    Manages execution state including history, storage, and event tracking.
    
    Backends (state_backend, artifact_backend) are excluded from serialization
    as they are runtime dependencies that must be reconstructed on restore.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    # State recovery fields
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    replay_mode: ReplayMode = Field(default=ReplayMode.NORMAL)
    resume_from_address: str | None = Field(default=None)
    _resume_point_reached: bool = False
    
    # Core state components (serializable)
    storage: ExecutionStore = Field(default_factory=ExecutionStore)
    current_execution: Any = None
    history: HistorySet = Field(default_factory=HistorySet)
    execution_stack: list[str] = Field(default_factory=list)
    
    # Runtime components (excluded from serialization)
    event_bus: EventBus = Field(default_factory=EventBus, exclude=True)
    state_backend: Optional["StateBackend"] = Field(default=None, exclude=True)
    artifact_backend: Optional["ArtifactBackend"] = Field(default=None, exclude=True)
```

#### 2.6: Create ActorOutput Protocol

**File:** `rh_agents/core/result_types.py`

**Add protocol and update result types:**
```python
from typing import Protocol, Any, runtime_checkable
from pydantic import BaseModel, Field

@runtime_checkable
class ActorOutput(Protocol):
    """
    Protocol for actor output types.
    
    All actor result types should implement this protocol
    to enable generic code while maintaining domain-specific fields.
    """
    success: bool
    error: str | None

class LLM_Tool_Call(BaseModel):
    """Model representing a tool call made by an LLM"""
    tool_name: str
    arguments: str

class LLM_Result(BaseModel):
    """
    Result type for LLM actor executions.
    
    Implements ActorOutput protocol.
    """
    content: str
    tools: list[LLM_Tool_Call] = Field(default_factory=list)
    tokens_used: int | None = None
    model_name: str | None = None
    error_message: str | None = None
    
    @property
    def is_content(self) -> bool:
        return bool(self.content and not self.tools)

    @property
    def is_tool_call(self) -> bool:
        return len(self.tools) > 0
    
    @property
    def succeeded(self) -> bool:
        """Alias for protocol compatibility."""
        return self.error_message is None
    
    @property
    def success(self) -> bool:
        """Protocol implementation."""
        return self.error_message is None
    
    @property
    def error(self) -> str | None:
        """Protocol implementation."""
        return self.error_message

class Tool_Result(BaseModel):
    """
    Result type for Tool actor executions.
    
    Implements ActorOutput protocol.
    """
    output: Any
    tool_name: str
    success: bool = True
    error: str | None = None

# REMOVE Agent_Result entirely (unused)
```

#### 2.7: Make Context Parameter Optional

**File:** `rh_agents/core/actors.py`

**Update AsyncHandler type alias:**
```python
# BEFORE
AsyncHandler = Callable[..., Coroutine[Any, Any, Any]]

# AFTER
AsyncHandler = Callable[[BaseModel, ExecutionState, str], Awaitable[Any]]
"""
Expected handler signature:
    async def handler(
        input_data: InputModel,
        execution_state: ExecutionState,
        context: str = ""
    ) -> OutputModel
    
The context parameter is optional and contains execution context information.
Use it for LLM prompts or debugging. Most handlers can ignore it.
"""
```

**Update ExecutionEvent.__call__:**
```python
# File: rh_agents/core/events.py

async def __call__(
    self, 
    input_data: Any, 
    execution_state: ExecutionState,
    context: str = ""  # Made optional
) -> ExecutionResult:
    """
    Execute the wrapped actor with replay awareness.
    
    Args:
        input_data: Input for the actor
        execution_state: Current execution state
        context: Optional execution context string (default: "")
    """
    # ... rest of implementation
    
    # When calling handler
    result = await self.actor.handler(input_data, execution_state, context)
```

**Update all handler signatures in examples and agents:**
```python
# BEFORE
async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Doctrine:

# AFTER
async def handler(input_data: Message, execution_state: ExecutionState, context: str = "") -> Doctrine:
```

**Files to update:**
- `rh_agents/agents.py` - All agent handlers
- `examples/*.py` - All example handlers
- `tests/*.py` - All test handlers

#### 2.8: Add Decorator-Based Actor Definition

**File:** `rh_agents/decorators.py` (NEW FILE)

Create a new module for decorator-based actor creation:

```python
"""
Decorator-based API for creating actors.

Provides a more Pythonic and concise way to define tools, agents, and LLMs
using decorators, similar to FastAPI or Flask.
"""
from typing import Callable, TypeVar, Any, Awaitable
from functools import wraps
from pydantic import BaseModel, create_model
from rh_agents.core.actors import Tool, Agent, LLM
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import Tool_Result, LLM_Result

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


def tool(
    name: str | None = None,
    description: str | None = None,
    cacheable: bool = False,
    version: str = "1.0.0"
) -> Callable[[F], Tool]:
    """
    Decorator to create a Tool from an async function.
    
    Usage:
        @tool(name="calculator", description="Performs calculations")
        async def calculate(input: CalculatorArgs, state: ExecutionState) -> ToolResult:
            return ToolResult(output=input.a + input.b, tool_name="calculator")
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        cacheable: Whether results should be cached
        version: Tool version for cache invalidation
    """
    def decorator(func: F) -> Tool:
        # Extract function metadata
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
        # Get type hints to determine input model
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        # First parameter should be the input model
        if not params:
            raise ValueError(f"Tool handler {func.__name__} must have at least one parameter (input_data)")
        
        input_param = params[0]
        if input_param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Tool handler {func.__name__} first parameter must be type-annotated")
        
        input_model = input_param.annotation
        
        # Wrap function to match expected signature
        @wraps(func)
        async def handler(input_data: BaseModel, state: ExecutionState, context: str = "") -> Any:
            # Call with optional context parameter if function accepts it
            if len(params) >= 3:
                return await func(input_data, state, context)
            else:
                return await func(input_data, state)
        
        return Tool(
            name=tool_name,
            description=tool_description,
            input_model=input_model,
            handler=handler,
            cacheable=cacheable,
            version=version
        )
    
    return decorator


def agent(
    name: str | None = None,
    description: str | None = None,
    tools: list[Tool] | None = None,
    llm: LLM | None = None,
    cacheable: bool = False
) -> Callable[[F], Agent]:
    """
    Decorator to create an Agent from an async function.
    
    Usage:
        @agent(name="DoctrineAgent", tools=[tool1, tool2], llm=my_llm)
        async def handle_doctrine(input: Message, state: ExecutionState) -> Doctrine:
            # Agent logic here
            return result
    
    Args:
        name: Agent name (defaults to function name)
        description: Agent description (defaults to function docstring)
        tools: List of tools available to the agent
        llm: LLM instance for the agent
        cacheable: Whether results should be cached
    """
    def decorator(func: F) -> Agent:
        from rh_agents.core.actors import ToolSet
        
        agent_name = name or func.__name__
        agent_description = description or func.__doc__ or f"Agent: {agent_name}"
        
        # Get type hints
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        if not params:
            raise ValueError(f"Agent handler {func.__name__} must have at least one parameter")
        
        input_param = params[0]
        if input_param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Agent handler {func.__name__} first parameter must be type-annotated")
        
        input_model = input_param.annotation
        
        # Get output type from return annotation
        output_model = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        
        @wraps(func)
        async def handler(input_data: BaseModel, state: ExecutionState, context: str = "") -> Any:
            if len(params) >= 3:
                return await func(input_data, state, context)
            else:
                return await func(input_data, state)
        
        return Agent(
            name=agent_name,
            description=agent_description,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
            tools=ToolSet(tools or []),
            llm=llm,
            cacheable=cacheable
        )
    
    return decorator
```

**Update `rh_agents/__init__.py`:**
```python
# Add to exports
from rh_agents.decorators import tool, agent

__all__ = [
    # ... existing exports ...
    "tool",
    "agent",
]
```

**Create example usage:**

**File:** `examples/decorator_example.py` (NEW)
```python
"""
Example of using decorator-based API for defining actors.
"""
import asyncio
from pydantic import BaseModel, Field
from rh_agents import tool, agent, ExecutionState, EventPrinter, Message, AuthorType
from rh_agents.core.result_types import Tool_Result


class CalculatorArgs(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


@tool(name="calculator", description="Performs basic arithmetic operations")
async def calculate(input: CalculatorArgs, state: ExecutionState) -> Tool_Result:
    """Simple calculator tool."""
    operations = {
        "add": input.a + input.b,
        "subtract": input.a - input.b,
        "multiply": input.a * input.b,
        "divide": input.a / input.b if input.b != 0 else float('inf')
    }
    result = operations.get(input.operation, 0)
    return Tool_Result(
        output=f"{input.a} {input.operation} {input.b} = {result}",
        tool_name="calculator"
    )


@agent(name="MathAgent", tools=[calculate], description="Handles math queries")
async def math_agent(input: Message, state: ExecutionState) -> Message:
    """Agent that processes math queries."""
    # Simple logic - in real agent, would use LLM to parse and call tools
    content = f"Processed math query: {input.content}"
    return Message(content=content, author=AuthorType.ASSISTANT)


async def main():
    state = ExecutionState()
    
    # Use the decorated tool
    calc_input = CalculatorArgs(a=10, b=5, operation="add")
    result = await calculate.handler(calc_input, state)
    print(f"Calculator result: {result.output}")
    
    # Use the decorated agent
    msg = Message(content="What is 10 + 5?", author=AuthorType.USER)
    agent_result = await math_agent.handler(msg, state)
    print(f"Agent response: {agent_result.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

#### 2.9: Add Validation Helpers

**File:** `rh_agents/validation.py` (NEW FILE)

```python
"""
Validation utilities for actors and execution state.

Provides helper functions to validate actor configurations and
execution state consistency, helping catch errors early.
"""
from typing import Any
from rh_agents.core.actors import BaseActor, Tool, Agent, LLM
from rh_agents.core.execution import ExecutionState
from pydantic import ValidationError


class ActorValidationError(Exception):
    """Raised when actor validation fails."""
    pass


class StateValidationError(Exception):
    """Raised when state validation fails."""
    pass


def validate_actor(actor: BaseActor) -> None:
    """
    Validate actor configuration.
    
    Checks:
    - Handler is async function
    - Input/output models are valid
    - Required fields are set
    - Version format is valid
    
    Args:
        actor: Actor to validate
        
    Raises:
        ActorValidationError: If validation fails
    """
    errors = []
    
    # Check name
    if not actor.name or not actor.name.strip():
        errors.append("Actor name is required and cannot be empty")
    
    # Check description
    if not actor.description:
        errors.append("Actor description is required")
    
    # Check input model
    if not actor.input_model:
        errors.append("input_model is required")
    else:
        try:
            # Verify it's a valid Pydantic model
            from pydantic import BaseModel
            if not issubclass(actor.input_model, BaseModel):
                errors.append(f"input_model must be a Pydantic BaseModel, got {type(actor.input_model)}")
        except TypeError:
            errors.append(f"input_model is not a valid class: {actor.input_model}")
    
    # Check handler
    if not actor.handler:
        errors.append("handler is required")
    else:
        import asyncio
        import inspect
        if not asyncio.iscoroutinefunction(actor.handler):
            errors.append("handler must be an async function")
        else:
            # Check handler signature
            sig = inspect.signature(actor.handler)
            params = list(sig.parameters.values())
            if len(params) < 2:
                errors.append(f"handler must accept at least 2 parameters (input_data, state), got {len(params)}")
    
    # Check version format
    if actor.version:
        parts = actor.version.split('.')
        if len(parts) != 3:
            errors.append(f"version must be in format 'X.Y.Z', got '{actor.version}'")
        else:
            for part in parts:
                if not part.isdigit():
                    errors.append(f"version parts must be numeric, got '{actor.version}'")
    
    # Type-specific validations
    if isinstance(actor, Agent):
        if actor.tools is None:
            errors.append("Agent must have tools (can be empty ToolSet)")
    
    if isinstance(actor, Tool):
        if actor.cacheable and actor.cache_ttl is None:
            errors.append("Cacheable tools should specify cache_ttl")
    
    if isinstance(actor, LLM):
        if not actor.cacheable:
            errors.append("LLMs should generally be cacheable for efficiency")
    
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ActorValidationError(f"Actor '{actor.name}' validation failed:\n{error_msg}")


def validate_state(state: ExecutionState) -> None:
    """
    Validate execution state consistency.
    
    Checks:
    - History events are valid
    - Storage is consistent
    - Execution stack is balanced
    - Backends are properly configured
    
    Args:
        state: ExecutionState to validate
        
    Raises:
        StateValidationError: If validation fails
    """
    errors = []
    
    # Check state ID
    if not state.state_id:
        errors.append("state_id is required")
    
    # Check execution stack
    if state.execution_stack is None:
        errors.append("execution_stack cannot be None")
    
    # Check history
    if state.history is None:
        errors.append("history cannot be None")
    else:
        # Validate events in history
        for event in state.history.get_event_list():
            if isinstance(event, dict):
                if 'address' not in event:
                    errors.append(f"Event in history missing 'address' field")
                if 'execution_status' not in event:
                    errors.append(f"Event at {event.get('address', 'unknown')} missing 'execution_status'")
    
    # Check storage
    if state.storage is None:
        errors.append("storage cannot be None")
    
    # Check backends if state recovery is being used
    if state.replay_mode != state.replay_mode.NORMAL:
        if not state.state_backend:
            errors.append(f"state_backend required for replay_mode={state.replay_mode}")
    
    # Check for resume_from_address validity
    if state.resume_from_address:
        # Verify address exists in history
        if not any(
            (isinstance(e, dict) and e.get('address') == state.resume_from_address) or
            (hasattr(e, 'address') and e.address == state.resume_from_address)
            for e in state.history.get_event_list()
        ):
            errors.append(f"resume_from_address '{state.resume_from_address}' not found in history")
    
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise StateValidationError(f"ExecutionState validation failed:\n{error_msg}")


def validate_handler_signature(handler: Any, actor_name: str = "unknown") -> None:
    """
    Validate that a handler function has the correct signature.
    
    Expected: async def handler(input_data, state, context="") -> result
    
    Args:
        handler: Handler function to validate
        actor_name: Name of actor for error messages
        
    Raises:
        ActorValidationError: If signature is invalid
    """
    import asyncio
    import inspect
    
    if not asyncio.iscoroutinefunction(handler):
        raise ActorValidationError(f"Handler for '{actor_name}' must be async function")
    
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    
    if len(params) < 2:
        raise ActorValidationError(
            f"Handler for '{actor_name}' must accept at least 2 parameters "
            f"(input_data, state), got {len(params)}"
        )
    
    # Check parameter names (helpful but not required)
    expected_names = ['input_data', 'state', 'context']
    for i, (param, expected) in enumerate(zip(params[:3], expected_names)):
        if param.name != expected:
            # Warning, not error - parameter names are convention
            import warnings
            warnings.warn(
                f"Handler for '{actor_name}' parameter {i} is named '{param.name}', "
                f"expected '{expected}' by convention"
            )
```

**Update `rh_agents/__init__.py`:**
```python
# Add to exports
from rh_agents.validation import (
    validate_actor,
    validate_state,
    validate_handler_signature,
    ActorValidationError,
    StateValidationError
)

__all__ = [
    # ... existing exports ...
    "validate_actor",
    "validate_state",
    "validate_handler_signature",
    "ActorValidationError",
    "StateValidationError",
]
```

**Create validation example:**

**File:** `examples/validation_example.py` (NEW)
```python
"""
Example of using validation helpers.
"""
import asyncio
from pydantic import BaseModel, Field
from rh_agents import Tool, ExecutionState, validate_actor, validate_state
from rh_agents.validation import ActorValidationError, StateValidationError
from rh_agents.core.result_types import Tool_Result


class TestInput(BaseModel):
    value: str


async def valid_handler(input: TestInput, state: ExecutionState, context: str = "") -> Tool_Result:
    return Tool_Result(output=input.value, tool_name="test")


async def invalid_handler(input: TestInput) -> Tool_Result:  # Missing state parameter
    return Tool_Result(output=input.value, tool_name="test")


async def main():
    print("=== Validation Examples ===\n")
    
    # Example 1: Valid actor
    print("1. Validating valid actor...")
    valid_tool = Tool(
        name="TestTool",
        description="A test tool",
        input_model=TestInput,
        handler=valid_handler
    )
    try:
        validate_actor(valid_tool)
        print("   ✓ Valid actor passed validation\n")
    except ActorValidationError as e:
        print(f"   ✗ Validation failed: {e}\n")
    
    # Example 2: Invalid actor (missing name)
    print("2. Validating actor with empty name...")
    try:
        invalid_tool = Tool(
            name="",
            description="Test",
            input_model=TestInput,
            handler=valid_handler
        )
        validate_actor(invalid_tool)
        print("   ✗ Should have failed validation\n")
    except ActorValidationError as e:
        print(f"   ✓ Caught validation error:\n{e}\n")
    
    # Example 3: Invalid handler signature
    print("3. Validating actor with invalid handler...")
    try:
        bad_tool = Tool(
            name="BadTool",
            description="Test",
            input_model=TestInput,
            handler=invalid_handler
        )
        validate_actor(bad_tool)
        print("   ✗ Should have failed validation\n")
    except ActorValidationError as e:
        print(f"   ✓ Caught validation error:\n{e}\n")
    
    # Example 4: Valid state
    print("4. Validating execution state...")
    state = ExecutionState()
    try:
        validate_state(state)
        print("   ✓ Valid state passed validation\n")
    except StateValidationError as e:
        print(f"   ✗ Validation failed: {e}\n")
    
    print("=== Validation Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
```

#### 2.10: Add Builder Pattern for Complex Agents

**File:** `rh_agents/builders.py` (NEW FILE)

```python
"""
Builder pattern for constructing complex actors.

Provides fluent API for building agents with many optional parameters,
making construction more readable and maintainable.
"""
from typing import Any, Callable, Awaitable
from pydantic import BaseModel
from rh_agents.core.actors import Agent, LLM, Tool, ToolSet
from rh_agents.core.execution import ExecutionState


class AgentBuilder:
    """
    Fluent builder for Agent instances.
    
    Usage:
        agent = (
            AgentBuilder()
            .name("MyAgent")
            .description("Does cool stuff")
            .with_llm(my_llm)
            .with_tools([tool1, tool2])
            .cacheable(True)
            .build()
        )
    """
    
    def __init__(self):
        self._name: str | None = None
        self._description: str | None = None
        self._input_model: type[BaseModel] | None = None
        self._output_model: type[BaseModel] | None = None
        self._handler: Callable[[BaseModel, ExecutionState, str], Awaitable[Any]] | None = None
        self._tools: list[Tool] = []
        self._llm: LLM | None = None
        self._cacheable: bool = False
        self._version: str = "1.0.0"
        self._preconditions: list[Callable] = []
        self._postconditions: list[Callable] = []
    
    def name(self, name: str) -> 'AgentBuilder':
        """Set agent name."""
        self._name = name
        return self
    
    def description(self, description: str) -> 'AgentBuilder':
        """Set agent description."""
        self._description = description
        return self
    
    def input_model(self, model: type[BaseModel]) -> 'AgentBuilder':
        """Set input model."""
        self._input_model = model
        return self
    
    def output_model(self, model: type[BaseModel]) -> 'AgentBuilder':
        """Set output model."""
        self._output_model = model
        return self
    
    def handler(self, handler: Callable) -> 'AgentBuilder':
        """Set handler function."""
        self._handler = handler
        return self
    
    def with_llm(self, llm: LLM) -> 'AgentBuilder':
        """Attach an LLM to the agent."""
        self._llm = llm
        return self
    
    def with_tools(self, tools: list[Tool]) -> 'AgentBuilder':
        """Set tools available to the agent."""
        self._tools = tools
        return self
    
    def add_tool(self, tool: Tool) -> 'AgentBuilder':
        """Add a single tool to the agent."""
        self._tools.append(tool)
        return self
    
    def cacheable(self, cacheable: bool = True) -> 'AgentBuilder':
        """Set whether agent results should be cached."""
        self._cacheable = cacheable
        return self
    
    def version(self, version: str) -> 'AgentBuilder':
        """Set agent version."""
        self._version = version
        return self
    
    def add_precondition(self, condition: Callable) -> 'AgentBuilder':
        """Add a precondition check."""
        self._preconditions.append(condition)
        return self
    
    def add_postcondition(self, condition: Callable) -> 'AgentBuilder':
        """Add a postcondition check."""
        self._postconditions.append(condition)
        return self
    
    def build(self) -> Agent:
        """
        Build the Agent instance.
        
        Returns:
            Configured Agent
            
        Raises:
            ValueError: If required fields are missing
        """
        if not self._name:
            raise ValueError("Agent name is required")
        if not self._description:
            raise ValueError("Agent description is required")
        if not self._input_model:
            raise ValueError("Agent input_model is required")
        if not self._handler:
            raise ValueError("Agent handler is required")
        
        return Agent(
            name=self._name,
            description=self._description,
            input_model=self._input_model,
            output_model=self._output_model,
            handler=self._handler,
            tools=ToolSet(self._tools),
            llm=self._llm,
            cacheable=self._cacheable,
            version=self._version,
            preconditions=self._preconditions,
            postconditions=self._postconditions
        )


class ToolBuilder:
    """
    Fluent builder for Tool instances.
    
    Usage:
        tool = (
            ToolBuilder()
            .name("MyTool")
            .description("Does something")
            .input_model(MyInput)
            .handler(my_handler)
            .cacheable(True)
            .build()
        )
    """
    
    def __init__(self):
        self._name: str | None = None
        self._description: str | None = None
        self._input_model: type[BaseModel] | None = None
        self._output_model: type[BaseModel] | None = None
        self._handler: Callable | None = None
        self._cacheable: bool = False
        self._version: str = "1.0.0"
        self._cache_ttl: int | None = None
    
    def name(self, name: str) -> 'ToolBuilder':
        """Set tool name."""
        self._name = name
        return self
    
    def description(self, description: str) -> 'ToolBuilder':
        """Set tool description."""
        self._description = description
        return self
    
    def input_model(self, model: type[BaseModel]) -> 'ToolBuilder':
        """Set input model."""
        self._input_model = model
        return self
    
    def output_model(self, model: type[BaseModel]) -> 'ToolBuilder':
        """Set output model."""
        self._output_model = model
        return self
    
    def handler(self, handler: Callable) -> 'ToolBuilder':
        """Set handler function."""
        self._handler = handler
        return self
    
    def cacheable(self, cacheable: bool = True, ttl: int | None = None) -> 'ToolBuilder':
        """Set caching configuration."""
        self._cacheable = cacheable
        if ttl is not None:
            self._cache_ttl = ttl
        return self
    
    def version(self, version: str) -> 'ToolBuilder':
        """Set tool version."""
        self._version = version
        return self
    
    def build(self) -> Tool:
        """Build the Tool instance."""
        if not self._name:
            raise ValueError("Tool name is required")
        if not self._description:
            raise ValueError("Tool description is required")
        if not self._input_model:
            raise ValueError("Tool input_model is required")
        if not self._handler:
            raise ValueError("Tool handler is required")
        
        return Tool(
            name=self._name,
            description=self._description,
            input_model=self._input_model,
            output_model=self._output_model,
            handler=self._handler,
            cacheable=self._cacheable,
            version=self._version,
            cache_ttl=self._cache_ttl
        )
```

**Update `rh_agents/__init__.py`:**
```python
# Add to exports
from rh_agents.builders import AgentBuilder, ToolBuilder

__all__ = [
    # ... existing exports ...
    "AgentBuilder",
    "ToolBuilder",
]
```

**Create builder example:**

**File:** `examples/builder_example.py` (NEW)
```python
"""
Example of using builder pattern for constructing actors.
"""
import asyncio
from pydantic import BaseModel, Field
from rh_agents import AgentBuilder, ToolBuilder, LLM, ExecutionState, Message, AuthorType
from rh_agents.core.result_types import Tool_Result, LLM_Result
from rh_agents.core.types import EventType


class CalculatorInput(BaseModel):
    a: float
    b: float


class SearchInput(BaseModel):
    query: str


async def calculator_handler(input: CalculatorInput, state: ExecutionState, context: str = "") -> Tool_Result:
    result = input.a + input.b
    return Tool_Result(output=result, tool_name="calculator")


async def search_handler(input: SearchInput, state: ExecutionState, context: str = "") -> Tool_Result:
    return Tool_Result(output=f"Search results for: {input.query}", tool_name="search")


async def agent_handler(input: Message, state: ExecutionState, context: str = "") -> Message:
    return Message(content=f"Processed: {input.content}", author=AuthorType.ASSISTANT)


async def llm_handler(input: BaseModel, state: ExecutionState, context: str = "") -> LLM_Result:
    return LLM_Result(content="LLM response")


async def main():
    print("=== Builder Pattern Examples ===\n")
    
    # Build tools using fluent API
    calculator = (
        ToolBuilder()
        .name("calculator")
        .description("Performs addition")
        .input_model(CalculatorInput)
        .handler(calculator_handler)
        .cacheable(True, ttl=3600)
        .version("1.0.0")
        .build()
    )
    
    search_tool = (
        ToolBuilder()
        .name("search")
        .description("Searches for information")
        .input_model(SearchInput)
        .handler(search_handler)
        .build()
    )
    
    print(f"✓ Built tools: {calculator.name}, {search_tool.name}")
    
    # Build LLM
    llm = LLM(
        name="TestLLM",
        description="Test LLM",
        input_model=BaseModel,
        handler=llm_handler
    )
    
    # Build complex agent using builder
    agent = (
        AgentBuilder()
        .name("MathAgent")
        .description("An agent that handles math and search queries")
        .input_model(Message)
        .output_model(Message)
        .handler(agent_handler)
        .with_llm(llm)
        .with_tools([calculator, search_tool])
        .cacheable(False)
        .version("2.0.0")
        .build()
    )
    
    print(f"✓ Built agent: {agent.name}")
    print(f"  - Tools: {len(agent.tools.tools)}")
    print(f"  - LLM: {agent.llm.name if agent.llm else 'None'}")
    print(f"  - Cacheable: {agent.cacheable}")
    print(f"  - Version: {agent.version}")
    
    print("\n=== Builder Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### PHASE 3: Breaking Changes (v2.0.0)
**Estimated: 1-2 days**

#### 3.1: Remove Deprecated Cache Modules

**Files to delete:**
- `rh_agents/core/cache.py`
- `rh_agents/cache_backends.py`

**Files to update - remove imports:**
- `tests/test_caching.py` - Either delete or update to use state recovery
- Any other files importing from these modules

**Search and remove references:**
```bash
grep -r "from rh_agents.core.cache import" .
grep -r "from rh_agents.cache_backends import" .
grep -r "CacheBackend" .
grep -r "cache_backend" .
```

**Update ExecutionState if it has cache_backend references:**
```python
# File: rh_agents/core/execution.py
# Remove any remaining cache_backend logic
# Keep state_backend and artifact_backend (they're different!)
```

**IMPORTANT NOTE:**
The `cacheable` field in BaseActor should be KEPT. It's used by the state recovery system to determine if actor results should be cached/replayed, NOT related to the deprecated cache system.

#### 3.2: Update All Documentation

**Files to update:**
- `README.md` - Update examples to use new imports
- `examples/QUICK_START.md` - Update getting started guide
- `docs/ARCHITECTURE_DIAGRAMS.md` - Update architecture diagrams
- All doc files mentioning cache.py or cache_backends

**Remove references to:**
- Old cache system
- `CacheBackend` class
- Hash-based caching

**Emphasize:**
- State recovery system
- `StateBackend` and `ArtifactBackend`
- Event replay

#### 3.3: Create Migration Guide

**File:** `docs_refactory/MIGRATION_GUIDE_v2.md`

```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes

### 1. Removed Deprecated Cache System
**Removed:**
- `rh_agents.core.cache` module
- `rh_agents.cache_backends` module
- `CacheBackend` class
- `InMemoryCacheBackend` class
- `FileCacheBackend` class

**Migration:**
Use state recovery system instead:

```python
# OLD (deprecated)
from rh_agents.cache_backends import FileCacheBackend
cache = FileCacheBackend(".cache")
state = ExecutionState(cache_backend=cache)

# NEW (state recovery)
from rh_agents import FileSystemStateBackend, FileSystemArtifactBackend
state_backend = FileSystemStateBackend(".state_store")
artifact_backend = FileSystemArtifactBackend(".artifacts")
state = ExecutionState(
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Save checkpoint
await state.save_checkpoint()

# Restore
restored_state = ExecutionState.load_from_state_id(
    state_id,
    state_backend=state_backend
)
```

### 2. Generic Types Removed from Public API
**Changed:**
- `ExecutionEvent[OutputT]` → `ExecutionEvent`
- `ExecutionResult[T]` → `ExecutionResult`
- `LLM[T]` → `LLM`

**Migration:**
Simply remove type parameters:

```python
# OLD
event = ExecutionEvent[LLM_Result](actor=llm)
result: ExecutionResult[LLM_Result] = await event(...)

# NEW
event = ExecutionEvent(actor=llm)
result: ExecutionResult = await event(...)
# Type information still available via actor.output_model
```

**Note:** Type stub files (`.pyi`) still provide full generic types for static type checkers.

### 3. Context Parameter Now Optional
**Changed:**
Handler signature order and optionality

```python
# OLD
async def handler(input_data: Input, context: str, state: ExecutionState) -> Output:

# NEW
async def handler(input_data: Input, state: ExecutionState, context: str = "") -> Output:
```

**Migration:**
Update all custom handlers to:
1. Swap parameter order (state before context)
2. Make context optional with default ""
3. If you don't use context, you can omit it

### 4. Agent_Result Removed
**Removed:**
`rh_agents.core.result_types.Agent_Result`

**Reason:**
Was never used in codebase.

**Migration:**
If you were using it (unlikely), use domain-specific return types directly.

### 5. ToolSet API Simplified
**Changed:**
- Removed `get_tool_list()` method
- Internal implementation simplified

**Migration:**
```python
# OLD
tools = tool_set.get_tool_list()

# NEW
tools = list(tool_set)  # or tool_set.tools
```

## Non-Breaking Improvements

### 1. Public API Exports
You can now import from top-level:

```python
# NEW (recommended)
from rh_agents import Agent, Tool, LLM, ExecutionState, EventPrinter

# OLD (still works)
from rh_agents.core.actors import Agent, Tool, LLM
from rh_agents.core.execution import ExecutionState
from rh_agents.bus_handlers import EventPrinter
```

### 2. Improved Type Safety
With `.pyi` stub files, type checkers now have full generic type information while runtime stays simple.

### 3. Better Documentation
All public classes now have comprehensive docstrings.

## Automated Migration Script

```python
# migration_script.py
import re
from pathlib import Path

def migrate_file(file_path: Path):
    content = file_path.read_text()
    
    # Remove generic type parameters
    content = re.sub(r'ExecutionEvent\[.*?\]', 'ExecutionEvent', content)
    content = re.sub(r'ExecutionResult\[.*?\]', 'ExecutionResult', content)
    content = re.sub(r'LLM\[.*?\]', 'LLM', content)
    
    # Update imports
    content = content.replace(
        'from rh_agents.core.actors import',
        'from rh_agents import'
    )
    
    # Remove cache imports
    content = re.sub(r'from rh_agents\.cache_backends import.*\n', '', content)
    content = re.sub(r'from rh_agents\.core\.cache import.*\n', '', content)
    
    file_path.write_text(content)

# Run on your codebase
for py_file in Path('.').rglob('*.py'):
    migrate_file(py_file)
```
```

---

## Implementation Checklist

### Phase 1 (Non-Breaking)
- [ ] Create `rh_agents/__init__.py` with public exports
- [ ] Update all examples to use new imports
- [ ] Add comprehensive docstrings to core classes
- [ ] Run tests to ensure nothing broke
- [ ] Update README with new import style

### Phase 2 (Simplification)
- [ ] Remove Generic[T] from LLM class
- [ ] Remove Generic[OutputT] from ExecutionEvent
- [ ] Remove Generic[T] from ExecutionResult
- [ ] Update all usages (search/replace)
- [ ] Create `.pyi` stub files
- [ ] Clean up field redeclarations in Tool, LLM, Agent
- [ ] Simplify ToolSet implementation
- [ ] Update ExecutionState backend management to use exclude=True
- [ ] Create ActorOutput protocol
- [ ] Remove Agent_Result class
- [ ] Make context parameter optional in handler signature
- [ ] Update all handlers in agents.py
- [ ] Update all example handlers
- [ ] Create decorator API (decorators.py)
- [ ] Create validation helpers (validation.py)
- [ ] Create builder pattern (builders.py)
- [ ] Add new modules to __init__.py exports
- [ ] Create examples for decorators, validation, builders
- [ ] Run full test suite
- [ ] Update type stubs if needed

### Phase 3 (Breaking Changes)
- [ ] Delete `rh_agents/core/cache.py`
- [ ] Delete `rh_agents/cache_backends.py`
- [ ] Remove all cache-related imports
- [ ] Delete or update `tests/test_caching.py`
- [ ] Verify `cacheable` field is still present (it's for state recovery)
- [ ] Update all documentation
- [ ] Create migration guide
- [ ] Update CHANGELOG
- [ ] Run full test suite
- [ ] Manual testing of examples
- [ ] Version bump to 2.0.0

---

## Testing Strategy

### After Each Phase:
1. Run unit tests: `pytest tests/`
2. Run type checker: `mypy rh_agents/` (if using mypy)
3. Run all examples: `python examples/*.py`
4. Check for import errors
5. Verify backward compatibility (Phases 1-2 only)

### Final Testing (Phase 3):
1. Full test suite
2. All examples working
3. Documentation builds
4. Type stubs validate
5. Install test in clean venv
6. Migration script tested on sample project

---

## Validation Criteria

**Phase 1 Complete When:**
- All imports work from top-level `rh_agents` package
- All examples run without changes
- Documentation updated
- No test failures

**Phase 2 Complete When:**
- No Generic type parameters in public API
- All type stubs created and validated
- Field redeclarations removed
- ToolSet simplified
- Context parameter optional
- All tests pass
- Examples updated and working

**Phase 3 Complete When:**
- Deprecated cache modules deleted
- No references to old cache system
- Migration guide complete
- All tests pass
- All examples working
- Documentation fully updated
- Ready for release

---

## Notes for Implementation

### Critical Points:
1. **DO NOT remove `cacheable` field** - It's used by state recovery, not deprecated cache
2. **Preserve backward compat in Phases 1-2** - Only break in Phase 3
3. **Test after each major change** - Don't accumulate untested changes
4. **Update examples immediately** - They serve as integration tests
5. **Type stubs are important** - They maintain type safety without runtime overhead

### Order Matters:
- Phase 1 must complete before Phase 2 (establishes baseline)
- Phase 2 must complete before Phase 3 (validates new patterns)
- Within each phase, follow the numbered order

### When in Doubt:
- Check existing tests for usage patterns
- Refer to examples for real-world usage
- State recovery system docs explain `cacheable` field
- Protocol types should have `@runtime_checkable` decorator

---

**END OF IMPLEMENTATION SPEC**
