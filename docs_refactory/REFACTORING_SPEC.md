# RH-Agents Architecture Refactoring Specification

**Date:** January 24, 2026  
**Version:** 1.0.0  
**Status:** Draft for Review

---

## Executive Summary

This document presents a comprehensive analysis of the `rh_agents` package architecture, identifying areas of unnecessary complexity, boilerplate, and verbose design patterns. While the package is functionally working, there are opportunities to simplify the interface, reduce generics usage, and eliminate unused structures without compromising type safety or functionality.

**Key Goals:**
- Simplify API surface while maintaining functionality
- Reduce unnecessary generics and type complexity
- Remove unused boilerplate (e.g., super().__init__ calls with unused params)
- Make the package more Pythonic, testable, and easier to use
- Maintain zero linting issues and strong type safety where it matters

---

## 1. Current Package Structure

```
rh_agents/
├── __init__.py                 # Empty, no exports
├── agents.py                   # Agent implementations (DoctrineReceverAgent, StepExecutorAgent, etc.)
├── bus_handlers.py             # EventPrinter and SSE streaming handlers
├── cache_backends.py           # DEPRECATED - backward compatibility only
├── llms.py                     # OpenAILLM wrapper
├── models.py                   # Basic data models (Message, AuthorType, ArtifactRef)
├── openai.py                   # OpenAI integration
├── state_backends.py           # FileSystem state/artifact backends
└── core/
    ├── __init__.py
    ├── actors.py               # Base actor classes (BaseActor, Tool, LLM, Agent)
    ├── cache.py                # DEPRECATED - old caching system
    ├── context.py              # Orchestrator abstraction
    ├── events.py               # ExecutionEvent, ExecutionResult
    ├── execution.py            # ExecutionState, EventBus, HistorySet
    ├── parallel.py             # Parallel execution support
    ├── result_types.py         # LLM_Result, Tool_Result, Agent_Result
    ├── state_backend.py        # Abstract backend interfaces
    ├── state_recovery.py       # State snapshot models
    └── types.py                # Enums (EventType, ExecutionStatus)
```

---

## 2. Identified Issues

### 2.1 Generic Type Overuse

**Issue:** The package uses Generic types extensively, even where they provide minimal value.

**Examples:**

1. **`LLM` class with Generic[T]:**
   ```python
   class LLM(BaseActor, Generic[T]):
       handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]
   ```
   - `T` is always bound to `input_model` type
   - Used in only one place (handler signature)
   - Could be simplified to use `input_model` directly

2. **`ExecutionEvent` with Generic[OutputT]:**
   ```python
   class ExecutionEvent(BaseModel, Generic[OutputT]):
   ```
   - Only used for return type annotation
   - Creates complex type signatures like `ExecutionEvent[llm.output_model]`
   - Most users don't care about the generic parameter

3. **`ExecutionResult` with Generic[T]:**
   ```python
   class ExecutionResult(BaseModel, Generic[T]):
       result: Union[T, None]
   ```
   - Similar issue - only for type hints
   - Could use `Any` or structural typing

**Impact:**
- Makes code harder to read and understand
- Creates verbose instantiation: `ExecutionEvent[llm.output_model](actor=llm)`
- Intimidating for beginners
- Limited runtime value (Python's gradual typing)

### 2.2 Redundant BaseModel Field Declarations

**Issue:** Actor classes repeat field declarations unnecessarily.

**Example in `Tool` class:**
```python
class Tool(BaseActor):
    name: str                    # Already in BaseActor
    description: str             # Already in BaseActor
    input_model: type[BaseModel] # Already in BaseActor
    output_model: ...            # Already in BaseActor
    handler: AsyncHandler        # Already in BaseActor
    # ... and so on
```

**Why it's redundant:**
- Pydantic automatically inherits fields from parent classes
- Re-declaring doesn't change behavior
- Creates maintenance burden (duplicate updates)

**Counter-argument:**
- Explicit field lists improve IDE autocomplete
- Makes the class contract clear at a glance

### 2.3 Unused super().__init__() Calls

**Issue:** Several `model_post_init` methods call `super().model_post_init()` when parent has no implementation.

**Example:**
```python
def model_post_init(self, __context) -> None:
    if self.output_model is None:
        from rh_agents.core.result_types import Tool_Result
        self.output_model = Tool_Result
    super().model_post_init(__context) if hasattr(super(), 'model_post_init') else None
```

The conditional check is awkward and unnecessary if we know the parent class structure.

### 2.4 Complex Event System Architecture

**Issue:** The event system has multiple overlapping concerns.

**Current components:**
- `ExecutionEvent` - Wraps actor execution with __call__
- `EventBus` - Pub/sub for events
- `HistorySet` - Event storage with dict-based lookup
- `ExecutionState` - Orchestrates everything

**Complexity sources:**
1. **Event address computation** - String concatenation of execution stack
2. **Replay logic** - Three replay modes with complex skip logic
3. **Parallel execution** - Group IDs, indices, tracking
4. **Mixed concerns** - State, events, caching, replay all intertwined

**Better design:** Separate concerns into distinct modules

### 2.5 Result Type Hierarchy

**Issue:** Three separate result types with overlapping structure.

```python
class LLM_Result(BaseModel):
    content: str
    tools: list[LLM_Tool_Call]
    tokens_used: int | None
    # ...

class Tool_Result(BaseModel):
    output: Any
    tool_name: str
    success: bool

class Agent_Result(BaseModel):
    response: Any
    agent_name: str
    sub_tasks_completed: list[str]
```

**Problems:**
1. `Agent_Result` is defined but **never used** in the codebase
2. Inconsistent field names: `content` vs `output` vs `response`
3. Could use a unified result protocol

### 2.6 Deprecated Modules Still Present

**Issue:** `cache.py` and `cache_backends.py` are fully deprecated but still in the package.

**Why remove:**
- Confusing for new users
- Maintenance burden
- Import warnings clutter output

**Why keep:**
- Backward compatibility for existing code

### 2.7 ToolSet Class Complexity

**Issue:** Custom dict-like class with duplicate storage.

```python
class ToolSet(BaseModel):
    tools: list[Tool] = Field(default_factory=list)
    __tools: dict[str, Tool] = {}  # Duplicate storage
```

**Problems:**
- Maintains both list and dict
- Custom `__init__` to sync them
- Could just use `dict[str, Tool]` directly

**Justification:**
- Pydantic serialization expects list
- Dict lookup is O(1) vs O(n)

### 2.8 ExecutionState Backend Management

**Issue:** Awkward property-based backend storage.

```python
def __init__(self, state_backend: Optional["StateBackend"] = None, ...):
    super().__init__(**data)
    self._state_backend = state_backend  # Not a Pydantic field

@property
def state_backend(self) -> Optional["StateBackend"]:
    return getattr(self, '_state_backend', None)
```

**Why awkward:**
- Backends can't be serialized (non-Pydantic objects)
- Using private attributes and properties to work around Pydantic
- Confusing for users extending the class

### 2.9 Empty __init__.py Files

**Issue:** Package and subpackage `__init__.py` files are mostly empty.

**Current state:**
- `rh_agents/__init__.py` - Just a comment
- `rh_agents/core/__init__.py` - Empty

**Impact:**
- Users must import from deep paths: `from rh_agents.core.actors import Agent`
- No clear public API surface
- Harder to discover functionality

### 2.10 Mixed Async Handler Types

**Issue:** Handlers have inconsistent signatures and type hints.

**Examples:**
```python
# In BaseActor
handler: AsyncHandler  # Type alias, not specific

# In LLM (with Generic)
handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]

# In agents.py
async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Doctrine:
```

**Problems:**
- Type checkers struggle with `AsyncHandler` alias
- Generic signatures are complex
- No runtime validation of handler signature

---

## 3. Overall Solution Design

### 3.1 Principles

1. **Pythonic over Java-esque**: Prefer simple classes over complex hierarchies
2. **Explicit over implicit**: Clear public API in `__init__.py`
3. **Practical generics**: Use generics only where they provide clear value
4. **Separation of concerns**: Split monolithic classes into focused modules
5. **User-first API**: Design for the 90% use case, not the 10% edge case

### 3.2 Proposed Architecture Changes

#### Phase 1: Simplification (Non-Breaking)
1. Simplify generic usage in core classes
2. Clean up redundant field declarations
3. Improve `__init__.py` exports for better API surface
4. Document public vs internal APIs

#### Phase 2: Refactoring (Breaking Changes)
1. Redesign Result types into unified protocol
2. Simplify event system architecture
3. Refactor ToolSet to be simpler
4. Remove deprecated cache modules
5. Improve ExecutionState backend handling

#### Phase 3: Polish
1. Add comprehensive type stubs (`.pyi` files)
2. Improve docstrings with examples
3. Create migration guide from old API to new API

---

## 4. Specific Refactoring Recommendations

### 4.1 Simplify LLM Generic

**Current:**
```python
T = TypeVar('T', bound=Any)

class LLM(BaseActor, Generic[T]):
    handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]
```

**Proposed:**
```python
class LLM(BaseActor):
    # No generic needed - use input_model at runtime
    handler: Callable[[BaseModel, str, ExecutionState], Awaitable[LLM_Result]]
```

**Benefits:**
- Simpler instantiation: `LLM(...)` instead of `LLM[OpenAIRequest](...)`
- Still type-safe via `input_model` field
- Easier to understand

### 4.2 Simplify ExecutionEvent

**Current:**
```python
class ExecutionEvent(BaseModel, Generic[OutputT]):
    async def __call__(...) -> ExecutionResult[OutputT]:
```

**Proposed:**
```python
class ExecutionEvent(BaseModel):
    async def __call__(...) -> ExecutionResult:
        # OutputT inferred from actor.output_model
```

**Benefits:**
- No need to specify generic at instantiation
- Return type still available via `actor.output_model`
- Cleaner usage

### 4.3 Unify Result Types

**Current:** Three separate classes (`LLM_Result`, `Tool_Result`, `Agent_Result`)

**Proposed:** Single protocol-based result

```python
class ExecutionOutput(BaseModel):
    """Unified output from any actor execution"""
    data: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None

# Specialized results via metadata
def llm_result(content: str, tools: list = [], **meta) -> ExecutionOutput:
    return ExecutionOutput(
        data={"content": content, "tools": tools},
        metadata=meta
    )
```

**Benefits:**
- Single result interface
- Flexible metadata field
- Easier to extend
- Backward compatible via factory functions

### 4.4 Simplify ToolSet

**Current:** Custom class with dual storage

**Proposed Option A:** Just use dict
```python
ToolSet = dict[str, Tool]  # Type alias
```

**Proposed Option B:** Simpler wrapper
```python
class ToolSet:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}
    
    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]
    
    def __iter__(self):
        return iter(self._tools.values())
```

### 4.5 Clean Up Field Declarations

**Proposed:** Remove redundant redeclarations, use docstrings instead

```python
class Tool(BaseActor):
    """
    Represents an executable tool with defined input/output.
    
    Inherits: name, description, input_model, handler, output_model from BaseActor
    """
    # Only declare NEW fields or fields with DIFFERENT defaults
    cacheable: bool = False  # Override default
    event_type: EventType = EventType.TOOL_CALL  # Fixed value
```

### 4.6 Improve __init__.py Exports

**Proposed structure:**

```python
# rh_agents/__init__.py
"""
RH-Agents - Doctrine-Driven AI Actors Orchestration Framework
"""

# Core classes
from rh_agents.core.actors import Agent, Tool, LLM, ToolSet
from rh_agents.core.execution import ExecutionState, ExecutionEvent
from rh_agents.core.result_types import LLM_Result, Tool_Result

# State management
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend

# Utilities
from rh_agents.bus_handlers import EventPrinter
from rh_agents.models import Message, AuthorType

__all__ = [
    # Core
    'Agent', 'Tool', 'LLM', 'ToolSet',
    'ExecutionState', 'ExecutionEvent',
    # Results
    'LLM_Result', 'Tool_Result',
    # Backends
    'FileSystemStateBackend', 'FileSystemArtifactBackend',
    # Utilities
    'EventPrinter', 'Message', 'AuthorType',
]

__version__ = "1.0.0"
```

### 4.7 Remove Deprecated Modules

**Proposed:** Create `_deprecated` package for backward compat

```
rh_agents/
├── _deprecated/
│   ├── __init__.py  # Imports with warnings
│   ├── cache.py     # Moved here
│   └── cache_backends.py
```

Or remove entirely in next major version.

---

## 5. Design Decision Questions

### Q1: Generic Type Usage in Core Classes

**Question:** Should we remove or keep Generic types in `LLM`, `ExecutionEvent`, and `ExecutionResult`?

**Context:**
- Generics provide compile-time type safety in static type checkers
- They make instantiation more verbose: `ExecutionEvent[SomeType](actor=...)`
- Python's gradual typing means generics have limited runtime value
- Most users don't use mypy/pyright in strict mode

**Options:**

**A) Remove all generics (Pythonic approach)**
- Pros:
  - Simpler API, easier to learn
  - Less verbose instantiation
  - Still type-safe via `input_model`/`output_model` fields
  - Matches Python's duck-typing philosophy
- Cons:
  - Lose some static type checking
  - IDE autocomplete may be less precise
  - Advanced users might miss the type safety

**B) Keep generics (Type-safe approach)**
- Pros:
  - Full static type checking support
  - Catches type errors at development time
  - Better IDE support in strict mode
  - Shows intent clearly in signatures
- Cons:
  - More complex for beginners
  - Verbose instantiation
  - Overkill for dynamic Python code

**C) Optional generics (Hybrid approach)**
```python
class ExecutionEvent(BaseModel):  # No generic by default
    ...

# For advanced users who want type safety:
class TypedExecutionEvent(Generic[OutputT], ExecutionEvent):
    ...
```
- Pros:
  - Best of both worlds
  - Users choose complexity level
- Cons:
  - Maintains two parallel APIs
  - Still somewhat complex

**Suggestion:** **Option A (Remove all generics)**

**Rationale:**
- Package is already working without strict type checking
- Focus on usability over theoretical type safety
- Python community generally prefers simplicity
- Can add types back in `.pyi` stub files for those who want them
- Aligns with "Pythonic" goal

**Your Decision:** [Ok with option A, but with a correct .pyi, because get the correct ouput is crucial, case it makes better the development experience. It's terrible when we receive a value and when we try to get that properties nothing happens because they are "Any" or "Uknown"]

---

### Q2: Result Type Unification

**Question:** Should we unify `LLM_Result`, `Tool_Result`, `Agent_Result` into a single type?

**Context:**
- Three separate result types with similar structure
- `Agent_Result` is defined but never used
- Inconsistent naming (content vs output vs response)
- Each type has domain-specific fields

**Options:**

**A) Unified result class**
```python
class ActorOutput(BaseModel):
    data: Any
    metadata: dict[str, Any] = {}
    success: bool = True
```
- Pros:
  - Single interface for all actors
  - Easy to extend with metadata
  - Simpler type system
- Cons:
  - Lose domain-specific field names
  - Less self-documenting
  - Need to look in `data` to understand structure

**B) Keep separate types**
- Pros:
  - Clear, domain-specific fields
  - Self-documenting
  - Type-safe access (e.g., `result.content` not `result.data['content']`)
- Cons:
  - Three parallel types
  - Need to know which type for which actor
  - Harder to write generic code

**C) Protocol-based approach**
```python
from typing import Protocol

class ActorOutput(Protocol):
    success: bool
    error: str | None

class LLM_Result(BaseModel):
    content: str
    tools: list
    # Implements ActorOutput protocol
    success: bool = True
    error: str | None = None
```
- Pros:
  - Domain-specific types
  - Common protocol for generic code
  - Best type safety
- Cons:
  - More complex
  - Requires understanding of protocols

**D) Remove Agent_Result, keep the other two**
```python
# Only LLM_Result and Tool_Result
# Agents can return any type (Message, Doctrine, etc.)
```
- Pros:
  - Minimal change
  - Keep domain-specific types that are actually used
  - Remove dead code
- Cons:
  - Still have two result types
  - Doesn't fully unify

**Suggestion:** **Option D (Remove unused, keep LLM_Result and Tool_Result)**

**Rationale:**
- `LLM_Result` and `Tool_Result` are actively used and have clear distinct purposes
- LLMs have specific outputs (content, tool calls, tokens) that deserve their own type
- Tools have simpler outputs that fit `Tool_Result`
- Agents are high-level and can return domain types directly
- Minimal disruption, removes dead code

**Your Decision:** [Let's go for D and C. Let's keep LLM_Result and Tool_result using protocol to tie them together and keep the consistancy ]

---

### Q3: ToolSet Design

**Question:** How should we handle the collection of tools?

**Context:**
- Current `ToolSet` maintains both list and dict for dual access patterns
- Adds complexity with custom `__init__` and `__getitem__`
- Used extensively in LLM function calling

**Options:**

**A) Plain dict**
```python
tools: dict[str, Tool]
```
- Pros:
  - Simplest possible
  - Built-in Python type
  - Fast O(1) lookup
- Cons:
  - Lose order (though dict is ordered in Python 3.7+)
  - Less discoverable (iteration gives keys, not values)
  - Serialization needs custom handling

**B) Simple wrapper class**
```python
class ToolSet:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}
    
    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]
    
    def __iter__(self):
        yield from self._tools.values()
```
- Pros:
  - Clean interface
  - Easy to iterate over tools
  - Name-based lookup
- Cons:
  - Not a Pydantic model (serialization issues)
  - Extra class to maintain

**C) Keep current ToolSet but simplify**
```python
class ToolSet(BaseModel):
    tools: list[Tool]
    
    @computed_field
    @property
    def by_name(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}
    
    def get(self, name: str) -> Tool | None:
        return self.by_name.get(name)
```
- Pros:
  - Pydantic model (serializable)
  - Primary storage is list (order preserved)
  - Dict computed on demand
- Cons:
  - Recomputes dict on each access (could cache)

**D) Just use list[Tool]**
```python
tools: list[Tool]

# Lookup via iteration
def get_tool(tools: list[Tool], name: str) -> Tool | None:
    return next((t for t in tools if t.name == name), None)
```
- Pros:
  - Simplest type
  - Works with Pydantic out of box
  - Sufficient for small tool lists
- Cons:
  - O(n) lookup
  - No built-in lookup method

**Suggestion:** **Option C (Simplified ToolSet with computed dict)**

**Rationale:**
- Maintains serialization compatibility
- Primary storage is list (natural, ordered)
- Dict lookup available when needed
- Can optimize with `@lru_cache` if performance matters
- Balances simplicity and functionality

**Your Decision:** [ Let's go for C, as you suggest]

---

### Q4: Deprecated Module Handling

**Question:** What should we do with deprecated `cache.py` and `cache_backends.py` modules?

**Context:**
- Modules are fully deprecated in favor of state recovery system
- Currently emit DeprecationWarning on import
- Still used by some existing code

**Options:**

**A) Remove completely**
- Pros:
  - Clean codebase
  - No confusion for new users
  - Forces migration to new system
- Cons:
  - Breaking change
  - Breaks existing user code
  - Need major version bump

**B) Move to _deprecated/ package**
```
rh_agents/_deprecated/
    cache.py
    cache_backends.py
```
- Pros:
  - Signals deprecation clearly
  - Maintains backward compat
  - Users can migrate gradually
- Cons:
  - Still need to maintain
  - Import paths change (also breaking)

**C) Keep with louder warnings**
```python
warnings.warn(
    "This module will be removed in v2.0.0. "
    "Use rh_agents.state_backends instead.",
    DeprecationWarning,
    stacklevel=2
)
```
- Pros:
  - No breaking changes
  - Clear deprecation path
  - Give users time to migrate
- Cons:
  - Clutters codebase
  - Warning fatigue

**D) Keep but hide from documentation**
- Pros:
  - Backward compat maintained
  - New users won't see it in docs
- Cons:
  - Still in codebase
  - Can still be discovered

**Suggestion:** **Option C (Keep with louder warnings)**

**Rationale:**
- This is interface-only refactoring, should avoid breaking changes
- Give users at least one version cycle to migrate
- Document migration path clearly
- Remove in v2.0.0

**Your Decision:** [Option A. In addition, check if we can remove the cached flag for actors, just checkout if it's related to another functionality and its name is missleading ]

**Investigation Result:** The `cacheable` field in BaseActor is **NOT** related to the deprecated cache system. It's used by the **state recovery system** to determine if an actor's results should be cached/replayed. This field should be **KEPT**.

---

### Q5: ExecutionState Backend Management

**Question:** How should ExecutionState handle optional backends (state_backend, artifact_backend)?

**Context:**
- Backends can't be Pydantic fields (not serializable)
- Current approach uses private attrs + properties
- Awkward for users and internal code

**Options:**

**A) Keep current approach (private attrs + properties)**
```python
def __init__(self, state_backend: Optional[StateBackend] = None, ...):
    super().__init__(**data)
    self._state_backend = state_backend

@property
def state_backend(self) -> Optional[StateBackend]:
    return getattr(self, '_state_backend', None)
```
- Pros:
  - Already implemented
  - Works with Pydantic serialization
- Cons:
  - Awkward and non-Pythonic
  - Confusing pattern

**B) Use Pydantic exclude**
```python
class ExecutionState(BaseModel):
    state_backend: Optional[StateBackend] = Field(default=None, exclude=True)
    artifact_backend: Optional[ArtifactBackend] = Field(default=None, exclude=True)
```
- Pros:
  - Clean field declarations
  - Pydantic handles exclusion
  - More idiomatic
- Cons:
  - Need `arbitrary_types_allowed = True`
  - Fields still in model

**C) Separate config object**
```python
class ExecutionConfig:
    state_backend: Optional[StateBackend] = None
    artifact_backend: Optional[ArtifactBackend] = None

class ExecutionState(BaseModel):
    # Only serializable fields
    storage: ExecutionStore
    history: HistorySet
    # ...
    
    def __init__(self, config: ExecutionConfig = None, **data):
        super().__init__(**data)
        self._config = config or ExecutionConfig()
```
- Pros:
  - Clear separation of serializable vs runtime state
  - Config object is reusable
  - Clean ExecutionState model
- Cons:
  - Extra class
  - Indirection (`state._config.state_backend`)

**D) Dataclass for ExecutionState**
```python
@dataclass
class ExecutionState:
    storage: ExecutionStore = field(default_factory=ExecutionStore)
    state_backend: Optional[StateBackend] = None
    # No serialization concerns, just plain Python
```
- Pros:
  - Simpler than Pydantic for internal state
  - No serialization constraints
  - More Pythonic
- Cons:
  - Lose Pydantic validation
  - Need custom serialization logic
  - Big change from current architecture

**Suggestion:** **Option B (Pydantic with exclude=True)**

**Rationale:**
- Minimal change from current code
- More idiomatic than private attrs
- Pydantic already allows excluding fields
- Keep validation benefits
- Backends are truly optional

**Your Decision:** [ Option B]

---

### Q6: Field Redeclaration in Subclasses

**Question:** Should subclasses redeclare inherited fields for clarity?

**Context:**
- Current code redeclares fields like `name`, `description` in `Tool`, `LLM`, `Agent`
- Pydantic inherits fields automatically
- Redeclaration doesn't change behavior

**Options:**

**A) Remove all redeclarations**
```python
class Tool(BaseActor):
    """
    Inherits: name, description, input_model, handler, etc.
    """
    # Only declare NEW or OVERRIDDEN fields
    cacheable: bool = False
```
- Pros:
  - DRY principle
  - Less code to maintain
  - Clear which fields are new/changed
- Cons:
  - Fields not visible at class level
  - IDE autocomplete might be worse
  - Less self-documenting

**B) Keep redeclarations**
```python
class Tool(BaseActor):
    name: str
    description: str
    # ... all inherited fields
```
- Pros:
  - Self-documenting
  - Better IDE support
  - See all fields at a glance
- Cons:
  - Verbose
  - Maintenance burden
  - Violates DRY

**C) Use ellipsis for inherited fields**
```python
class Tool(BaseActor):
    # Inherited from BaseActor
    name: str = ...
    description: str = ...
    # New fields
    cacheable: bool = False
```
- Pros:
  - Shows inherited fields clearly
  - Signals "not redefined here"
  - IDE can still see fields
- Cons:
  - Unusual Python pattern
  - Still verbose

**D) Use TYPE_CHECKING blocks**
```python
if TYPE_CHECKING:
    class Tool(BaseActor):
        name: str
        description: str
        # ... for type checkers only

class Tool(BaseActor):
    # Only runtime code
    cacheable: bool = False
```
- Pros:
  - Best type checking
  - Minimal runtime code
- Cons:
  - Complex pattern
  - Duplicate declarations

**Suggestion:** **Option A (Remove redeclarations, use docstrings)**

**Rationale:**
- Trust Pydantic's inheritance
- Docstrings can list inherited fields
- Keeps classes focused on what they add
- Modern IDEs can navigate to parent class
- More Pythonic (trust the framework)

**Your Decision:** [ Option A]

---

### Q7: Empty __init__.py Exports

**Question:** Should we populate `__init__.py` files to create a public API?

**Context:**
- Current imports require deep paths: `from rh_agents.core.actors import Agent`
- Unclear what is "public" vs "internal"
- No version number exposed

**Options:**

**A) Flat public API in rh_agents/__init__.py**
```python
# rh_agents/__init__.py
from rh_agents.core.actors import Agent, Tool, LLM
from rh_agents.core.execution import ExecutionState
# ... etc
```
- Pros:
  - Simple imports: `from rh_agents import Agent`
  - Clear public API
  - Standard Python pattern
- Cons:
  - Need to decide what's public
  - Circular import risk

**B) Namespace packages (keep deep imports)**
```python
# Keep as is
from rh_agents.core.actors import Agent
```
- Pros:
  - No import magic
  - Clear organization
  - Explicit naming
- Cons:
  - Verbose imports
  - Harder for beginners
  - No clear API boundary

**C) Hybrid approach**
```python
# rh_agents/__init__.py - common items
from rh_agents.core.actors import Agent, Tool, LLM

# rh_agents/core/__init__.py - advanced items
from rh_agents.core.actors import BaseActor
from rh_agents.core.events import ExecutionEvent
```
- Pros:
  - Common stuff easy to import
  - Advanced stuff in submodules
  - Flexible
- Cons:
  - Two APIs to maintain
  - Still somewhat complex

**Suggestion:** **Option A (Flat public API)**

**Rationale:**
- Makes package more approachable
- Standard practice in Python ecosystem
- Can mark internal modules with `_` prefix
- Improves discoverability
- Aligns with "easy to use" goal

**Your Decision:** [ Option A]

---

### Q8: Async Handler Type Signatures

**Question:** How should we type async handler functions?

**Context:**
- Currently use `AsyncHandler` type alias
- Also use full `Callable[..., Coroutine[...]]` signatures
- Generic signatures are complex
- No runtime validation

**Options:**

**A) Simple Protocol**
```python
from typing import Protocol

class ActorHandler(Protocol):
    async def __call__(
        self,
        input_data: BaseModel,
        context: str,
        state: ExecutionState
    ) -> Any:
        ...
```
- Pros:
  - Clear contract
  - Type-safe
  - Self-documenting
- Cons:
  - Protocols are somewhat advanced
  - Extra class definition

**B) Type alias with comments**
```python
AsyncHandler = Callable[[BaseModel, str, ExecutionState], Awaitable[Any]]
# Expected signature: async def handler(input_data, context, state) -> result
```
- Pros:
  - Simple
  - Works with current code
  - Comment explains usage
- Cons:
  - Less precise
  - No protocol checking

**C) Generic Callable (current approach)**
```python
handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, R]]
```
- Pros:
  - Precise types
  - Full generic support
- Cons:
  - Complex
  - Hard to read
  - Overkill

**D) Just use Any**
```python
handler: Callable[..., Awaitable[Any]]
# Trust developers to use correct signature
```
- Pros:
  - Simplest
  - Flexible
  - Pythonic duck-typing
- Cons:
  - No type checking
  - Mistakes caught at runtime

**Suggestion:** **Option B (Type alias with clear docs)**

**Rationale:**
- Balances simplicity and safety
- Most type checkers handle Awaitable well
- Docstrings/examples show usage
- Can always add runtime validation if needed
- Matches Python's gradual typing philosophy

**Your Decision:** [ Let's got for B ]

---

### Q9: Context String Parameter

**Question:** Should we keep the `context: str` parameter passed to all handlers?

**Context:**
- Every handler receives `context: str` parameter
- Currently used for passing execution context as string
- Often not used by handlers
- Unclear purpose in many cases

**Options:**

**A) Remove it entirely**
```python
async def handler(input_data: InputType, state: ExecutionState) -> Output:
    # Access context via state if needed
    context = state.get_context_string()
```
- Pros:
  - Simpler signatures
  - One source of truth (ExecutionState)
  - Less parameter passing
- Cons:
  - Breaking change
  - Existing handlers need updates

**B) Keep it but make optional**
```python
async def handler(
    input_data: InputType,
    state: ExecutionState,
    context: str = ""
) -> Output:
```
- Pros:
  - Backward compatible
  - Can be ignored if not needed
- Cons:
  - Still in signature
  - Confusing optional parameter

**C) Move to state only**
```python
state.context_string  # Access via state
```
- Pros:
  - Available when needed
  - Not cluttering signatures
- Cons:
  - Need to add to ExecutionState
  - Breaking change

**D) Keep as is**
- Pros:
  - No breaking changes
  - Explicit context passing
- Cons:
  - Extra parameter burden
  - Often unused

**Suggestion:** **Option D (Keep as is for now)**

**Rationale:**
- Interface-only refactoring should minimize breaking changes
- Context is sometimes useful (e.g., for LLM prompts)
- Can deprecate in future version
- Document proper usage

**Your Decision:** [ Let's go for B. Keep it, but optionally]

---

### Q10: Event Address System

**Question:** Should we simplify or keep the current event address system?

**Context:**
- Addresses built from execution stack: `"agent::tool_call::llm_call"`
- Used for replay, caching, event tracking
- Complex string manipulation and concatenation
- Relies on execution stack in ExecutionState

**Options:**

**A) Keep current system**
```python
address = "::".join(execution_stack) + "::" + event_type
```
- Pros:
  - Already works
  - Hierarchical structure
  - Enables selective replay
- Cons:
  - String manipulation
  - Fragile (depends on order)

**B) Use structured address**
```python
@dataclass
class EventAddress:
    path: list[str]
    event_type: EventType
    
    def __str__(self) -> str:
        return "::".join(self.path + [self.event_type.value])
```
- Pros:
  - Type-safe
  - Easier to manipulate
  - Clear structure
- Cons:
  - More complex
  - Need conversion to/from string

**C) UUID-based addresses**
```python
address = str(uuid.uuid4())
# Store parent-child relationships separately
```
- Pros:
  - No collisions
  - Simple generation
  - Unique across runs
- Cons:
  - Lose hierarchical info in address
  - Can't do prefix matching for replay
  - Need separate structure for relationships

**D) Hybrid: UUID + path metadata**
```python
address = str(uuid.uuid4())
metadata = {
    "path": ["agent", "tool_call"],
    "event_type": "llm_call"
}
```
- Pros:
  - Best of both worlds
  - Unique IDs
  - Hierarchical info preserved
- Cons:
  - More complex event model
  - Two sources of truth

**Suggestion:** **Option A (Keep current system)**

**Rationale:**
- String addresses work well for replay
- Hierarchical structure is useful
- Simple to understand and debug
- No compelling reason to change
- Can optimize implementation without changing API

**Your Decision:** [Let's keep it as option A ]

---

## 6. Additional Suggestions

### S1: Type Stub Files (.pyi)

**Suggestion:** Add `.pyi` stub files for complex generics

Even if we remove runtime generics, we can provide stub files for type checkers:

```python
# rh_agents/core/actors.pyi
from typing import TypeVar, Generic

T = TypeVar('T')
R = TypeVar('R')

class LLM(BaseActor, Generic[T, R]):
    def __call__(self, input: T) -> Awaitable[R]: ...
```

**Benefits:**
- Best static type checking for those who want it
- Runtime simplicity for everyone else
- Doesn't affect runtime behavior

---

### S2: Decorator-Based Actor Definition

**Suggestion:** Add decorator syntax for simple actors

```python
@rh_agents.tool(name="calculator")
async def calculate(input: CalculatorArgs, state: ExecutionState) -> ToolResult:
    return ToolResult(output=input.a + input.b, tool_name="calculator")

# Equivalent to:
calculator = Tool(
    name="calculator",
    input_model=CalculatorArgs,
    handler=calculate,
    ...
)
```

**Benefits:**
- More Pythonic
- Less boilerplate
- Familiar pattern (FastAPI, Flask)

---

### S3: Builder Pattern for Complex Agents

**Suggestion:** Add fluent builder API

```python
agent = (
    Agent.builder()
    .name("MyAgent")
    .with_llm(llm)
    .with_tools([tool1, tool2])
    .cacheable(True)
    .build()
)
```

**Benefits:**
- Clearer construction
- Optional parameters obvious
- Chainable

---

### S4: Validation Helpers

**Suggestion:** Add validation utilities

```python
from rh_agents.validation import validate_actor, validate_state

# Check actor configuration
validate_actor(my_agent)  # Raises ValidationError if misconfigured

# Check state consistency
validate_state(execution_state)  # Checks history, storage, etc.
```

**Benefits:**
- Catch errors early
- Better error messages
- Help debugging

---


## 7. Implementation Plan

### Phase 1: Non-Breaking Improvements (v1.1.0)
**Timeline:** 1-2 weeks

1. Improve `__init__.py` exports
2. Add comprehensive docstrings
3. Create `.pyi` stub files (optional)
4. Add validation helpers
5. Enhance deprecation warnings
6. Update examples and documentation

**Risk:** Low - No breaking changes

### Phase 2: Simplification (v1.5.0)
**Timeline:** 2-3 weeks

1. Simplify generic usage (with backward compat)
2. Clean up field redeclarations
3. Simplify ToolSet
4. Improve ExecutionState backend handling
5. Remove unused Agent_Result
6. Add decorator syntax (optional)

**Risk:** Medium - Some API changes, but backward compatible

### Phase 3: Major Refactoring (v2.0.0)
**Timeline:** 3-4 weeks

1. Remove deprecated cache modules
2. Unify result types (if decided)
3. Remove context parameter (if decided)
4. Any other breaking changes approved
5. Migration guide
6. Updated examples

**Risk:** High - Breaking changes, but that's what major versions are for

---

## 8. Migration Guide Template

For each breaking change, we'll provide:

### Example: Removing Generics from ExecutionEvent

**Old Code:**
```python
event = ExecutionEvent[LLM_Result](actor=llm)
result = await event(input_data, context, state)
```

**New Code:**
```python
event = ExecutionEvent(actor=llm)
result = await event(input_data, context, state)
# Type still available via event.actor.output_model
```

**Migration:** Remove generic type parameter from instantiation.

**Automated:** Yes - regex find/replace

---

## 9. Success Metrics

How we'll measure success:

1. **API Simplicity**
   - Lines of code in minimal example reduced by 20%
   - Fewer imports needed for common tasks

2. **Type Safety**
   - Zero mypy errors in strict mode
   - All public APIs type-annotated

3. **Documentation**
   - 100% of public APIs documented
   - At least 10 complete examples

4. **Performance**
   - No regression in benchmarks
   - Faster instantiation if possible

5. **User Feedback**
   - Survey existing users
   - "Ease of use" rating improvement

---

## 10. Questions for Package Author

Please review and provide decisions on:

1. ✅ Generic type usage (Q1)
2. ✅ Result type unification (Q2)
3. ✅ ToolSet design (Q3)
4. ✅ Deprecated module handling (Q4)
5. ✅ Backend management (Q5)
6. ✅ Field redeclaration (Q6)
7. ✅ __init__.py exports (Q7)
8. ✅ Async handler types (Q8)
9. ✅ Context parameter (Q9)
10. ✅ Event address system (Q10)

Additional feedback:
- Are there other pain points users have mentioned?
- Any specific use cases we should prioritize?
- Timeline constraints?
- Backward compatibility requirements?

---

## Appendix A: Code Statistics

**Current Package:**
- Total lines: ~4,500
- Core modules: 12 files
- Public classes: ~25
- Generic classes: 3
- Deprecated modules: 2

**Estimated After Refactoring:**
- Total lines: ~3,800 (15% reduction)
- Deprecated modules: 0
- Generic classes: 0-1 (based on decisions)
- Cleaner, more focused code

---

## Appendix B: Related Documentation

- [docs/ARCHITECTURE_DIAGRAMS.md](../docs/ARCHITECTURE_DIAGRAMS.md)
- [docs/STATE_RECOVERY_SPEC.md](../docs/STATE_RECOVERY_SPEC.md)
- [docs/PARALLEL_EXECUTION_SPEC.md](../docs/parallel/PARALLEL_EXECUTION_SPEC.md)
- [examples/QUICK_START.md](../examples/QUICK_START.md)

---

**End of Refactoring Specification**
