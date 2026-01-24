# 🤖 RH-Agents

**A powerful Python framework for building stateless, resumable multi-agent AI workflows with built-in state recovery.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Soon, more docs...

---

## 🌟 Overview

RH-Agents is a doctrine-driven orchestration framework that enables you to build complex AI pipelines with:

- **🔄 State Recovery**: Save and resume execution from any point
- **📦 Artifact Storage**: Content-addressable storage for large objects
- **⚡ Smart Replay**: Automatically skip completed steps during resume
- **🎯 Selective Resume**: Resume from specific addresses in your pipeline
- **🔗 Multi-Agent Coordination**: Orchestrate LLMs, tools, and agents seamlessly
- **📡 Event Streaming**: Real-time execution monitoring via event bus
- **🧪 Validation Mode**: Compare replay results with historical execution
- **🎨 Flexible Actor Creation**: Decorators, builders, or classes - choose your style

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Creating Actors](#-creating-actors)
  - [Decorator API](#decorator-api)
  - [Builder Pattern](#builder-pattern)
  - [Class-Based Approach](#class-based-approach)
- [Validation](#-validation)
- [Core Concepts](#-core-concepts)
- [State Recovery](#-state-recovery)
- [Examples](#-examples)
- [API Reference](#-api-reference)
- [Architecture](#-architecture)

---

## ✨ Features

### Modern Actor Creation
- **Decorator API**: FastAPI-style decorators (`@tool_decorator`, `@agent_decorator`) for quick development
- **Builder Pattern**: Fluent API for complex configurations with method chaining
- **Class-Based**: Traditional OOP approach with full control
- **Validation Helpers**: Built-in validation to catch errors early (`validate_actor`, `validate_state`)

### State Recovery System
- **Checkpoint & Resume**: Save execution state at any point and resume later
- **Smart Replay**: Automatically skip completed events, only re-execute what's needed
- **Resume from Address**: Jump to any specific step in your pipeline
- **Validation Mode**: Verify replay consistency by comparing with historical results

### Multi-Agent Orchestration
- **Agent Pipeline**: Chain multiple AI agents with dependency management
- **Tool Calling**: LLM-powered tool selection and execution
- **Context Sharing**: Agents share results through ExecutionState
- **Event-Driven**: Subscribe to execution events for monitoring and logging

### Storage & Caching
- **Artifact System**: Store large objects separately with content-addressable IDs
- **File-Based Backends**: JSON for states, pickle for artifacts
- **Deduplication**: SHA256-based artifact identification

### Developer Experience
- **Type-Safe**: Full Pydantic v2 integration with type hints
- **Event Printer**: Beautiful console output with execution tree visualization
- **Async-First**: Built on asyncio for high performance
- **Extensible**: Easy to create custom agents, tools, and LLMs

---

## 📦 Installation

```bash
pip install rh-agents
```

**Requirements:**
- Python 3.12+
- pydantic 2.12.5+
- jsonpatch 1.33+

**Note:** This package is in **beta**. APIs may change between releases as we iterate on design and gather feedback.

**Optional:**
- openai (for OpenAI LLM integration)
- fastapi (for streaming API examples)

### Migrating from Earlier Versions

If you're upgrading from a previous version:

- **Deprecated cache system removed**: Use `FileSystemStateBackend` instead of `FileCacheBackend`
- **New APIs available**: Decorators, builders, and validation helpers are now available
- **Breaking changes expected**: As a beta package, we prioritize API improvements over stability

See `/docs_refactory/MIGRATION_GUIDE_v2.md` for detailed migration steps.

---

## 🚀 Quick Start

### Basic Agent Execution

```python
import asyncio
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Agent
from rh_agents.models import Message

# Define a simple agent
class GreetingAgent(Agent):
    async def handler(self, input_data: Message, context: str, execution_state: ExecutionState) -> Message:
        return Message(
            content=f"Hello, {input_data.content}!",
            role="assistant"
        )

# Create execution state
async def main():
    bus = EventBus()
    state = ExecutionState(event_bus=bus)
    
    # Create and execute agent
    agent = GreetingAgent(
        name="Greeter",
        description="A friendly greeting agent"
    )
    
    message = Message(content="World", role="user")
    result = await ExecutionEvent[Message](actor=agent)(message, "", state)
    
    print(result.result.content)  # Output: Hello, World!

asyncio.run(main())
```

---

## 🎨 Creating Actors

RH-Agents offers **three ways** to create actors, from simple to complex:

### Decorator API

The fastest way to create tools and agents with a clean, Pythonic syntax:

#### Creating Tools with `@tool_decorator`

```python
from rh_agents import tool_decorator, Tool_Result, ExecutionState
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")

@tool_decorator(
    name="calculator",
    description="Performs basic arithmetic operations",
    cacheable=True,
    version="1.0.0"
)
async def calculator(
    input: CalculatorInput,
    context: str,
    state: ExecutionState
) -> Tool_Result:
    """A simple calculator tool."""
    result = {
        "add": input.a + input.b,
        "subtract": input.a - input.b,
        "multiply": input.a * input.b,
        "divide": input.a / input.b if input.b != 0 else None
    }.get(input.operation)
    
    return Tool_Result(
        output=result,
        tool_name="calculator"
    )

# calculator is now a Tool instance ready to use
```

#### Creating Agents with `@agent_decorator`

```python
from rh_agents import agent_decorator, Agent, Message
from rh_agents.openai import OpenAILLM

my_llm = OpenAILLM(model="gpt-4")
tools = [calculator]  # From above

@agent_decorator(
    name="MathAgent",
    description="Solves math problems using tools",
    tools=tools,
    llm=my_llm,
    cacheable=False
)
async def math_handler(
    input: Message,
    context: str,
    state: ExecutionState
) -> Message:
    """Handle math-related queries."""
    # Your agent logic here
    return Message(
        content=f"Processed: {input.content}",
        role="assistant"
    )

# math_handler is now an Agent instance
```

**Benefits:**
- ✅ Clean, minimal syntax
- ✅ Type inference from function signature
- ✅ Decorator parameters are optional (uses defaults)
- ✅ Similar to FastAPI/Flask decorators

---

### Builder Pattern

For complex agents with many configuration options:

```python
from rh_agents import AgentBuilder, ToolBuilder, Message, ExecutionState

# Build a tool using the builder
search_tool = (
    ToolBuilder()
    .name("web_search")
    .description("Searches the web for information")
    .input_model(SearchInput)
    .handler(search_handler)
    .cacheable(True)
    .version("2.0.0")
    .build()
)

# Build an agent with fluent API
agent = (
    AgentBuilder()
    .name("ResearchAgent")
    .description("Conducts research using web search")
    .input_model(Message)
    .output_model(Message)
    .handler(research_handler)
    .with_llm(my_llm)
    .with_tools([search_tool, calculator])
    .cacheable(True)
    .version("1.5.0")
    .add_precondition(lambda state: state.storage is not None)
    .add_postcondition(lambda result: len(result.content) > 0)
    .build()
)
```

**Builder Methods:**

**AgentBuilder:**
- `.name(str)` - Set agent name
- `.description(str)` - Set description
- `.input_model(type)` - Set input Pydantic model
- `.output_model(type)` - Set output Pydantic model
- `.handler(callable)` - Set async handler function
- `.with_llm(LLM)` - Attach LLM instance
- `.with_tools(list[Tool])` - Set available tools
- `.add_tool(Tool)` - Add a single tool
- `.cacheable(bool)` - Enable/disable caching
- `.version(str)` - Set version string
- `.add_precondition(callable)` - Add validation before execution
- `.add_postcondition(callable)` - Add validation after execution
- `.build()` - Build the Agent instance

**ToolBuilder:**
- `.name(str)` - Set tool name
- `.description(str)` - Set description
- `.input_model(type)` - Set input Pydantic model
- `.output_model(type)` - Set output Pydantic model
- `.handler(callable)` - Set async handler function
- `.cacheable(bool)` - Enable/disable caching
- `.version(str)` - Set version string
- `.build()` - Build the Tool instance

**Benefits:**
- ✅ Readable for complex configurations
- ✅ Method chaining for fluent API
- ✅ Clear separation of concerns
- ✅ Optional preconditions/postconditions
- ✅ Build-time validation

---

### Class-Based Approach

Traditional approach with full control and type safety:

```python
from rh_agents import Tool, Agent, Message, ExecutionState, Tool_Result
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str
    max_results: int = 10

class SearchTool(Tool):
    def __init__(self):
        async def handler(
            args: SearchInput,
            context: str,
            execution_state: ExecutionState
        ) -> Tool_Result:
            # Your search logic
            results = perform_search(args.query, args.max_results)
            return Tool_Result(
                output=results,
                tool_name="search"
            )
        
        super().__init__(
            name="search",
            description="Search the web",
            input_model=SearchInput,
            handler=handler,
            cacheable=True,
            version="1.0.0"
        )

class ResearchAgent(Agent):
    def __init__(self, llm, tools):
        async def handler(
            input_data: Message,
            context: str,
            execution_state: ExecutionState
        ) -> Message:
            # Your agent logic
            return Message(content="Research complete", role="assistant")
        
        super().__init__(
            name="ResearchAgent",
            description="Conducts research",
            input_model=Message,
            output_model=Message,
            handler=handler,
            tools=tools,
            llm=llm
        )
```

**Benefits:**
- ✅ Full type safety
- ✅ IDE autocomplete support
- ✅ Familiar OOP patterns
- ✅ Easy to test and extend

---

### Choosing an Approach

| Approach | Best For | Complexity |
|----------|----------|------------|
| **Decorator API** | Quick prototypes, simple tools/agents | ⭐ Low |
| **Builder Pattern** | Complex agents with many options | ⭐⭐ Medium |
| **Class-Based** | Production code, reusable components | ⭐⭐⭐ High |

**Recommendation:** Start with decorators, graduate to builders for complex configs, use classes for production systems.

---

## ✅ Validation

RH-Agents provides built-in validation helpers to catch configuration errors early:

### Validate Actors

```python
from rh_agents import validate_actor, ActorValidationError

try:
    # Validate your actor configuration
    validate_actor(my_agent)
    print("✅ Agent is valid!")
except ActorValidationError as e:
    print(f"❌ Validation failed: {e}")
```

**Checks performed:**
- ✅ Name is not empty
- ✅ Description is provided
- ✅ Input model is a valid Pydantic BaseModel
- ✅ Handler is an async function
- ✅ Handler signature has correct parameters
- ✅ Version format is valid (if provided)
- ✅ Agent has tools or LLM (for Agent type)

### Validate Execution State

```python
from rh_agents import validate_state, StateValidationError

try:
    validate_state(execution_state)
    print("✅ State is valid!")
except StateValidationError as e:
    print(f"❌ State validation failed: {e}")
```

**Checks performed:**
- ✅ State ID exists
- ✅ Execution stack is initialized
- ✅ History is initialized
- ✅ Events have required fields
- ✅ Storage is not None
- ✅ Resume address is valid (if set)

### Validate Handler Signatures

```python
from rh_agents import validate_handler_signature
import warnings

# Check handler function signature
with warnings.catch_warnings(record=True) as w:
    validate_handler_signature(my_handler, "MyTool")
    
    if w:
        for warning in w:
            print(f"⚠️  {warning.message}")
```

**Checks performed:**
- ✅ Function is async
- ✅ Has minimum required parameters (input_data, context)
- ⚠️  Warns if parameter names don't follow conventions
- ⚠️  Warns if ExecutionState parameter is missing

### Using Validation in Production

```python
# Validate during initialization
from rh_agents import AgentBuilder, validate_actor

agent = (
    AgentBuilder()
    .name("MyAgent")
    .description("Does things")
    # ... more config
    .build()
)

# Validate before use
validate_actor(agent)

# Safe to use now
result = await ExecutionEvent(actor=agent)(input_data, "", state)
```

**Best Practices:**
- ✅ Validate actors after creation in development
- ✅ Use validation in tests to catch errors early
- ✅ Consider removing validation in hot paths for performance
- ✅ Validation helps during refactoring and maintenance

---

### With State Recovery

```python
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.state_recovery import StateStatus, StateMetadata

# Initialize backends
state_backend = FileSystemStateBackend(".state_store")
artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")

# Create execution state with recovery
state = ExecutionState(
    event_bus=bus,
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Execute your pipeline
await ExecutionEvent[Message](actor=agent)(message, "", state)

# Save checkpoint
state.save_checkpoint(
    status=StateStatus.COMPLETED,
    metadata=StateMetadata(
        tags=["greeting", "example"],
        description="Simple greeting pipeline"
    )
)

# Later, resume from checkpoint
restored_state = ExecutionState.load_from_state_id(
    state_id=state.state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend,
    event_bus=EventBus()
)
```

---

## 🧠 Core Concepts

### Actors

RH-Agents has three types of actors:

```python
from rh_agents.core.actors import Agent, Tool, LLM

# 1. Agents - High-level AI actors
class MyAgent(Agent):
    async def handler(self, input_data, context, execution_state):
        # Agent logic here
        return result

# 2. Tools - Callable functions for agents
class MyTool(Tool):
    async def handler(self, args, context, execution_state):
        # Tool logic here
        return Tool_Result(output=data, tool_name="MyTool")

# 3. LLMs - Language model wrappers
class MyLLM(LLM):
    async def handler(self, request, context, execution_state):
        # LLM API call here
        return LLM_Result(content=response)
```

### Execution State

The `ExecutionState` is the heart of RH-Agents:

```python
state = ExecutionState(
    event_bus=bus,              # For event publishing
    state_backend=backend,       # For state persistence
    artifact_backend=art_backend # For large object storage
)

# Store step results
state.add_step_result(0, result)

# Retrieve results
result = state.get_steps_result([0])

# Get execution address
address = state.get_current_address(EventType.AGENT_CALL)
# Returns: "ParentAgent::ChildAgent::agent_call"
```

### Event System

All executions publish events:

```python
from rh_agents.bus_handlers import EventPrinter

# Subscribe to events
printer = EventPrinter(show_timestamp=True, show_address=True)
bus.subscribe(printer)

# Events are automatically published during execution
# Output shows execution tree with timing and status
```

---

## 🔄 State Recovery

### Checkpoint & Resume

**Save a checkpoint:**

```python
# Execute your pipeline
result = await ExecutionEvent[Message](actor=omni_agent)(message, "", state)

# Save checkpoint with metadata
state_id = state.save_checkpoint(
    status=StateStatus.COMPLETED,
    metadata=StateMetadata(
        tags=["production", "v1"],
        description="Full pipeline execution",
        custom={"user_id": "123", "session": "abc"}
    )
)
```

**Resume from checkpoint:**

```python
from rh_agents.core.state_recovery import ReplayMode

# Load saved state
restored_state = ExecutionState.load_from_state_id(
    state_id=saved_state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend,
    event_bus=EventBus(),
    replay_mode=ReplayMode.NORMAL  # Skip completed events
)

# Execute same pipeline - completed steps are skipped instantly
result = await ExecutionEvent[Message](actor=omni_agent)(message, "", restored_state)
```

### Replay Modes

```python
from rh_agents.core.state_recovery import ReplayMode

# NORMAL: Skip completed events (fast resume)
ReplayMode.NORMAL

# VALIDATION: Re-execute everything, compare with history
ReplayMode.VALIDATION

# REPUBLISH_ALL: Republish all events to event bus
ReplayMode.REPUBLISH_ALL
```

### Resume from Specific Address

Resume execution from any point in your pipeline:

```python
# Resume from ReviewerAgent step (skip all earlier steps)
restored_state = ExecutionState.load_from_state_id(
    state_id=saved_state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend,
    event_bus=EventBus(),
    resume_from_address="OmniAgent::ReviewerAgent::final_review::agent_call"
)

# Execute - skips to ReviewerAgent, then continues normally
result = await ExecutionEvent[Message](actor=omni_agent)(message, "", restored_state)
```

**Benefits:**
- ⚡ Skip expensive early computation
- 🐛 Debug specific steps
- 🔄 Iterate on later stages without re-running everything
- 💰 Save API costs by avoiding redundant LLM calls

---

## 📚 Examples

### Example 1: Multi-Agent Pipeline

```python
from rh_agents.agents import (
    DoctrineReceverAgent,
    StepExecutorAgent,
    ReviewerAgent,
    OmniAgent,
    OpenAILLM
)

# Initialize LLM and agents
llm = OpenAILLM()
doctrine_agent = DoctrineReceverAgent(llm=llm, tools=tools)
executor_agent = StepExecutorAgent(llm=llm, tools=tools)
reviewer_agent = ReviewerAgent(llm=llm, tools=[])

# Create orchestrator
omni_agent = OmniAgent(
    receiver_agent=doctrine_agent,
    step_executor_agent=executor_agent,
    reviewer_agent=reviewer_agent
)

# Execute pipeline
message = Message(content="Analyze legal documents...", author=AuthorType.USER)
result = await ExecutionEvent[Message](actor=omni_agent)(message, "", state)
```

### Example 2: Custom Tool

Three ways to create the same tool:

**Option 1: Decorator API (Recommended for simple tools)**

```python
from rh_agents import tool_decorator, Tool_Result, ExecutionState
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Max results")

@tool_decorator(name="search", description="Search the knowledge base")
async def search_tool(
    args: SearchArgs,
    context: str,
    execution_state: ExecutionState
) -> Tool_Result:
    results = search_api(args.query, args.limit)
    return Tool_Result(output=results, tool_name="search")

# Use directly
tools = [search_tool]
```

**Option 2: Builder Pattern (For complex configuration)**

```python
from rh_agents import ToolBuilder

search_tool = (
    ToolBuilder()
    .name("search")
    .description("Search the knowledge base")
    .input_model(SearchArgs)
    .handler(search_handler)
    .cacheable(True)
    .version("2.0.0")
    .build()
)
```

**Option 3: Class-Based (For production systems)**

```python
from rh_agents import Tool, Tool_Result, ExecutionState
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Max results")

class SearchTool(Tool):
    def __init__(self):
        async def handler(args: SearchArgs, context: str, execution_state: ExecutionState):
            # Your search logic here
            results = search_api(args.query, args.limit)
            return Tool_Result(
                output=results,
                tool_name="SearchTool"
            )
        
        super().__init__(
            name="SearchTool",
            description="Search the knowledge base",
            input_model=SearchArgs,
            handler=handler
        )

# Use in agent
tools = [SearchTool()]
agent = StepExecutorAgent(llm=llm, tools=tools)
```

### Example 3: Event Monitoring

```python
from rh_agents.bus_handlers import EventPrinter

# Create custom event handler
def custom_handler(event):
    if event.execution_status == ExecutionStatus.COMPLETED:
        print(f"✅ {event.actor.name} completed in {event.execution_time:.2f}s")

# Subscribe handlers
printer = EventPrinter(show_timestamp=True, show_address=True)
bus.subscribe(printer)
bus.subscribe(custom_handler)

# Events are published automatically during execution
```

### Example 4: Artifact Storage

```python
# Mark agent as producing artifacts
class DataProcessorAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            name="DataProcessor",
            description="Processes large datasets",
            is_artifact=True,  # Results stored as artifacts
            handler=handler,
            # ... other params
        )

# Artifacts are automatically:
# - Stored with content-addressable IDs (SHA256)
# - Deduplicated across executions
# - Loaded during replay
```

### Example 5: Streaming API

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from rh_agents import EventBus, ExecutionEvent, EventStreamer

app = FastAPI()

@app.post("/execute")
async def execute_agent(request: dict):
    async def event_generator():
        bus = EventBus()
        streamer = EventStreamer()
        bus.subscribe(streamer)
        
        # Stream events as Server-Sent Events
        async def stream_events():
            async for event in streamer.stream():
                yield f"data: {event.model_dump_json()}\n\n"
        
        # Execute in background
        asyncio.create_task(
            ExecutionEvent[Message](actor=agent)(message, "", state)
        )
        
        async for data in stream_events():
            yield data
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### Example 6: Complete Agent with Decorators

```python
from rh_agents import (
    tool_decorator,
    agent_decorator,
    Tool_Result,
    Message,
    ExecutionState,
    validate_actor
)
from rh_agents.openai import OpenAILLM
from pydantic import BaseModel

# Define tools using decorators
class WeatherInput(BaseModel):
    city: str
    units: str = "celsius"

@tool_decorator(name="get_weather", description="Get weather for a city")
async def weather_tool(
    input: WeatherInput,
    context: str,
    state: ExecutionState
) -> Tool_Result:
    weather_data = fetch_weather(input.city, input.units)
    return Tool_Result(output=weather_data, tool_name="get_weather")

class NewsInput(BaseModel):
    topic: str
    count: int = 5

@tool_decorator(name="get_news", description="Get latest news")
async def news_tool(
    input: NewsInput,
    context: str,
    state: ExecutionState
) -> Tool_Result:
    news_data = fetch_news(input.topic, input.count)
    return Tool_Result(output=news_data, tool_name="get_news")

# Create agent with tools
llm = OpenAILLM(model="gpt-4")
tools = [weather_tool, news_tool]

@agent_decorator(
    name="InfoAgent",
    description="Provides weather and news information",
    tools=tools,
    llm=llm,
    cacheable=False
)
async def info_agent(
    input: Message,
    context: str,
    state: ExecutionState
) -> Message:
    # Agent uses LLM and tools automatically
    response = await llm.handler(
        input,
        context,
        state
    )
    return Message(content=response.content, role="assistant")

# Validate and use
validate_actor(info_agent)
result = await ExecutionEvent(actor=info_agent)(
    Message(content="What's the weather in London?", role="user"),
    "",
    state
)
```

### Example 7: Builder Pattern for Complex Agent

```python
from rh_agents import AgentBuilder, ToolBuilder, validate_actor
from rh_agents.openai import OpenAILLM

# Build tools
db_tool = (
    ToolBuilder()
    .name("database_query")
    .description("Query the database")
    .input_model(QueryInput)
    .handler(db_handler)
    .cacheable(True)
    .version("1.2.0")
    .build()
)

api_tool = (
    ToolBuilder()
    .name("api_call")
    .description("Call external API")
    .input_model(APIInput)
    .handler(api_handler)
    .cacheable(False)
    .version("1.0.0")
    .build()
)

# Build complex agent with validation
agent = (
    AgentBuilder()
    .name("DataAgent")
    .description("Retrieves data from multiple sources")
    .input_model(Message)
    .output_model(Message)
    .handler(data_handler)
    .with_llm(OpenAILLM(model="gpt-4"))
    .with_tools([db_tool, api_tool])
    .cacheable(True)
    .version("2.1.0")
    .add_precondition(lambda state: state.storage is not None)
    .add_postcondition(lambda result: result is not None)
    .build()
)

# Validate before use
validate_actor(agent)

# Execute safely
result = await ExecutionEvent(actor=agent)(input_msg, "", state)
```

---

## 📖 API Reference

### Actor Creation APIs

#### Decorator API

```python
from rh_agents import tool_decorator, agent_decorator

@tool_decorator(
    name: str | None = None,          # Tool name (default: function name)
    description: str | None = None,    # Description (default: docstring)
    cacheable: bool = False,           # Enable caching
    version: str = "1.0.0"             # Version string
)
async def my_tool(input: InputModel, context: str, state: ExecutionState) -> Tool_Result:
    ...

@agent_decorator(
    name: str | None = None,           # Agent name (default: function name)
    description: str | None = None,    # Description (default: docstring)
    tools: list[Tool] | None = None,   # Available tools
    llm: LLM | None = None,            # LLM instance
    cacheable: bool = False            # Enable caching
)
async def my_agent(input: InputModel, context: str, state: ExecutionState) -> OutputModel:
    ...
```

#### Builder API

```python
from rh_agents import AgentBuilder, ToolBuilder

# AgentBuilder
agent = (
    AgentBuilder()
    .name(str) -> AgentBuilder
    .description(str) -> AgentBuilder
    .input_model(type[BaseModel]) -> AgentBuilder
    .output_model(type[BaseModel]) -> AgentBuilder
    .handler(Callable) -> AgentBuilder
    .with_llm(LLM) -> AgentBuilder
    .with_tools(list[Tool]) -> AgentBuilder
    .add_tool(Tool) -> AgentBuilder
    .cacheable(bool) -> AgentBuilder
    .version(str) -> AgentBuilder
    .add_precondition(Callable) -> AgentBuilder
    .add_postcondition(Callable) -> AgentBuilder
    .build() -> Agent
)

# ToolBuilder
tool = (
    ToolBuilder()
    .name(str) -> ToolBuilder
    .description(str) -> ToolBuilder
    .input_model(type[BaseModel]) -> ToolBuilder
    .output_model(type[BaseModel]) -> ToolBuilder
    .handler(Callable) -> ToolBuilder
    .cacheable(bool) -> ToolBuilder
    .version(str) -> ToolBuilder
    .build() -> Tool
)
```

#### Validation API

```python
from rh_agents import (
    validate_actor,
    validate_state,
    validate_handler_signature,
    ActorValidationError,
    StateValidationError
)

# Validate actor configuration
validate_actor(actor: BaseActor) -> None
# Raises: ActorValidationError

# Validate execution state
validate_state(state: ExecutionState) -> None
# Raises: StateValidationError

# Validate handler function signature
validate_handler_signature(
    handler: Callable,
    actor_name: str
) -> None
# Issues warnings for signature problems
```

### ExecutionState

```python
class ExecutionState:
    # Core methods
    def push_context(self, name: str) -> None
    def pop_context(self) -> str | None
    def get_current_address(self, event_type: EventType) -> str
    
    # Storage
    def add_step_result(self, step_index: int, value: Any) -> None
    def get_steps_result(self, step_indices: list[int]) -> list[Any]
    
    # State recovery
    def save_checkpoint(self, status: StateStatus, metadata: StateMetadata) -> bool
    @classmethod
    def load_from_state_id(cls, state_id: str, ...) -> ExecutionState
    
    # Replay control
    def should_skip_event(self, address: str) -> bool
```

### ExecutionEvent

```python
class ExecutionEvent[OutputT]:
    actor: BaseActor
    address: str
    execution_time: float | None
    result: Any | None
    is_replayed: bool
    
    # Execute actor
    async def __call__(
        self,
        input_data,
        extra_context: str,
        execution_state: ExecutionState
    ) -> ExecutionResult[OutputT]
```

### StateBackend

```python
class StateBackend:
    def save_state(self, snapshot: StateSnapshot) -> bool
    def load_state(self, state_id: str) -> StateSnapshot | None
    def list_states(self) -> list[str]
    def delete_state(self, state_id: str) -> bool
```

### ArtifactBackend

```python
class ArtifactBackend:
    def save_artifact(self, artifact_id: str, artifact: Any) -> bool
    def load_artifact(self, artifact_id: str) -> Any | None
    def delete_artifact(self, artifact_id: str) -> bool
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Application                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Agent   │───▶│  Agent   │───▶│  Agent   │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │                  │
│       └───────────────┴───────────────┘                  │
│                       ▼                                   │
│              ┌─────────────────┐                         │
│              │ ExecutionState  │                         │
│              │  - History      │                         │
│              │  - Storage      │                         │
│              │  - Stack        │                         │
│              └────────┬────────┘                         │
│                       │                                   │
│          ┌────────────┴────────────┐                     │
│          ▼                         ▼                     │
│   ┌─────────────┐          ┌─────────────┐              │
│   │ StateBackend│          │ArtifactBackend│             │
│   │  (JSON)     │          │   (Pickle)   │             │
│   └─────────────┘          └─────────────┘              │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Actors** (Agent/Tool/LLM) - Execution units
2. **ExecutionEvent** - Wraps actors with replay logic
3. **ExecutionState** - Manages execution context and history
4. **EventBus** - Publishes events to subscribers
5. **Backends** - Persist state and artifacts
6. **State Recovery** - Checkpoint/resume functionality

---

## 💡 Best Practices

### Actor Creation

**✅ DO:**
- Use decorators for quick prototypes and simple tools
- Use builders for complex agents with many configuration options
- Use classes for production-ready, reusable components
- Validate actors after creation during development
- Keep handler functions focused and testable
- Use type hints for better IDE support

**❌ DON'T:**
- Mix creation patterns unnecessarily (choose one style per project)
- Skip validation in development (catches errors early)
- Create overly complex handlers (split into smaller functions)
- Ignore handler signature warnings

### State Management

**✅ DO:**
- Save checkpoints at logical breakpoints in your pipeline
- Use meaningful tags and metadata for searchability
- Clean up old states periodically
- Use `resume_from_address` to debug specific steps
- Enable validation mode when debugging replay issues

**❌ DON'T:**
- Save checkpoints on every single event (overhead)
- Store large objects outside artifact system
- Modify state history manually
- Forget to configure backends before saving

### Performance

**✅ DO:**
- Enable caching for deterministic, expensive operations
- Use artifact system for large data (>1MB)
- Process independent tasks in parallel (see `ParallelEventGroup`)
- Stream events for long-running operations
- Use selective resume to skip expensive early steps

**❌ DON'T:**
- Cache non-deterministic operations
- Store ephemeral data in state
- Chain too many agents without checkpoints
- Re-run entire pipelines when debugging later stages

### Development Workflow

1. **Prototype** with decorator API
2. **Validate** using `validate_actor()` and `validate_state()`
3. **Test** with different inputs and edge cases
4. **Graduate** to builders/classes for production
5. **Monitor** with EventPrinter during development
6. **Checkpoint** at key pipeline stages
7. **Debug** using selective resume and validation mode

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional LLM integrations (Anthropic, Cohere, etc.)
- [ ] Database-backed state storage
- [ ] Distributed execution support
- [ ] Enhanced validation mode
- [ ] Schema versioning and migrations
- [ ] Automatic garbage collection

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **GitHub**: [rhodief/rh-agents](https://github.com/rhodief/rh-agents)
- **Documentation**: Coming soon
- **Examples**: See `/examples` directory

---

## 🙏 Acknowledgments

Built with:
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [OpenAI](https://openai.com/) - LLM integration
- [FastAPI](https://fastapi.tiangolo.com/) - API examples

---

**Made with ❤️ by [rhodie](https://github.com/rhodief)**