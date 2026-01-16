# 🤖 RH-Agents

**A powerful Python framework for building stateless, resumable multi-agent AI workflows with built-in state recovery.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [State Recovery](#-state-recovery)
- [Examples](#-examples)
- [API Reference](#-api-reference)
- [Architecture](#-architecture)

---

## ✨ Features

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

**Optional:**
- openai (for OpenAI LLM integration)
- fastapi (for streaming API examples)

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

```python
from rh_agents.core.actors import Tool
from rh_agents.core.result_types import Tool_Result
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

app = FastAPI()

@app.post("/execute")
async def execute_agent(request: dict):
    async def event_generator():
        bus = EventBus()
        
        # Stream events as SSE
        async def stream_events():
            async for event in bus.stream():
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

---

## 📖 API Reference

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