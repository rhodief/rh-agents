# Builder Pattern User Guide

## Overview

The Builder Pattern provides simplified APIs for creating agents that follow common patterns, dramatically reducing boilerplate code while maintaining full type safety and ExecutionEvent integration.

## Quick Start

```python
from rh_agents.builders import StructuredAgent, CompletionAgent
from rh_agents.openai import OpenAILLM

llm = OpenAILLM()

# Create an agent in one line (+ optional configuration)
parser = (
    await StructuredAgent.from_model(
        name="TaskParser",
        llm=llm,
        input_model=UserRequest,
        output_model=ParsedTask,
        system_prompt="Parse user request into structured task."
    )
    .with_temperature(0.7)
    .with_max_tokens(500)
    .as_cacheable()
)
```

**Before builders (40+ lines):**
```python
# Define handler function
async def parse_handler(input_data, context, execution_state):
    # Build system prompt
    prompt = "Parse user request..."
    if context:
        prompt += f"\n\nContext: {context}"
    
    # Create ExecutionEvent for LLM
    llm_event = ExecutionEvent(actor=llm)
    
    # Prepare request
    llm_input = OpenAIRequest(
        system_message=prompt,
        prompt=input_data.content,
        model='gpt-4o',
        temperature=0.7,
        max_completion_tokens=500,
        tool_choice={"type": "function", "function": {"name": "ParseTool"}}
    )
    
    # Execute and handle errors
    result = await llm_event(llm_input, context, execution_state)
    if not result.ok:
        raise Exception(f"LLM failed: {result.erro_message}")
    
    # Extract tool call
    if not result.result.is_tool_call:
        raise Exception("Expected tool call")
    
    tool_call = result.result.tools[0]
    return ParsedTask.model_validate_json(tool_call.arguments)

# Create agent
parser = Agent(
    name="TaskParser",
    description="Parse user request into structured task.",
    input_model=UserRequest,
    output_model=ParsedTask,
    handler=parse_handler,
    event_type=EventType.AGENT_CALL,
    llm=llm,
    cacheable=True
)
```

---

## The Four Builder Types

### 1. StructuredAgent

**When to use:** You need the LLM to return **typed, validated data** matching a Pydantic schema.

**Pattern:** Forces LLM to call a tool that validates against `output_model`. Guarantees structured output.

```python
from pydantic import BaseModel, Field

class ParsedTask(BaseModel):
    task_type: str = Field(description="Type: search, create, analyze")
    subject: str
    priority: str = "normal"

parser = await StructuredAgent.from_model(
    name="TaskParser",
    llm=llm,
    input_model=UserRequest,
    output_model=ParsedTask,  # LLM output MUST match this schema
    system_prompt="Parse user's request into structured task format."
)
```

**Key characteristics:**
- Output always matches `output_model` schema (or raises error)
- Uses tool choice to force structured response
- Perfect for parsing, extraction, classification
- Type-safe by design

**Common use cases:**
- Parse user input → workflow parameters
- Extract entities from text → structured data
- Classify documents → category + metadata
- Transform unstructured → structured data

---

### 2. CompletionAgent

**When to use:** You need **natural language text output** (not structured data).

**Pattern:** Simple LLM completion without tools. Returns `Message` with text content.

```python
summarizer = await CompletionAgent.from_prompt(
    name="Summarizer",
    llm=llm,
    input_model=Document,
    output_model=Message,  # Returns Message with text
    system_prompt="Summarize the document in 2-3 sentences."
)
```

**Key characteristics:**
- No tool calling
- Returns `Message` with natural language content
- Fast and simple
- Perfect for generation, summarization, analysis

**Common use cases:**
- Summarize long documents
- Generate reports or descriptions
- Answer questions (Q&A)
- Creative writing
- Analyze and explain data

---

### 3. ToolExecutorAgent

**When to use:** The LLM needs to **interact with tools** (APIs, databases, computations).

**Pattern:** LLM decides which tools to call and with what parameters. Executes them in parallel and aggregates results.

```python
executor = await ToolExecutorAgent.from_tools(
    name="DataPipeline",
    llm=llm,
    input_model=Query,
    output_model=Result,  # Not used - returns ToolExecutionResult
    system_prompt="Use available tools to fulfill the request.",
    tools=[SearchTool(), AnalyzeTool(), ProcessTool()]
)

# Result is always ToolExecutionResult
result = await executor(query)
print(result.results)  # {'SearchTool': [...], 'AnalyzeTool': {...}}
print(result.execution_order)  # ['SearchTool', 'AnalyzeTool']
print(result.has_errors())  # False
```

**Key characteristics:**
- LLM autonomously selects tools
- Parallel execution via `asyncio.gather`
- Returns `ToolExecutionResult` with aggregated outputs
- Handles tool errors gracefully

**Output structure:**
```python
class ToolExecutionResult:
    results: dict[str, Any]  # {tool_name: output}
    execution_order: list[str]  # Tools called, in order
    errors: dict[str, str]  # {tool_name: error_message}
    
    def get(self, tool_name: str) -> Any: ...
    def first(self) -> Any: ...
    def has_errors(self) -> bool: ...
    def all_failed(self) -> bool: ...
```

**Common use cases:**
- Multi-step data pipelines
- Research workflows (search → analyze → report)
- System interactions (query DB → process → save)
- Agent workflows with multiple capabilities

---

### 4. DirectToolAgent

**When to use:** You need **deterministic tool execution without LLM reasoning**.

**Pattern:** Bypasses LLM entirely. Directly invokes tool handler with input.

```python
validator = await DirectToolAgent.from_tool(
    name="Validator",
    tool=ValidationTool()
)

# Faster, cheaper, deterministic
result = await validator(data)
```

**Key characteristics:**
- No LLM call (faster, cheaper)
- Deterministic behavior
- Direct input → tool → output
- Still uses ExecutionEvent for tracking

**Common use cases:**
- Data validation
- Format transformation
- Database queries (known parameters)
- API calls (predetermined)
- Any non-reasoning operation

**When NOT to use:**
- If you need LLM to decide what to do
- If parameters require interpretation
- If you need natural language understanding

---

## Chainable Configuration Methods

All builders return `BuilderAgent` instances with these chainable methods. All methods return `self` for chaining.

### LLM Configuration

Control the LLM behavior with validated parameters:

```python
agent = (
    await CompletionAgent.from_prompt(...)
    .with_temperature(0.7)        # Control randomness (0.0-2.0, validated)
    .with_max_tokens(2000)         # Max response length (1-128000, validated)
    .with_model('gpt-4o-mini')     # Override model name
)
```

**Parameter Details:**

| Method | Parameter | Range | Description |
|--------|-----------|-------|-------------|
| `with_temperature()` | `temperature: float` | 0.0 - 2.0 | Creativity level: 0=deterministic, 2=very random |
| `with_max_tokens()` | `max_tokens: int` | 1 - 128000 | Maximum tokens to generate |
| `with_model()` | `model: str` | Any valid model | Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini') |

All parameters are **validated** at configuration time. Invalid values raise `ValueError` immediately.

**Examples:**
```python
# Low creativity for factual content
agent.with_temperature(0.2)

# High creativity for brainstorming
agent.with_temperature(1.5)

# Short responses
agent.with_max_tokens(500)

# Use cheaper model
agent.with_model('gpt-4o-mini')
```

### Tool Configuration

```python
agent = (
    await StructuredAgent.from_model(...)
    .with_tools([Tool1(), Tool2()])  # Set/replace tools
    .with_tool_choice("Tool1")        # Force specific tool (structured output)
)
```

### Prompt Engineering

Customize how prompts are built dynamically:

```python
# Static context transform
agent = agent.with_context_transform(
    lambda ctx: f"\n\n[CONTEXT]: {ctx.upper()}"
)

# Dynamic prompt builder with full access
async def build_prompt(input_data, context, state):
    # Access to input, context, and execution state
    prior_results = state.get_steps_result(['step1', 'step2'])
    return f"Process {input_data.task_type}: {input_data.subject}\n\nPrior: {prior_results}"

agent = agent.with_system_prompt_builder(build_prompt)
```

**Use Cases:**
- **Context Transform**: Format context string before appending to system prompt
- **Prompt Builder**: Full dynamic prompt generation with access to execution state, prior results, etc.

### Error Handling

Comprehensive error control with retry support:

```python
from rh_agents.core.types import ErrorStrategy

agent = (
    agent
    .with_error_strategy(ErrorStrategy.RETURN_NONE)
    .with_retry(
        max_attempts=3,      # 1-10 attempts (validated)
        initial_delay=1.0    # Must be > 0 (validated)
    )
)
```

**Error Strategies:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `RAISE` (default) | Raise exception immediately | Critical operations, debugging |
| `RETURN_NONE` | Return `None` or `ExecutionResult(ok=False)` | Optional processing, graceful degradation |
| `LOG_AND_CONTINUE` | Log warning, return empty instance | Non-critical operations |
| `SILENT` | Return `None` without logging | Background tasks |

**Retry Configuration:**
```python
# Conservative retry
agent.with_retry(max_attempts=3, initial_delay=1.0)

# Aggressive retry with backoff
agent.with_retry(
    max_attempts=5,
    initial_delay=0.5,
    backoff_factor=2.0,  # Exponential backoff
    max_delay=10.0       # Cap at 10 seconds
)
```

### Optimization

Performance and caching options:

```python
agent = (
    agent
    .as_cacheable(ttl=300)  # Enable caching (300s TTL, validated >= 0)
    .as_artifact()           # Mark output as artifact for separate storage
)
```

**Caching Details:**
- `ttl=None`: Cache indefinitely (until manual clear)
- `ttl=0`: Cache with no expiration
- `ttl=300`: Cache for 300 seconds
- `ttl` must be non-negative (validated)

**Artifacts:**
- Stores result separately from main execution trace
- Useful for large outputs, generated content, etc.

### Tool Execution Control

```python
# For ToolExecutorAgent only
executor = (
    executor
    .with_first_result_only()  # Stop after first successful tool
)
```

### Result Aggregation Strategies

**For `ToolExecutorAgent` only** - Control how multiple tool execution results are combined.

```python
from rh_agents.core.types import AggregationStrategy

# Default: Return ToolExecutionResult (dict-like access)
executor = await ToolExecutorAgent.from_tools(...)

# LIST: Return ordered list of results
executor = executor.with_aggregation(AggregationStrategy.LIST)
# Returns: [result1, result2, result3]

# CONCATENATE: Join results as string
executor = executor.with_aggregation(
    AggregationStrategy.CONCATENATE, 
    separator="\n"  # Optional, default is " "
)
# Returns: "result1\nresult2\nresult3"

# FIRST: Return only the first result
executor = executor.with_aggregation(AggregationStrategy.FIRST)
# Returns: result1
```

**Strategy Details:**

| Strategy | Return Type | Use Case | Example |
|----------|------------|----------|---------|
| **DICT** (default) | `ToolExecutionResult` | Need dict-like access by tool name | `result["Tool1"]`, `result.execution_order` |
| **LIST** | `list` | Sequential processing, iteration | `[r for r in results]` |
| **CONCATENATE** | `str` | Text aggregation, summaries | `"Result 1. Result 2. Result 3"` |
| **FIRST** | `Any` | Only care about first result | `first_result.field` |

**Example Use Case:**

```python
# Multi-source search aggregation
search_executor = (
    await ToolExecutorAgent.from_tools(
        name="MultiSearch",
        llm=llm,
        input_model=SearchQuery,
        output_model=str,  # Want concatenated string
        system_prompt="Search using available tools",
        tools=[web_search, doc_search, db_search]
    )
    .with_aggregation(AggregationStrategy.CONCATENATE, separator="\n\n---\n\n")
)

result = await search_executor("find user data")
# Returns:
# "Web: Found 3 results...
# ---
# Docs: API endpoint is...
# ---
# DB: User record: {...}"
```

---

## Decision Tree: Which Builder?

```
Do you need structured output matching a schema?
├─ YES → StructuredAgent
│   └─ Example: Parse text → Pydantic model
│
└─ NO → Do you need to call external tools/APIs?
    ├─ YES → Does the LLM need to decide which tools?
    │   ├─ YES → ToolExecutorAgent
    │   │   └─ Example: "Search and analyze" → LLM picks tools
    │   │
    │   └─ NO → DirectToolAgent
    │       └─ Example: Validate data (deterministic)
    │
    └─ NO → CompletionAgent
        └─ Example: Summarize, generate text, Q&A
```

---

## Design Patterns

### Pattern 1: Parse → Validate → Execute

```python
# 1. Parse user input (structured)
parser = await StructuredAgent.from_model(
    name="Parser",
    llm=llm,
    input_model=UserMessage,
    output_model=ParsedTask,
    system_prompt="Parse user request"
)

# 2. Validate task (deterministic)
validator = await DirectToolAgent.from_tool(
    name="Validator",
    tool=ValidationTool()
)

# 3. Execute task (tools)
executor = await ToolExecutorAgent.from_tools(
    name="Executor",
    llm=llm,
    input_model=ParsedTask,
    output_model=TaskResult,
    system_prompt="Execute task using tools",
    tools=[...tools...]
)

# 4. Summarize results (completion)
summarizer = await CompletionAgent.from_prompt(
    name="Summarizer",
    llm=llm,
    input_model=ToolExecutionResult,
    output_model=Message,
    system_prompt="Summarize execution results"
)

# Chain them in a workflow
```

### Pattern 2: Parallel Execution

```python
# Multiple independent operations
agents = [
    await CompletionAgent.from_prompt(...),
    await CompletionAgent.from_prompt(...),
    await CompletionAgent.from_prompt(...)
]

# Execute in parallel
results = await asyncio.gather(*[agent(input) for agent in agents])
```

### Pattern 3: Dynamic Prompts

```python
async def build_context_prompt(input_data, context, state):
    # Access current state
    history = await state.get_history()
    
    # Build dynamic prompt
    return f"""
    Current task: {input_data.task}
    Previous results: {len(history)} operations
    
    Process this request considering the context above.
    """

agent = (
    await CompletionAgent.from_prompt(...)
    .with_system_prompt_builder(build_context_prompt)
)
```

### Pattern 4: Fallback Chains

```python
primary = (
    await ToolExecutorAgent.from_tools(...)
    .with_error_strategy(ErrorStrategy.RETURN_NONE)
)

fallback = (
    await CompletionAgent.from_prompt(...)
)

# Try primary, fall back if it fails
result = await primary(input)
if result is None or not result.ok:
    result = await fallback(input)
```

---

## Best Practices

### 1. Choose the Right Builder

✅ **DO:**
- Use `StructuredAgent` when output schema matters
- Use `CompletionAgent` for natural language responses
- Use `ToolExecutorAgent` for multi-tool workflows
- Use `DirectToolAgent` for deterministic operations

❌ **DON'T:**
- Use `StructuredAgent` for free-form text (use `CompletionAgent`)
- Use `ToolExecutorAgent` when parameters are known (use `DirectToolAgent`)
- Use `DirectToolAgent` when LLM reasoning is needed

### 2. Configure Appropriately

✅ **DO:**
```python
# Structured output → low temperature
parser = (
    await StructuredAgent.from_model(...)
    .with_temperature(0.3)  # Deterministic
)

# Creative writing → high temperature
writer = (
    await CompletionAgent.from_prompt(...)
    .with_temperature(0.9)  # Creative
)
```

❌ **DON'T:**
```python
# High temperature for parsing (inconsistent results)
parser = await StructuredAgent.from_model(...).with_temperature(1.5)
```

### 3. Error Handling

✅ **DO:**
```python
# Critical operations → RAISE
auth = agent.with_error_strategy(ErrorStrategy.RAISE)

# Optional enhancements → RETURN_NONE
recommendations = agent.with_error_strategy(ErrorStrategy.RETURN_NONE)
```

❌ **DON'T:**
```python
# Silent failures in critical paths
payment = agent.with_error_strategy(ErrorStrategy.SILENT)
```

### 4. Caching Strategy

✅ **DO:**
```python
# Expensive, stable operations → cache with TTL
researcher = (
    agent
    .as_cacheable(ttl=3600)  # 1 hour
)

# User-specific, dynamic → no cache
personalizer = agent  # No caching
```

❌ **DON'T:**
```python
# Cache user-specific data
profile_builder = agent.as_cacheable(ttl=3600)  # BAD: varies per user
```

### 5. System Prompts

✅ **DO:**
```python
system_prompt = """
Parse the user's request into a structured task format.

Extract:
1. Task type (search, create, analyze, etc.)
2. Main subject or topic
3. Any parameters or constraints
4. Priority level if mentioned
"""
```

❌ **DON'T:**
```python
system_prompt = "Parse this"  # Too vague
```

---

## Performance Tips

### 1. Model Selection

```python
# Simple tasks → use cheaper model
classifier = (
    await StructuredAgent.from_model(...)
    .with_model('gpt-4o-mini')  # Faster, cheaper
)

# Complex reasoning → use powerful model
strategist = (
    await ToolExecutorAgent.from_tools(...)
    .with_model('gpt-4o')  # More capable
)
```

### 2. Token Optimization

```python
# Short responses → limit tokens
summarizer = agent.with_max_tokens(200)

# Long-form → allow more tokens
analyst = agent.with_max_tokens(4000)
```

### 3. Parallel Execution

```python
# ToolExecutorAgent executes tools in parallel automatically
executor = await ToolExecutorAgent.from_tools(
    tools=[SlowTool1(), SlowTool2(), SlowTool3()]
)
# All 3 tools run concurrently!
```

### 4. Direct Tool for Known Operations

```python
# Skip LLM when you know what to do
validator = await DirectToolAgent.from_tool(tool=ValidationTool())
# Faster: no LLM API call
# Cheaper: no tokens used
# Predictable: deterministic behavior
```

---

## Testing

### Unit Tests

```python
from unittest.mock import AsyncMock, patch

async def test_completion_agent():
    llm = MagicMock()
    
    agent = await CompletionAgent.from_prompt(
        name="Test",
        llm=llm,
        input_model=Input,
        output_model=Message,
        system_prompt="Test"
    )
    
    # Mock ExecutionEvent
    with patch('rh_agents.builders.ExecutionEvent') as mock:
        mock_event = AsyncMock()
        mock_event.return_value = ExecutionResult(
            result=LLM_Result(content="Mock response", is_content=True),
            ok=True
        )
        mock.return_value = mock_event
        
        result = await agent.handler(Input(content="test"), "", state)
        assert result.content == "Mock response"
```

### Integration Tests

```python
async def test_full_workflow():
    """Test complete pipeline with real LLM."""
    llm = OpenAILLM()
    
    parser = await StructuredAgent.from_model(...)
    executor = await ToolExecutorAgent.from_tools(...)
    summarizer = await CompletionAgent.from_prompt(...)
    
    # Run full pipeline
    task = await parser(user_input)
    results = await executor(task)
    summary = await summarizer(results)
    
    assert isinstance(summary, Message)
```

---

## Common Patterns

### 1. First Result Only

```python
# Stop after first successful tool call
executor = (
    await ToolExecutorAgent.from_tools(...)
    .with_first_result_only()
)

result = await executor(query)
# Only one tool in result.results
```

### 2. Retry on Failure

```python
agent = agent.with_retry(
    max_attempts=3,
    initial_delay=1.0,
    exponential_backoff=True
)
```

### 3. Custom Error Messages

```python
async def custom_error_handler(input_data, context, state):
    try:
        return await agent.handler(input_data, context, state)
    except Exception as e:
        await state.log(f"Custom error: {e}")
        return default_response
```

---

## Migration from Manual Agents

**Before (manual):**
```python
async def handler(input_data, context, execution_state):
    llm_event = ExecutionEvent(actor=llm)
    # ... 30+ lines of boilerplate ...
    
agent = Agent(
    name="Parser",
    handler=handler,
    # ... many parameters ...
)
```

**After (builder):**
```python
agent = (
    await StructuredAgent.from_model(
        name="Parser",
        llm=llm,
        input_model=Input,
        output_model=Output,
        system_prompt="Parse this"
    )
    .with_temperature(0.7)
    .as_cacheable()
)
```

**Benefits:**
- 90% less boilerplate
- Type-safe by default
- ExecutionEvent integration automatic
- Chainable configuration
- Consistent error handling

---

## See Also

- **Specification:** `docs/BOILERPLATE_REDUCTION_SPEC.md`
- **Examples:** `examples/builder_basic.py`
- **Tests:** `tests/test_builders_unit.py`, `tests/test_builders_integration.py`
- **API Reference:** `rh_agents/builders.py`

---

## Quick Reference

| Builder | Use Case | Returns | LLM | Tools |
|---------|----------|---------|-----|-------|
| `StructuredAgent` | Typed data extraction | `output_model` | Yes | Optional |
| `CompletionAgent` | Text generation | `Message` | Yes | No |
| `ToolExecutorAgent` | Multi-tool workflows | `ToolExecutionResult` | Yes | Yes (required) |
| `DirectToolAgent` | Deterministic ops | Tool output | No | Yes (1 tool) |

**Chainable methods (all builders):**
- `.with_temperature(float)`
- `.with_max_tokens(int)`
- `.with_model(str)`
- `.with_tools(list[Tool])`
- `.with_tool_choice(str)`
- `.with_context_transform(fn)`
- `.with_system_prompt_builder(fn)`
- `.with_error_strategy(ErrorStrategy)`
- `.with_retry(max_attempts, ...)`
- `.with_first_result_only()` (ToolExecutorAgent only)
- `.as_artifact()`
- `.as_cacheable(ttl=None)`
