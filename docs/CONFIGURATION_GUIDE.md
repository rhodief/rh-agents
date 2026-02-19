# Builder Configuration Guide

## Overview

This guide covers all configuration options available in the Builder Pattern API. All builders (`StructuredAgent`, `CompletionAgent`, `ToolExecutorAgent`, `DirectToolAgent`) support chainable configuration methods.

---

## Quick Reference

```python
agent = (
    await StructuredAgent.from_model(...)
    .with_temperature(0.7)              # LLM randomness
    .with_max_tokens(2000)              # Response length
    .with_model('gpt-4o-mini')          # Model selection
    .with_error_strategy(ErrorStrategy.RETURN_NONE)  # Error handling
    .with_retry(max_attempts=3)         # Retry behavior
    .as_cacheable(ttl=300)              # Caching
    .as_artifact()                      # Artifact storage
)
```

---

## Configuration Categories

### 1. LLM Parameters

Control how the LLM generates responses.

#### `with_temperature(temperature: float)`

Controls output randomness/creativity.

**Parameters:**
- `temperature`: Float between 0.0 and 2.0 (validated)

**Ranges:**
- `0.0`: Completely deterministic, always picks most likely token
- `0.1-0.3`: Very focused, factual, consistent
- `0.7-1.0`: Balanced, default for most tasks
- `1.3-2.0`: Creative, diverse, less predictable

**Examples:**
```python
# Factual content extraction
parser = parser.with_temperature(0.2)

# Creative writing
writer = writer.with_temperature(1.5)

# Balanced analysis
analyzer = analyzer.with_temperature(0.7)
```

**Validation:**
- Raises `ValueError` if temperature < 0.0 or > 2.0
- Validated at configuration time (fail-fast)

---

#### `with_max_tokens(max_tokens: int)`

Sets maximum response length in tokens.

**Parameters:**
- `max_tokens`: Integer between 1 and 128000 (validated)

**Guidelines:**
- 1 token ≈ 0.75 words (English)
- Leave headroom: request 10-20% more tokens than expected
- Balance between cost and completeness

**Examples:**
```python
# Short responses (summaries, classifications)
agent.with_max_tokens(500)

# Medium responses (analysis, explanations)
agent.with_max_tokens(2000)

# Long responses (comprehensive reports)
agent.with_max_tokens(8000)
```

**Validation:**
- Raises `ValueError` if max_tokens < 1 or > 128000
- Note: Actual model limits may be lower (check model docs)

---

#### `with_model(model: str)`

Selects which LLM model to use.

**Parameters:**
- `model`: Model identifier string

**Common Models:**
- `gpt-4o`: Latest, most capable (expensive)
- `gpt-4o-mini`: Fast, cost-effective, good quality
- `gpt-3.5-turbo`: Legacy, cheaper (deprecated soon)

**Examples:**
```python
# High-quality critical operations
agent.with_model('gpt-4o')

# Cost-optimized standard operations
agent.with_model('gpt-4o-mini')
```

**No validation** - model name passed directly to LLM provider

---

### 2. Prompt Engineering

Customize how prompts are constructed.

#### `with_context_transform(fn: Callable[[str], str])`

Transforms context string before appending to system prompt.

**Parameters:**
- `fn`: Function that takes context string, returns transformed string

**Use Cases:**
- Formatting context with special markers
- Filtering sensitive information
- Adding structure (XML, JSON, etc.)

**Examples:**
```python
# Add XML tags
agent.with_context_transform(
    lambda ctx: f"<context>{ctx}</context>" if ctx else ""
)

# Uppercase for emphasis
agent.with_context_transform(
    lambda ctx: f"\n\nIMPORTANT CONTEXT:\n{ctx.upper()}"
)

# Custom formatting
def format_context(ctx):
    if not ctx:
        return ""
    lines = ctx.split('\n')
    return "\n".join(f"- {line}" for line in lines)

agent.with_context_transform(format_context)
```

---

#### `with_system_prompt_builder(fn: Callable)`

Completely overrides system prompt generation with custom logic.

**Signature:**
```python
async def custom_prompt(
    input_data: BaseModel,
    context: str,
    execution_state: ExecutionState
) -> str:
    # Return complete system prompt
    ...
```

**Access:**
- `input_data`: Current input to the agent
- `context`: Context string passed to execution
- `execution_state`: Full execution state with prior results

**Use Cases:**
- Dynamic prompts based on input
- Incorporate prior execution results
- Complex multi-stage workflows

**Examples:**
```python
# Access prior results
async def build_with_history(input_data, context, state):
    prior = state.get_steps_result(['step1', 'step2'])
    return f"Task: {input_data.task}\n\nPrior results:\n{prior}"

agent.with_system_prompt_builder(build_with_history)

# Conditional prompts
async def conditional_prompt(input_data, context, state):
    if input_data.task_type == "search":
        return "You are a search specialist..."
    else:
        return "You are an analysis specialist..."

agent.with_system_prompt_builder(conditional_prompt)
```

---

### 3. Error Handling

Control failure behavior and retry logic.

#### `with_error_strategy(strategy: ErrorStrategy)`

Sets how errors are handled during execution.

**Strategies:**

| Strategy | Behavior | Return Value | Exception | Use Case |
|----------|----------|--------------|-----------|----------|
| `RAISE` | Fail immediately | N/A | Yes | Critical operations, debugging |
| `RETURN_NONE` | Return failure result | `None` or `ExecutionResult(ok=False)` | No | Optional processing |
| `LOG_AND_CONTINUE` | Log and return empty | Empty instance | No | Non-critical operations |
| `SILENT` | Suppress completely | `None` | No | Background tasks |

**Examples:**
```python
from rh_agents.core.types import ErrorStrategy

# Critical: must succeed or fail
payment_agent.with_error_strategy(ErrorStrategy.RAISE)

# Optional: gracefully handle failure
optional_enrichment.with_error_strategy(ErrorStrategy.RETURN_NONE)

# Non-critical: log warnings
logging_agent.with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)

# Background: silent failure
background_sync.with_error_strategy(ErrorStrategy.SILENT)
```

---

#### `with_retry(max_attempts: int, initial_delay: float, **kwargs)`

Configures automatic retry on failure.

**Parameters:**
- `max_attempts`: Number of retry attempts (1-10, validated)
- `initial_delay`: Delay in seconds before first retry (> 0, validated)
- `**kwargs`: Additional retry configuration (backoff_factor, max_delay, etc.)

**Validation:**
- `max_attempts` must be 1-10 (raises `ValueError`)
- `initial_delay` must be > 0 (raises `ValueError`)

**Examples:**
```python
# Conservative retry
agent.with_retry(max_attempts=3, initial_delay=1.0)

# Aggressive with backoff
agent.with_retry(
    max_attempts=5,
    initial_delay=0.5,
    backoff_factor=2.0,  # 0.5s, 1s, 2s, 4s, 8s
    max_delay=10.0       # Cap at 10 seconds
)

# Quick retries for flaky APIs
agent.with_retry(max_attempts=10, initial_delay=0.1)
```

**Applies to:**
- LLM API calls
- Tool executions
- All ExecutionEvent operations

---

### 4. Performance & Caching

#### `as_cacheable(ttl: int | None = None)`

Enables result caching.

**Parameters:**
- `ttl`: Time-to-live in seconds (None = indefinite, ≥ 0, validated)

**Validation:**
- Raises `ValueError` if ttl < 0

**TTL Options:**
- `None`: Cache forever (until manual clear)
- `0`: Cache with no automatic expiration
- `> 0`: Cache expires after N seconds

**Cache Key:**
- Based on: agent name, input data, context
- Identical inputs return cached result

**Examples:**
```python
# Short-lived cache (user session)
agent.as_cacheable(ttl=300)  # 5 minutes

# Long-lived cache (static data)
agent.as_cacheable(ttl=86400)  # 24 hours

# Indefinite cache (reference data)
agent.as_cacheable(ttl=None)

# No expiration
agent.as_cacheable(ttl=0)
```

**Best Practices:**
- Cache read-only operations
- Don't cache time-sensitive data
- Consider cache invalidation strategy

---

#### `as_artifact()`

Marks agent output as artifact for separate storage.

**Use Cases:**
- Large outputs (reports, documents)
- Generated content (images, files)
- Separate concerns (traces vs. artifacts)

**Examples:**
```python
# Large report generation
report_generator.as_artifact()

# Document synthesis
doc_writer.as_artifact()

# Generated code
code_generator.as_artifact()
```

**Behavior:**
- Result stored separately from execution trace
- Artifact ID returned in result metadata
- Retrieve via artifact storage system

---

### 5. Tool Configuration

For `ToolExecutorAgent` only.

#### `with_tools(tools: list[Tool])`

Sets or replaces available tools.

**Examples:**
```python
executor.with_tools([SearchTool(), AnalyzeTool()])
```

---

#### `with_tool_choice(tool_name: str)`

Forces LLM to call specific tool (for structured output).

**Examples:**
```python
# Force JSON extraction tool
extractor.with_tool_choice("ExtractJSON")
```

---

#### `with_first_result_only()`

Stops after first successful tool execution.

**Use Cases:**
- Racing multiple tools for fastest result
- Fallback strategies (try tool A, then B, then C)

**Examples:**
```python
# Try multiple search sources, use first
search_executor.with_first_result_only()
```

---

#### `with_aggregation(strategy: AggregationStrategy, separator: str = "\\n\\n")`

Sets result aggregation strategy.

**Strategies:**
- `DICT` (default): Return `ToolExecutionResult` (dict-like)
- `LIST`: Return ordered list `[result1, result2, ...]`
- `CONCATENATE`: Return joined string `"result1{sep}result2..."`
- `FIRST`: Return only first result

**Parameters:**
- `strategy`: One of `AggregationStrategy` enum values
- `separator`: String separator for CONCATENATE (default: double newline)

**Examples:**
```python
from rh_agents.core.types import AggregationStrategy

# List for iteration
executor.with_aggregation(AggregationStrategy.LIST)

# String for reports
executor.with_aggregation(
    AggregationStrategy.CONCATENATE,
    separator="\n\n---\n\n"
)

# First only
executor.with_aggregation(AggregationStrategy.FIRST)
```

See [BUILDERS_GUIDE.md](BUILDERS_GUIDE.md#result-aggregation-strategies) for detailed examples.

---

## Configuration Patterns

### Pattern 1: Production-Ready Agent

```python
agent = (
    await StructuredAgent.from_model(
        name="ProductionAgent",
        llm=llm,
        input_model=Request,
        output_model=Response,
        system_prompt="Process user requests"
    )
    .with_temperature(0.3)          # Low randomness
    .with_max_tokens(2000)          # Reasonable limit
    .with_model('gpt-4o')           # Best quality
    .with_error_strategy(ErrorStrategy.RETURN_NONE)  # Graceful failures
    .with_retry(max_attempts=3, initial_delay=1.0)   # Retry transient errors
    .as_cacheable(ttl=300)          # Cache for 5 minutes
)
```

### Pattern 2: Cost-Optimized Agent

```python
agent = (
    await CompletionAgent.from_prompt(...)
    .with_temperature(0.7)
    .with_max_tokens(500)           # Shorter responses
    .with_model('gpt-4o-mini')      # Cheaper model
    .as_cacheable(ttl=3600)         # Aggressive caching
)
```

### Pattern 3: Fault-Tolerant Agent

```python
agent = (
    await ToolExecutorAgent.from_tools(...)
    .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)
    .with_retry(
        max_attempts=5,
        initial_delay=0.5,
        backoff_factor=2.0,
        max_delay=10.0
    )
    .with_first_result_only()       # Fast failure recovery
)
```

### Pattern 4: Dynamic Context Agent

```python
async def build_context_prompt(input_data, context, state):
    # Get prior results
    prior = state.get_steps_result(['step1', 'step2'])
    
    # Build dynamic prompt
    return f"""
Process: {input_data.task_type}
Input: {input_data.content}

Prior Results:
{prior}

Context: {context}
"""

agent = (
    await StructuredAgent.from_model(...)
    .with_system_prompt_builder(build_context_prompt)
    .with_temperature(0.8)
)
```

---

## Validation Summary

All configuration methods validate parameters at configuration time (fail-fast):

| Method | Validation | Error |
|--------|-----------|-------|
| `with_temperature()` | 0.0 ≤ value ≤ 2.0 | ValueError |
| `with_max_tokens()` | 1 ≤ value ≤ 128000 | ValueError |
| `with_retry()` | 1 ≤ max_attempts ≤ 10, initial_delay > 0 | ValueError |
| `as_cacheable()` | ttl ≥ 0 or None | ValueError |

**Benefits:**
- Catch configuration errors early
- Clear error messages
- No runtime surprises

---

## See Also

- [BUILDERS_GUIDE.md](BUILDERS_GUIDE.md) - Complete builder pattern guide
- [builder_basic.py](../examples/builder_basic.py) - Basic usage examples
- [Phase 3 Completion](../PHASE3_COMPLETE.md) - Implementation details
