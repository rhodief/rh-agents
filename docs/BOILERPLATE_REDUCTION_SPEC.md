# Boilerplate Reduction Specification

**Version:** 1.3  
**Date:** February 18, 2026  
**Status:** ✅ Ready for Implementation - All Decisions Finalized

---

## 📋 Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Refactoring Proposal](#2-refactoring-proposal)
3. [Design Decisions](#3-design-decisions)
4. [Additional Suggestions](#4-additional-suggestions)
5. [Complete Pipeline Example](#5-complete-pipeline-example)
6. [Builder API Documentation](#6-builder-api-documentation)
7. [Implementation Plan](#7-implementation-plan)
8. [Implementation Readiness Verification](#8-implementation-readiness-verification)
9. [Appendix](#9-appendix)

---

## 1. Current State Analysis

### 1.1 Overview

After extensive use of the `rh-agents` package, we've identified that **most agent implementations follow repetitive patterns** with significant boilerplate code. This analysis examines the current state and identifies opportunities for simplification.

### 1.2 Common Patterns Identified

Based on `index.py`, `agents.py`, and the core architecture, we've identified **four primary usage patterns**:

#### Pattern 1: ModelCall
**Purpose:** Force LLM to return structured data using tool choice

**Current Implementation:**
```python
class DoctrineReceverAgent(Agent):
    def __init__(self, llm: LLM, tools: Union[list[Tool], None] = None):
        INTENT_PARSER_PROMPT = '''System prompt here...'''
        
        async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Union[Doctrine, Message]:
            llm_event = ExecutionEvent(actor=llm)
            
            # Construct LLM request
            llm_input = OpenAIRequest(
                system_message=INTENT_PARSER_PROMPT + f'\nContexto: {context}',
                prompt=input_data.content,
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                temperature=1,
                tools=ToolSet(tools=tools if tools else []),
                tool_choice={"type": "function", "function": {"name": "DoctrineTool"}}
            )
            
            # Execute LLM
            execution_result = await llm_event(llm_input, context, execution_state)
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            
            result = execution_result.result
            if result.is_content:
                return Message(content=result.content, author=AuthorType.ASSISTANT)
            
            if not (result.is_tool_call and result.tools and result.tools[0]):
                raise Exception("LLM did not return a valid tool call for DoctrineTool.")
            
            # Parse tool call output
            tool_call = result.tools[0]
            return Doctrine.model_validate_json(tool_call.arguments)
        
        super().__init__(
            name="DoctrineReceverAgent",
            description=INTENT_PARSER_PROMPT,
            input_model=Message,
            output_model=Doctrine,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools=tools) if tools else ToolSet(tools=[]),
            is_artifact=True,
            cacheable=True
        )
```

**Boilerplate Count:** ~40 lines of repetitive code
- ExecutionEvent creation
- OpenAIRequest construction
- Error handling
- Tool call extraction
- JSON validation

#### Pattern 2: CompletionCall
**Purpose:** Simple LLM call expecting only text content

**Current Implementation:**
```python
class SimpleCompletionAgent(Agent):
    def __init__(self, llm: LLM):
        PROMPT = '''You are an assistant...'''
        
        async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Message:
            llm_event = ExecutionEvent(actor=llm)
            
            llm_input = OpenAIRequest(
                system_message=PROMPT + f'\n{context}',
                prompt=input_data.content,
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                temperature=1
            )
            
            execution_result = await llm_event(llm_input, context, execution_state)
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            
            response = execution_result.result
            return Message(content=response.content or "", author=AuthorType.ASSISTANT)
        
        super().__init__(...)
```

**Boilerplate Count:** ~25 lines of repetitive code

#### Pattern 3: ModelTools
**Purpose:** LLM decides whether to call tools, execute them, and aggregate results

**Current Implementation (from StepExecutorAgent):**
```python
async def handler(input_data: DoctrineStep, context: str, execution_state: ExecutionState) -> StepResult:
    llm_event = ExecutionEvent(actor=llm)
    
    # Build context with dependencies
    dependencies_list = execution_state.get_steps_result(input_data.required_steps)
    dependencies_context = '\n'.join(dependencies_list) if dependencies_list else 'Nenhuma execução anterior.'
    system_context = STEP_EXECUTOR_PROMPT + f'\n{context}\n\nExecuções anteriores:\n{dependencies_context}'
    
    # LLM Call
    llm_input = OpenAIRequest(
        system_message=system_context,
        prompt=input_data.description,
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        temperature=1,
        tools=tool_set
    )
    execution_result = await llm_event(llm_input, context, execution_state)
    if not execution_result.ok or execution_result.result is None:
        raise Exception(f"LLM execution failed: {execution_result.erro_message}")
    
    response = execution_result.result
    all_outputs = []
    errors = []
    
    # Handle tool calls
    if response.is_tool_call:
        for tool_call in response.tools:
            tool = tool_set[tool_call.tool_name]
            if tool is None:
                errors.append(f"Tool '{tool_call.tool_name}' not found.")
                continue
            
            try:
                tool_event = ExecutionEvent(actor=tool)
                tool_input = tool.input_model.model_validate_json(tool_call.arguments)
                tool_result = await tool_event(tool_input, context, execution_state)
                
                if not tool_result.ok or tool_result.result is None:
                    errors.append(f"Tool '{tool_call.tool_name}' execution failed")
                else:
                    output = getattr(tool_result.result, 'output', tool_result.result)
                    all_outputs.append(str(output))
            except Exception as e:
                errors.append(f"Error in {tool_call.tool_name}: {str(e)}")
    else:
        all_outputs.append(response.content or "")
    
    # Aggregate results
    if errors and not all_outputs:
        return StepResult(
            step_index=input_data.index,
            result=ExecutionResult[str](result=None, ok=False, erro_message="; ".join(errors))
        )
    
    combined_output = "\n".join(all_outputs)
    if errors:
        combined_output += f"\nErrors: {'; '.join(errors)}"
    
    return StepResult(
        step_index=input_data.index,
        result=ExecutionResult[str](result=combined_output, ok=True)
    )
```

**Boilerplate Count:** ~55 lines of repetitive code
- LLM execution
- Tool call iteration
- Error aggregation
- Result combination

#### Pattern 4: ToolsOnly
**Purpose:** Execute tool handlers directly without LLM involvement

**Current Implementation (hypothetical):**
```python
class DirectToolAgent(Agent):
    def __init__(self, tool: Tool):
        async def handler(input_data: BaseModel, context: str, execution_state: ExecutionState) -> Tool_Result:
            tool_event = ExecutionEvent(actor=tool)
            tool_result = await tool_event(input_data, context, execution_state)
            
            if not tool_result.ok or tool_result.result is None:
                raise Exception(f"Tool execution failed: {tool_result.erro_message}")
            
            return tool_result.result
        
        super().__init__(...)
```

**Boilerplate Count:** ~15 lines of repetitive code

### 1.3 Boilerplate Analysis Summary

| Pattern | Lines of Boilerplate | Key Repetitions |
|---------|---------------------|-----------------|
| ModelCall | ~40 | ExecutionEvent creation, OpenAIRequest construction, error handling, tool call extraction, JSON validation |
| CompletionCall | ~25 | ExecutionEvent creation, OpenAIRequest construction, error handling, content extraction |
| ModelTools | ~55 | LLM execution, tool iteration, error aggregation, result combination |
| ToolsOnly | ~15 | ExecutionEvent creation, error handling |

**Total estimated reduction opportunity:** ~135 lines per typical multi-agent application

### 1.4 Pain Points Identified

1. **Repetitive ExecutionEvent Creation:** Every handler creates `ExecutionEvent(actor=...)` manually
2. **OpenAIRequest Verbosity:** Same parameters repeated across agents (model, max_tokens, temperature)
3. **Manual Error Handling:** Every handler checks `ok` and `result is None`
4. **Tool Call Extraction:** Parsing and validating tool calls is verbose
5. **Tool Execution Loop:** Iterating and executing tools is repeated code
6. **Result Transformation:** Converting LLM_Result → domain model requires boilerplate
7. **Parameter Passing:** `context` and `execution_state` threaded through every call

### 1.5 Current Strengths to Maintain

✅ **Type Safety:** Pydantic models provide excellent validation  
✅ **Event System:** ExecutionEvent enables monitoring and state recovery  
✅ **Flexibility:** Current system allows any custom logic  
✅ **Caching & Retry:** Actor-level configuration is powerful  
✅ **Testability:** Handlers are testable in isolation

---

## 2. Refactoring Proposal

### 2.1 Core Idea: Builder Pattern + Smart Defaults

Introduce **lightweight builder/factory classes** that encapsulate common patterns while maintaining the existing architecture.

**Key Principles:**
1. **No Breaking Changes:** Existing code continues to work
2. **Opt-in Simplification:** Users choose when to use builders
3. **Preserve Flexibility:** Advanced users can still use raw handlers
4. **Type Safety First:** Leverage Pydantic for validation

### 2.2 Proposed API Design

#### Pattern 1: ModelCall → `StructuredAgent.from_model()`

**Before (40 lines):**
```python
class DoctrineReceverAgent(Agent):
    def __init__(self, llm: LLM, tools: list[Tool]):
        PROMPT = '''...'''
        async def handler(input_data: Message, ...) -> Doctrine:
            llm_event = ExecutionEvent(actor=llm)
            llm_input = OpenAIRequest(...)
            execution_result = await llm_event(...)
            # ... 30 more lines of boilerplate
        super().__init__(...)
```

**After (8 lines):**
```python
doctrine_agent = StructuredAgent.from_model(
    name="DoctrineReceverAgent",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="Analisa o pedido do usuário e gera um plano estruturado...",
    tools=[DoctrineTool()],
    tool_choice="DoctrineTool",  # Forces this tool to be called
    is_artifact=True
)
```

**How it works:**
- Internally creates a Tool from `output_model` schema
- Configures `tool_choice` to force structured output
- Extracts and validates JSON from tool call
- Returns typed instance of `output_model`

#### Pattern 2: CompletionCall → `CompletionAgent.from_prompt()`

**Before (25 lines):**
```python
class ReviewerAgent(Agent):
    def __init__(self, llm: LLM):
        PROMPT = '''...'''
        async def handler(input_data: Doctrine, ...) -> Message:
            llm_event = ExecutionEvent(actor=llm)
            # ... boilerplate
        super().__init__(...)
```

**After (6 lines):**
```python
reviewer_agent = CompletionAgent.from_prompt(
    name="ReviewerAgent",
    llm=llm,
    input_model=Doctrine,
    output_model=Message,
    system_prompt="Você é um revisor especializado..."
)
```

**How it works:**
- Simple LLM call, no tools
- Extracts `content` from LLM_Result
- Returns Message with content

#### Pattern 3: ModelTools → `ToolExecutorAgent.from_tools()`

**Before (55 lines):**
```python
class StepExecutorAgent(Agent):
    def __init__(self, llm: LLM, tools: list[Tool]):
        async def handler(input_data: DoctrineStep, ...) -> StepResult:
            # ... 50+ lines of tool execution logic
        super().__init__(...)
```

**After (10 lines):**
```python
step_executor_agent = ToolExecutorAgent.from_tools(
    name="StepExecutorAgent",
    llm=llm,
    input_model=DoctrineStep,
    output_model=StepResult,
    system_prompt="Você é um executor de passos...",
    tools=[ListPecasTool(), GetTextoPecaTool()],
    aggregation_strategy="concatenate",  # or "list", "dict", "first"
    include_errors=True
)
```

**How it works:**
- LLM decides which tools to call
- Executes all tool calls in parallel (optional)
- Aggregates results based on strategy
- Handles errors gracefully

#### Pattern 4: ToolsOnly → `DirectToolAgent.from_tool()`

**Before (15 lines):**
```python
class DirectToolAgent(Agent):
    def __init__(self, tool: Tool):
        async def handler(...):
            # boilerplate
        super().__init__(...)
```

**After (4 lines):**
```python
direct_agent = DirectToolAgent.from_tool(
    name="DirectAgent",
    tool=my_tool
)
```

**How it works:**
- Bypasses LLM entirely
- Directly executes tool handler
- Returns tool result

### 2.3 Advanced: Transform Functions (Optional)

For cases where output transformation is needed:

```python
async def transform_doctrine_to_message(doctrine: Doctrine) -> Message:
    """Custom transformation logic"""
    return Message(content=f"Plan: {doctrine.goal}", author=AuthorType.ASSISTANT)

agent = StructuredAgent.from_model(
    name="DoctrineAgent",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="...",
    tools=[DoctrineTool()],
    tool_choice="DoctrineTool",
    transform=transform_doctrine_to_message  # Optional post-processing
)
```

### 2.4 Example: Refactored index.py

**Before (180+ lines):**
```python
# Full implementation of DoctrineReceverAgent, StepExecutorAgent, ReviewerAgent
# ... 180 lines of boilerplate
```

**After (40 lines):**
```python
from rh_agents import OpenAILLM
from rh_agents.builders import StructuredAgent, CompletionAgent, ToolExecutorAgent

llm = OpenAILLM()
tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]

# Receiver: Forces structured output
doctrine_receiver = StructuredAgent.from_model(
    name="DoctrineReceiver",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="Analisa o pedido do usuário e gera um plano estruturado...",
    tools=[DoctrineTool()],
    tool_choice="DoctrineTool",
    is_artifact=True,
    cacheable=True
)

# Executor: Runs tools based on LLM decisions
step_executor = ToolExecutorAgent.from_tools(
    name="StepExecutor",
    llm=llm,
    input_model=DoctrineStep,
    output_model=StepResult,
    system_prompt="Você é um executor de passos...",
    tools=[ListPecasTool(), GetTextoPecaTool()],
    aggregation_strategy="concatenate"
)

# Reviewer: Simple completion
reviewer = CompletionAgent.from_prompt(
    name="Reviewer",
    llm=llm,
    input_model=Doctrine,
    output_model=Message,
    system_prompt="Você é um revisor especializado..."
)

# Orchestration (unchanged)
omni_agent = OmniAgent(
    receiver_agent=doctrine_receiver,
    step_executor_agent=step_executor,
    reviewer_agent=reviewer
)
```

**Lines saved:** ~140 lines (~78% reduction)

---

## 3. Design Decisions

### Decision 3.1: Builder Pattern with Hybrid API ✅

**Chosen Approach:** Builder/Factory Methods with mandatory parameters as keyword args and optional configuration via method chaining.

**API Design:**
```python
# Mandatory parameters in factory method
agent = await StructuredAgent.from_model(
    name="DoctrineAgent",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="..."
)

# Optional configuration via chaining
agent = (
    await StructuredAgent.from_model(
        name="DoctrineAgent",
        llm=llm,
        input_model=Message,
        output_model=Doctrine,
        system_prompt="..."
    )
    .with_tools([DoctrineTool()])
    .with_tool_choice("DoctrineTool")
    .as_artifact()
    .as_cacheable()
    .with_retry(max_attempts=3)
)
```

**Benefits:**
- Mandatory parameters are clear and explicit
- Optional features don't clutter the initial call
- Easy to discover options via IDE autocomplete
- Maintains backward compatibility
- Composition over inheritance

---

### Decision 3.2: Error Handling Strategy ✅

**Chosen Approach:** Configurable error strategy with "raise" as default.

**Available Strategies:**

```python
from enum import Enum

class ErrorStrategy(str, Enum):
    RAISE = "raise"              # Raise exception immediately (default)
    RETURN_NONE = "return_none"  # Return ExecutionResult with ok=False
    LOG_AND_CONTINUE = "log"     # Log error and return partial results
    SILENT = "silent"            # Suppress errors, return None
```

**Usage:**
```python
# Default behavior - raises on error (integrates with retry)
agent = await StructuredAgent.from_model(
    name="DoctrineAgent",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="..."
)

# Custom error handling
agent = (
    await StructuredAgent.from_model(...)
    .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)
)

# For optional operations
agent = (
    await StructuredAgent.from_model(...)
    .with_error_strategy(ErrorStrategy.RETURN_NONE)
)
```

**Strategy Behaviors:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `RAISE` (default) | Throws exception | Critical operations, leverages retry mechanism |
| `RETURN_NONE` | Returns ExecutionResult(ok=False) | Caller handles errors programmatically |
| `LOG_AND_CONTINUE` | Logs error, returns partial data | Best-effort operations |
| `SILENT` | Suppresses all errors | Fire-and-forget operations |

---

### Decision 3.3: Tool Execution Strategy ✅

**Chosen Approach:** Parallel execution by default (no configuration needed).

**Implementation:**
```python
# All tools execute in parallel using asyncio.gather
tasks = [execute_tool(tc) for tc in response.tools]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Benefits:**
- ⚡ Faster for independent tools (most common case)
- 🔄 Better resource utilization
- 🎯 Aligns with `parallel` feature in core
- 📊 Minimal overhead even for single tool call

**Design Note:** Tools are assumed to be independent by design. If tools have dependencies, they should be called in separate agent steps.

---

### Decision 3.4: Result Aggregation Strategy ✅

**Chosen Approach:** Dictionary by tool name (default), with optional "first" mode.

**Result Structure:**
```python
from pydantic import BaseModel
from typing import Dict, Any

class ToolExecutionResult(BaseModel):
    results: Dict[str, Any]  # Keyed by tool name
    execution_order: list[str]  # Track which tools ran
    errors: Dict[str, str]  # Tool name -> error message
    
    def get(self, tool_name: str) -> Any:
        """Get result for specific tool."""
        return self.results.get(tool_name)
    
    def first(self) -> Any:
        """Get first result (by execution order)."""
        if self.execution_order:
            return self.results[self.execution_order[0]]
        return None
```

**Usage:**
```python
# Default: Execute all tools, return dictionary
agent = await ToolExecutorAgent.from_tools(
    name="Executor",
    llm=llm,
    input_model=DoctrineStep,
    output_model=StepResult,
    system_prompt="...",
    tools=[ListPecasTool(), GetTextoPecaTool()]
)

result = await agent(step, "", state)
list_result = result.get("lista_pecas_por_tipo")
text_result = result.get("get_texto_peca")

# "First" mode: Stop after first successful tool execution
agent = (
    await ToolExecutorAgent.from_tools(...)
    .with_first_result_only()
)

result = await agent(step, "", state)
first_output = result.first()  # Returns first successful execution
```

**Benefits:**
- 🗂️ Named access to results (no order dependency)
- 🎯 Tools are independent by design
- ⚡ "First" mode for fallback patterns (try tools until one succeeds)
- 📊 Execution order preserved for debugging

---

### Decision 3.5: Context Building ✅

**Chosen Approach:** Append to system prompt by default, with optional context transformer.

**Default Behavior:**
```python
# If context is non-empty, automatically append to system prompt
full_system_prompt = f"{base_system_prompt}\n\nContext: {context}" if context else base_system_prompt
```

**Custom Context Transformation:**
```python
# Optional: Override default context formatting
agent = (
    await StructuredAgent.from_model(
        name="DoctrineAgent",
        llm=llm,
        input_model=Message,
        output_model=Doctrine,
        system_prompt="You are an assistant..."
    )
    .with_context_transform(lambda ctx: f"\n\n=== EXECUTION CONTEXT ===\n{ctx}\n===\n")
)
```

**Benefits:**
- ✅ Works out of the box - context automatically available to LLM
- ✅ Only appends if context is non-empty (no noise)
- ✅ Advanced users can customize formatting via lambda
- 📝 Placed after system prompt (doesn't override instructions)

---

### Decision 3.6: System Prompt Templating ✅

**Chosen Approach:** Simple string by default, with optional callback function for advanced cases.

**Default - Simple String:**
```python
agent = await StructuredAgent.from_model(
    name="Agent",
    llm=llm,
    input_model=Message,
    output_model=Doctrine,
    system_prompt="You are an assistant. Analyze the user request."
)
# Context automatically appended if present (see Decision 3.5)
```

**Advanced - Dynamic Prompt Builder:**
```python
# For cases requiring dynamic prompts based on state/input
async def build_prompt(input_data: Message, context: str, execution_state: ExecutionState) -> str:
    processo_id = execution_state.get_metadata("processo_id")
    user_role = execution_state.get_metadata("user_role")
    return f"""You are analyzing process {processo_id} as {user_role}.
    
    Guidelines:
    - Be concise
    - Focus on legal arguments
    
    Context: {context}
    """

agent = (
    await StructuredAgent.from_model(
        name="Agent",
        llm=llm,
        input_model=Message,
        output_model=Doctrine
        # No system_prompt here
    )
    .with_system_prompt_builder(build_prompt)
)
```

**Benefits:**
- 📝 Simple string for 90% of use cases
- 🔧 Callback function for dynamic prompts
- 💡 Full access to input, context, and state
- ⚡ No template engine complexity

---

### Decision 3.7: LLM Configuration ✅

**Chosen Approach:** Hybrid - inherit from LLM with optional per-agent overrides.

**Default - Inherit from LLM:**
```python
# LLM has default configuration
llm = OpenAILLM(model="gpt-4o", temperature=1.0, max_tokens=2500)

# Agent inherits all settings
agent = await StructuredAgent.from_model(
    name="Agent",
    llm=llm,  # Uses gpt-4o, temp=1.0, max_tokens=2500
    input_model=Message,
    output_model=Doctrine,
    system_prompt="..."
)
```

**Override Specific Parameters:**
```python
# Override just what you need
agent = (
    await StructuredAgent.from_model(
        name="Agent",
        llm=llm,  # Base configuration
        input_model=Message,
        output_model=Doctrine,
        system_prompt="..."
    )
    .with_temperature(0.5)  # Override temperature
    .with_max_tokens(1000)  # Override max tokens
    # model still inherited from llm
)

# Or override model
agent = (
    await StructuredAgent.from_model(...)
    .with_model("gpt-4o-mini")  # Use cheaper model
)
```

**Benefits:**
- 🎯 Sensible defaults (DRY principle)
- 🔧 Granular control when needed
- 📊 Easy to swap LLMs globally
- ⚙️ Per-agent optimization

---

### Decision 3.8: Module Organization ✅

**Chosen Approach:** New `rh_agents.builders` module.

**Import Pattern:**
```python
from rh_agents.builders import (
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
```

**Module Structure:**
```
rh_agents/
├── builders.py          # Builder classes
├── templates.py         # Pre-built templates (Phase 5)
├── agents.py            # Existing agents (unchanged)
├── core/
│   ├── actors.py
│   └── ...
```

**Benefits:**
- 📌 Clear namespace: "these are builder utilities"
- 📦 Doesn't clutter top-level
- 🔗 Still part of main package
- 🚀 Future-proof: room for more builders

---

## 4. Additional Suggestions

### Suggestion 4.1: Pre-built Agent Templates ✅

**Chosen Approach:** Template library with 5-10 essential templates.

**Template Module Structure:**
```python
from rh_agents.templates import (
    create_json_extractor,
    create_summarizer,
    create_classifier,
    create_qa_agent,
    create_validator,
    create_ranker,
    create_translator
)
```

**Example Template:**
```python
# In rh_agents/templates.py

async def create_json_extractor(
    llm: LLM,
    output_model: type[BaseModel],
    name: str = "JsonExtractor",
    additional_instructions: str = ""
) -> Agent:
    """Create an agent that extracts structured data from text."""
    system_prompt = f"""Extract structured information from the provided text.
    Return only the requested fields, nothing more.
    {additional_instructions}
    """
    
    return await StructuredAgent.from_model(
        name=name,
        llm=llm,
        input_model=Message,
        output_model=output_model,
        system_prompt=system_prompt
    ).as_cacheable()


async def create_summarizer(
    llm: LLM,
    max_length: int = 500,
    style: str = "concise",
    name: str = "Summarizer"
) -> Agent:
    """Create a text summarization agent."""
    system_prompt = f"""Summarize the provided text in a {style} style.
    Maximum length: {max_length} words.
    Focus on key points and main ideas.
    """
    
    return await CompletionAgent.from_prompt(
        name=name,
        llm=llm,
        input_model=Message,
        output_model=Message,
        system_prompt=system_prompt
    ).with_max_tokens(max_length * 2)
```

**Usage:**
```python
# Quick setup with templates
extractor = await create_json_extractor(
    llm=llm,
    output_model=Doctrine,
    additional_instructions="Focus on legal arguments."
)

summarizer = await create_summarizer(
    llm=llm,
    max_length=300,
    style="technical"
)
```

---

### Suggestion 4.2: Debug/Inspection Mode ✅

**Chosen Approach:** No special support - use existing EventPrinter.

**Rationale:** The existing event system with `EventPrinter` already provides comprehensive debugging:

```python
from rh_agents import EventPrinter, EventBus, ExecutionState

# Existing debugging infrastructure
printer = EventPrinter(show_timestamp=True, show_address=True)
bus = EventBus()
bus.subscribe(printer)
state = ExecutionState(event_bus=bus)

# All builder-generated agents emit events automatically
agent = await StructuredAgent.from_model(...)
result = await agent(input_data, "", state)

# EventPrinter shows:
# - LLM calls with tokens
# - Tool executions
# - Timing information
# - Errors and retries
```

---

### Suggestion 4.3: Async Builder Pattern ✅

**Chosen Approach:** Async-only builders.

**Implementation:**
```python
# All builders are async
agent = await StructuredAgent.from_model(...)
agent = await CompletionAgent.from_prompt(...)
agent = await ToolExecutorAgent.from_tools(...)
```

**Rationale:**
- Consistent with existing async architecture
- Allows for future validation if needed
- Python async/await is standard in the codebase

---

### Suggestion 4.4: Type Inference ✅

**Chosen Approach:** Keep models explicit (no inference).

**Implementation:**
```python
# Always explicit
agent = await StructuredAgent.from_model(
    name="Agent",
    llm=llm,
    input_model=Message,  # Explicit
    output_model=Doctrine,  # Explicit
    system_prompt="..."
)
```

**Rationale:**
- Explicit is better than implicit
- Type inference is complex and fragile
- Still achieves 80% boilerplate reduction
- Can explore inference in future versions

---

### Suggestion 4.5: Chainable Builders ✅

**Chosen Approach:** Keyword args for mandatory, chainable methods for optional.

**API Design:**
```python
# Mandatory parameters as keyword arguments
agent = await StructuredAgent.from_model(
    name="DoctrineAgent",      # Required
    llm=llm,                    # Required
    input_model=Message,        # Required
    output_model=Doctrine,      # Required
    system_prompt="..."         # Required
)

# Optional configuration via chaining
agent = (
    await StructuredAgent.from_model(
        name="DoctrineAgent",
        llm=llm,
        input_model=Message,
        output_model=Doctrine,
        system_prompt="..."
    )
    .with_tools([DoctrineTool()])              # Optional
    .with_tool_choice("DoctrineTool")          # Optional
    .with_temperature(0.7)                     # Optional
    .with_max_tokens(3000)                     # Optional
    .with_context_transform(custom_fn)         # Optional
    .with_system_prompt_builder(builder_fn)    # Optional
    .with_error_strategy(ErrorStrategy.LOG)    # Optional
    .with_retry(max_attempts=3, initial_delay=1.0)  # Optional
    .as_artifact()                             # Optional
    .as_cacheable()                            # Optional
)
```

**Available Chainable Methods:**

| Method | Description |
|--------|-------------|
| `.with_tools(tools)` | Add tools for LLM to call |
| `.with_tool_choice(name)` | Force specific tool (for StructuredAgent) |
| `.with_temperature(t)` | Override LLM temperature |
| `.with_max_tokens(n)` | Override max tokens |
| `.with_model(name)` | Override LLM model |
| `.with_context_transform(fn)` | Custom context formatting |
| `.with_system_prompt_builder(fn)` | Dynamic prompt builder |
| `.with_error_strategy(strategy)` | Error handling strategy |
| `.with_retry(...)` | Retry configuration |
| `.with_first_result_only()` | Stop after first tool succeeds (ToolExecutorAgent) |
| `.as_artifact()` | Mark as artifact producer |
| `.as_cacheable(ttl=None)` | Enable caching |

**Benefits:**
- ✅ Clear distinction: mandatory vs optional
- ✅ No guessing which parameters are required
- ✅ IDE autocomplete for chaining methods
- ✅ Readable - one option per line
- ✅ Easy to discover optional features

---

## 5. Complete Pipeline Example

This section demonstrates a complete real-world pipeline implementation using the new builder design, revealing any additional decisions that may be needed.

### 5.1 The Use Case: Legal Document Analysis

**Goal:** Analyze legal documents (decisions and appeals) to generate a comparative report.

**Pipeline Steps:**
1. **DoctrineReceiver** - Parse user request into structured execution plan
2. **StepExecutor** - Execute each step (fetch documents, analyze content)
3. **Reviewer** - Generate final comparative report

### 5.2 Current Implementation (Before Builders)

```python
# examples/index.py - BEFORE (180+ lines)
import asyncio
from db import DOC_LIST, DOCS
from rh_agents.agents import OpenAILLM
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.openai import OpenAIRequest
from rh_agents.models import Message, AuthorType
from pydantic import BaseModel, Field

MODEL = 'gpt-4o'
MAX_TOKENS = 2500

class Doctrine(BaseModel):
    goal: str
    steps: list[DoctrineStep]
    constraints: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)

class DoctrineStep(BaseModel):    
    index: int
    description: str
    feasible: bool
    required_steps: list[int] = Field(default_factory=list)

class StepResult(BaseModel):
    step_index: int
    result: ExecutionResult[str]

# ... 150+ lines of agent definitions (omitted for brevity)
# DoctrineReceverAgent, StepExecutorAgent, ReviewerAgent classes

if __name__ == "__main__":
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools[1:])
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    msg = 'Faça um relatório com o resumo combinado dos óbices jurídicos...'
    message = Message(content=msg, author=AuthorType.USER)
    
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    state = ExecutionState(event_bus=bus)
    
    async def main():
        result = await ExecutionEvent[Message](
            actor=omni_agent,
            retry_config=RetryConfig(max_attempts=3, initial_delay=1.0)
        )(message, "", state)
        
        printer.print_summary()
        print("Final Result:")
        print(result.result)
    
    asyncio.run(main())
```

### 5.3 New Implementation (With Builders)

```python
# examples/index_with_builders.py - AFTER (~60 lines)
import asyncio
from db import DOC_LIST, DOCS
from rh_agents import OpenAILLM, EventPrinter, EventBus, ExecutionState
from rh_agents.builders import StructuredAgent, CompletionAgent, ToolExecutorAgent
from rh_agents.models import Message, AuthorType, Doctrine, DoctrineStep, StepResult
from rh_agents.core.retry import RetryConfig
from rh_agents.core.events import ExecutionEvent
from custom_tools import DoctrineTool, ListPecasTool, GetTextoPecaTool

async def build_step_context(
    input_data: DoctrineStep,
    context: str,
    execution_state: ExecutionState
) -> str:
    """Build context for step executor with dependencies."""
    dependencies = execution_state.get_steps_result(input_data.required_steps)
    deps_text = '\\n'.join(dependencies) if dependencies else 'Nenhuma execução anterior.'
    return f"Processo: 123456789\\n\\nExecuções anteriores:\\n{deps_text}"

async def build_reviewer_context(
    input_data: Doctrine,
    context: str,
    execution_state: ExecutionState
) -> str:
    """Build comprehensive context for reviewer."""
    all_results = execution_state.get_all_steps_results()
    results_text = "\\n\\n".join([
        f"Resultado da Etapa {idx}:\\n{result}"
        for idx, result in all_results.items()
    ]) if all_results else "Nenhum resultado disponível."
    
    return f"""
OBJETIVO GERAL: {input_data.goal}

DIRETRIZES: {', '.join(input_data.guidelines) if input_data.guidelines else 'Nenhuma'}

RESTRIÇÕES: {', '.join(input_data.constraints) if input_data.constraints else 'Nenhuma'}

RESULTADOS DAS ETAPAS:
{results_text}
"""

async def main():
    # Initialize LLM and tools
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    
    # 1. Doctrine Receiver: Parse user intent into structured plan
    doctrine_receiver = (
        await StructuredAgent.from_model(
            name="DoctrineReceiver",
            llm=llm,
            input_model=Message,
            output_model=Doctrine,
            system_prompt="""Analisa o pedido do usuário e gera um plano estruturado.
            Cada passo deve conter uma única ação clara e objetiva.
            Use tool DoctrineTool para estruturar a resposta."""
        )
        .with_tools([DoctrineTool()])
        .with_tool_choice("DoctrineTool")  # Force structured output
        .as_artifact()  # Store as artifact
        .as_cacheable()  # Cache doctrine generation
    )
    
    # 2. Step Executor: Execute individual steps with tools
    step_executor = (
        await ToolExecutorAgent.from_tools(
            name="StepExecutor",
            llm=llm,
            input_model=DoctrineStep,
            output_model=StepResult,
            system_prompt="""Você é um executor de passos.
            Execute o passo fornecido usando as ferramentas disponíveis.
            Se já tiver informações no contexto, não chame ferramentas desnecessariamente.""",
            tools=[ListPecasTool(), GetTextoPecaTool()]
        )
        .with_system_prompt_builder(build_step_context)  # Dynamic context
        .with_temperature(0.9)  # Slightly more focused
    )
    
    # 3. Reviewer: Generate final report from all step results
    reviewer = (
        await CompletionAgent.from_prompt(
            name="Reviewer",
            llm=llm,
            input_model=Doctrine,
            output_model=Message,
            system_prompt="""Você é um revisor especializado em análise jurídica.
            Elabore uma resposta final completa e bem estruturada.
            Sintetize as informações coletadas em um relatório coeso."""
        )
        .with_system_prompt_builder(build_reviewer_context)  # Dynamic context
        .with_max_tokens(3000)  # Longer output for report
    )
    
    # 4. Orchestrator: Coordinates the pipeline
    async def orchestrate(message: Message, context: str, state: ExecutionState) -> Message:
        """Orchestrate the complete pipeline."""
        # Step 1: Parse user intent
        doctrine_result = await ExecutionEvent(actor=doctrine_receiver)(message, "", state)
        if not doctrine_result.ok:
            raise Exception(f"Doctrine parsing failed: {doctrine_result.erro_message}")
        
        doctrine = doctrine_result.result
        if isinstance(doctrine, Message):  # Fallback to simple message
            return doctrine
        
        await state.log(f"Doctrine received: {doctrine.goal} with {len(doctrine.steps)} steps")
        
        # Step 2: Execute each step
        for step in doctrine.steps:
            if not step.feasible:
                raise Exception(f"Step {step.index} not feasible")
            
            step_result = await ExecutionEvent(
                actor=step_executor,
                tag=f'step_{step.index}-{len(doctrine.steps) - 1}'
            )(step, '', state)
            
            if not step_result.ok:
                raise Exception(f"Step {step.index} failed: {step_result.erro_message}")
            
            state.add_step_result(step.index, step_result.result)
        
        # Step 3: Generate final review
        await state.log("Iniciando Revisão Final...")
        review_result = await ExecutionEvent(
            actor=reviewer,
            tag='final_review'
        )(doctrine, '', state)
        
        if not review_result.ok:
            raise Exception(f"Review failed: {review_result.erro_message}")
        
        final_message = review_result.result
        print("\\n" + "═" * 60)
        print("📄 RELATÓRIO FINAL")
        print("═" * 60)
        print(final_message.content)
        print("═" * 60 + "\\n")
        
        return final_message
    
    # Setup event bus for monitoring
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    state = ExecutionState(event_bus=bus)
    
    # Execute pipeline
    user_message = Message(
        content='Faça um relatório com o resumo combinado dos óbices jurídicos...',
        author=AuthorType.USER
    )
    
    print(f"\\n{'═' * 60}")
    print(f"{'🚀 EXECUTION STARTED':^60}")
    print(f"{'═' * 60}\\n")
    
    result = await orchestrate(user_message, "", state)
    
    print(f"\\n{'═' * 60}")
    print(f"{'✅ EXECUTION FINISHED':^60}")
    print(f"{'═' * 60}\\n")
    
    printer.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
```

**Lines of code:** ~180 → ~60 (67% reduction)

### 5.4 Key Improvements Demonstrated

1. **✅ Mandatory vs Optional Clear:**
   - Required params: `name`, `llm`, `input_model`, `output_model`, `system_prompt`
   - Optional config via chaining: `.with_tools()`, `.as_cacheable()`, etc.

2. **✅ Type Safety Maintained:**
   - All models explicitly typed
   - Pydantic validation still works

3. **✅ Dynamic Context Building:**
   - `build_step_context()` - Injects step dependencies
   - `build_reviewer_context()` - Aggregates all results
   - Custom formatting via `with_system_prompt_builder()`

4. **✅ LLM Parameter Overrides:**
   - Base LLM has defaults
   - Step executor overrides temperature: `.with_temperature(0.9)`
   - Reviewer overrides max_tokens: `.with_max_tokens(3000)`

5. **✅ Tool Execution Pattern:**
   - `ToolExecutorAgent` handles parallel tool execution
   - Returns dictionary by tool name
   - Error handling built-in

6. **✅ Structured Output Pattern:**
   - `StructuredAgent` with `.with_tool_choice()` forces schema
   - Automatic JSON validation
   - Returns typed `Doctrine` object

7. **✅ No Breaking Changes:**
   - Still uses `ExecutionEvent` for retry/monitoring
   - Event bus integration unchanged
   - State recovery compatible

### 5.5 Alternative: Using Templates

```python
# Even simpler with templates
from rh_agents.templates import create_json_extractor, create_qa_agent

doctrine_receiver = await create_json_extractor(
    llm=llm,
    output_model=Doctrine,
    name="DoctrineReceiver",
    additional_instructions="Focus on legal workflow planning."
)

step_executor = await create_qa_agent(
    llm=llm,
    name="StepExecutor",
    tools=[ListPecasTool(), GetTextoPecaTool()],
    context_style="detailed"
)

reviewer = await create_summarizer(
    llm=llm,
    name="Reviewer",
    max_length=1000,
    style="technical"
)
```

### 5.6 Future Enhancement: Orchestrator Builder

**Note:** An `OrchestrationAgent` builder for common pipeline patterns has been identified as a potential enhancement. However, it will be **deferred to post-v1** to allow real-world usage patterns to inform the design.

**Rationale:**
- ✅ Manual orchestration (shown above) is already clean and works well
- ✅ Need to learn from actual usage patterns first
- ✅ Core builders provide immediate 67% code reduction
- ✅ Can be added later without breaking changes

**Implementation Plan:** Consider for Phase 7 (post-release) after gathering user feedback on common orchestration patterns.

---

## 6. Builder API Documentation

### 6.1 StructuredAgent - Docstring

```python
class StructuredAgent:
    """Builder for agents that force LLM to return structured data.
    
    Creates an agent that uses tool choice to guarantee structured output matching
    a Pydantic model. The LLM is forced to call a specific tool that validates
    and returns an instance of the output_model.
    
    Usage Pattern:
        Use when you need the LLM to return typed, validated data structures
        rather than free-form text. Perfect for parsing user input into workflows,
        extracting information, or any scenario requiring schema compliance.
    
    Examples:
        Basic usage with mandatory parameters:
        >>> agent = await StructuredAgent.from_model(
        ...     name="DoctrineParser",
        ...     llm=llm,
        ...     input_model=Message,
        ...     output_model=Doctrine,
        ...     system_prompt="Parse user request into structured plan."
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await StructuredAgent.from_model(
        ...         name="DoctrineParser",
        ...         llm=llm,
        ...         input_model=Message,
        ...         output_model=Doctrine,
        ...         system_prompt="Parse user request into structured plan."
        ...     )
        ...     .with_tools([DoctrineTool()])  # Add tools for LLM to choose from
        ...     .with_tool_choice("DoctrineTool")  # Force specific tool (structured output)
        ...     .with_temperature(0.7)  # Override LLM temperature
        ...     .with_max_tokens(3000)  # Override max tokens
        ...     .with_context_transform(lambda ctx: f"\\n\\nCONTEXT: {ctx}")  # Custom context formatting
        ...     .with_system_prompt_builder(dynamic_prompt_fn)  # Dynamic prompt generation
        ...     .with_error_strategy(ErrorStrategy.RAISE)  # Error handling strategy
        ...     .with_retry(max_attempts=3, initial_delay=1.0)  # Retry configuration
        ...     .as_artifact()  # Mark output as artifact for storage
        ...     .as_cacheable(ttl=3600)  # Enable caching with TTL
        ... )
    
    Factory Method:
        from_model(name, llm, input_model, output_model, system_prompt) -> StructuredAgent
            Create a new StructuredAgent with mandatory parameters.
            
            Args:
                name: Unique identifier for the agent
                llm: LLM instance to use for generation
                input_model: Pydantic model for input validation
                output_model: Pydantic model for output validation (enforced via tool choice)
                system_prompt: Base system prompt for the agent
            
            Returns:
                StructuredAgent instance ready for optional configuration
    
    Chainable Methods (Optional Configuration):
        .with_tools(tools: list[Tool]) -> Self
            Add tools that the LLM can choose to call.
            
        .with_tool_choice(tool_name: str) -> Self
            Force LLM to call a specific tool (enables structured output).
            Use tool name matching the output_model schema.
            
        .with_temperature(temperature: float) -> Self
            Override LLM temperature (0.0-2.0). Default inherits from LLM instance.
            
        .with_max_tokens(max_tokens: int) -> Self
            Override maximum tokens in response. Default inherits from LLM instance.
            
        .with_model(model_name: str) -> Self
            Override LLM model (e.g., "gpt-4o-mini"). Default inherits from LLM instance.
            
        .with_context_transform(fn: Callable[[str], str]) -> Self
            Custom function to format context before appending to system prompt.
            Default: lambda ctx: f"\\n\\nContext: {ctx}" if ctx else ""
            
        .with_system_prompt_builder(fn: Callable[[InputModel, str, ExecutionState], str]) -> Self
            Dynamic prompt builder with access to input, context, and state.
            Overrides static system_prompt if provided.
            
        .with_error_strategy(strategy: ErrorStrategy) -> Self
            Error handling strategy: RAISE (default), RETURN_NONE, LOG_AND_CONTINUE, SILENT
            
        .with_retry(max_attempts: int, initial_delay: float, **kwargs) -> Self
            Configure retry behavior for LLM failures.
            
        .as_artifact() -> Self
            Mark agent output as artifact for separate storage.
            
        .as_cacheable(ttl: int | None = None) -> Self
            Enable caching of agent results. TTL in seconds (None = no expiration).
    
    Returns:
        Agent instance that returns typed output_model instances.
    
    Raises:
        ValidationError: If LLM output doesn't match output_model schema
        Exception: If LLM execution fails (with error_strategy=RAISE)
    
    Notes:
        - Tool choice enforcement ensures type-safe structured output
        - Automatically creates tool from output_model schema
        - Integrates with ExecutionEvent for monitoring and retry
        - Compatible with state recovery and caching systems
    """
```

---

### 6.2 CompletionAgent - Docstring

```python
class CompletionAgent:
    """Builder for agents that perform simple text completions.
    
    Creates an agent that calls the LLM without tool usage, returning text content
    directly. Ideal for summarization, analysis, generation, and any task where
    free-form text output is desired.
    
    Usage Pattern:
        Use when you need natural language output rather than structured data.
        Perfect for summarization, creative writing, analysis, Q&A, and text
        transformation tasks.
    
    Examples:
        Basic usage:
        >>> agent = await CompletionAgent.from_prompt(
        ...     name="Summarizer",
        ...     llm=llm,
        ...     input_model=Message,
        ...     output_model=Message,
        ...     system_prompt="Summarize the provided text concisely."
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await CompletionAgent.from_prompt(
        ...         name="LegalReviewer",
        ...         llm=llm,
        ...         input_model=Doctrine,
        ...         output_model=Message,
        ...         system_prompt="Generate comprehensive legal analysis."
        ...     )
        ...     .with_temperature(0.8)  # More creative output
        ...     .with_max_tokens(3000)  # Longer responses
        ...     .with_system_prompt_builder(dynamic_context_fn)  # Dynamic prompts
        ...     .with_context_transform(custom_formatter)  # Custom context formatting
        ...     .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)  # Best-effort operation
        ...     .as_cacheable(ttl=1800)  # Cache for 30 minutes
        ... )
    
    Factory Method:
        from_prompt(name, llm, input_model, output_model, system_prompt) -> CompletionAgent
            Create a new CompletionAgent with mandatory parameters.
            
            Args:
                name: Unique identifier for the agent
                llm: LLM instance to use for generation
                input_model: Pydantic model for input validation
                output_model: Pydantic model for output (typically Message)
                system_prompt: Base system prompt for the agent
            
            Returns:
                CompletionAgent instance ready for optional configuration
    
    Chainable Methods (Optional Configuration):
        .with_temperature(temperature: float) -> Self
            Override LLM temperature (0.0-2.0). Default inherits from LLM instance.
            
        .with_max_tokens(max_tokens: int) -> Self
            Override maximum tokens in response. Default inherits from LLM instance.
            
        .with_model(model_name: str) -> Self
            Override LLM model (e.g., "gpt-4o-mini"). Default inherits from LLM instance.
            
        .with_context_transform(fn: Callable[[str], str]) -> Self
            Custom function to format context before appending to system prompt.
            Default: lambda ctx: f"\\n\\nContext: {ctx}" if ctx else ""
            
        .with_system_prompt_builder(fn: Callable[[InputModel, str, ExecutionState], str]) -> Self
            Dynamic prompt builder with access to input, context, and state.
            Overrides static system_prompt if provided.
            
        .with_error_strategy(strategy: ErrorStrategy) -> Self
            Error handling strategy: RAISE (default), RETURN_NONE, LOG_AND_CONTINUE, SILENT
            
        .with_retry(max_attempts: int, initial_delay: float, **kwargs) -> Self
            Configure retry behavior for LLM failures.
            
        .as_cacheable(ttl: int | None = None) -> Self
            Enable caching of agent results. TTL in seconds (None = no expiration).
    
    Returns:
        Agent instance that returns text content wrapped in output_model.
    
    Raises:
        Exception: If LLM execution fails (with error_strategy=RAISE)
    
    Notes:
        - No tool calls - pure text generation
        - Extracts 'content' field from LLM response
        - Integrates with ExecutionEvent for monitoring and retry
        - Compatible with state recovery and caching systems
    """
```

---

### 6.3 ToolExecutorAgent - Docstring

```python
class ToolExecutorAgent:
    """Builder for agents that execute tools based on LLM decisions.
    
    Creates an agent that gives the LLM access to tools, lets it decide which to call,
    executes them in parallel, and aggregates results. Handles tool call parsing,
    execution, error handling, and result combination automatically.
    
    Usage Pattern:
        Use when the LLM needs to interact with external systems, fetch data,
        perform computations, or take actions. The LLM decides which tools to call
        and with what parameters based on the input and context.
    
    Examples:
        Basic usage:
        >>> agent = await ToolExecutorAgent.from_tools(
        ...     name="DataFetcher",
        ...     llm=llm,
        ...     input_model=Query,
        ...     output_model=Result,
        ...     system_prompt="Fetch and analyze data using available tools.",
        ...     tools=[DatabaseTool(), APITool(), CalculatorTool()]
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await ToolExecutorAgent.from_tools(
        ...         name="StepExecutor",
        ...         llm=llm,
        ...         input_model=DoctrineStep,
        ...         output_model=StepResult,
        ...         system_prompt="Execute step using available tools.",
        ...         tools=[ListTool(), GetTool(), ProcessTool()]
        ...     )
        ...     .with_temperature(0.9)  # More deterministic tool selection
        ...     .with_system_prompt_builder(build_step_context)  # Dynamic context with dependencies
        ...     .with_first_result_only()  # Stop after first successful tool
        ...     .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)  # Continue on errors
        ...     .with_retry(max_attempts=3)  # Retry failed tool calls
        ... )
    
    Factory Method:
        from_tools(name, llm, input_model, output_model, system_prompt, tools) -> ToolExecutorAgent
            Create a new ToolExecutorAgent with mandatory parameters.
            
            Args:
                name: Unique identifier for the agent
                llm: LLM instance to use for tool selection
                input_model: Pydantic model for input validation
                output_model: Pydantic model for output
                system_prompt: Base system prompt for the agent
                tools: List of Tool instances available to the LLM
            
            Returns:
                ToolExecutorAgent instance ready for optional configuration
    
    Chainable Methods (Optional Configuration):
        .with_temperature(temperature: float) -> Self
            Override LLM temperature (0.0-2.0). Default inherits from LLM instance.
            
        .with_max_tokens(max_tokens: int) -> Self
            Override maximum tokens in response. Default inherits from LLM instance.
            
        .with_model(model_name: str) -> Self
            Override LLM model (e.g., "gpt-4o-mini"). Default inherits from LLM instance.
            
        .with_context_transform(fn: Callable[[str], str]) -> Self
            Custom function to format context before appending to system prompt.
            Default: lambda ctx: f"\\n\\nContext: {ctx}" if ctx else ""
            
        .with_system_prompt_builder(fn: Callable[[InputModel, str, ExecutionState], str]) -> Self
            Dynamic prompt builder with access to input, context, and state.
            Overrides static system_prompt if provided.
            
        .with_first_result_only() -> Self
            Stop execution after first successful tool call.
            Useful for fallback patterns (try tools until one succeeds).
            
        .with_error_strategy(strategy: ErrorStrategy) -> Self
            Error handling strategy: RAISE (default), RETURN_NONE, LOG_AND_CONTINUE, SILENT
            
        .with_retry(max_attempts: int, initial_delay: float, **kwargs) -> Self
            Configure retry behavior for tool execution failures.
            
        .as_cacheable(ttl: int | None = None) -> Self
            Enable caching of agent results. TTL in seconds (None = no expiration).
    
    Tool Execution:
        - All tool calls execute in PARALLEL using asyncio.gather
        - Results aggregated into dictionary keyed by tool name
        - Execution order preserved for debugging
        - Errors captured per-tool without blocking others
    
    Result Structure:
        Returns ToolExecutionResult with:
        - results: Dict[str, Any] - Results keyed by tool name
        - execution_order: list[str] - Order tools were called
        - errors: Dict[str, str] - Any errors that occurred
        
        Access patterns:
        >>> result = await agent(input, "", state)
        >>> db_result = result.get("database_tool")  # Get specific tool result
        >>> first_result = result.first()  # Get first successful result
    
    Raises:
        Exception: If all tools fail (with error_strategy=RAISE)
    
    Notes:
        - Tools execute in parallel for performance
        - Tools are assumed independent by design
        - For dependent operations, use separate agent steps
        - Integrates with ExecutionEvent for monitoring
        - Compatible with state recovery and caching
    """
```

---

### 6.4 DirectToolAgent - Docstring

```python
class DirectToolAgent:
    """Builder for agents that execute a single tool directly without LLM.
    
    Creates an agent that bypasses the LLM entirely and directly invokes a tool's
    handler. Useful for deterministic operations, data transformations, or when
    LLM reasoning is not needed.
    
    Usage Pattern:
        Use when you have a specific operation that doesn't require LLM reasoning,
        such as data validation, transformation, database queries, or API calls
        where the parameters are already known.
    
    Examples:
        Basic usage:
        >>> agent = await DirectToolAgent.from_tool(
        ...     name="Validator",
        ...     tool=validation_tool
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await DirectToolAgent.from_tool(
        ...         name="DatabaseQuery",
        ...         tool=db_tool
        ...     )
        ...     .with_error_strategy(ErrorStrategy.RETURN_NONE)  # Return None on failure
        ...     .with_retry(max_attempts=3, initial_delay=0.5)  # Retry on transient failures
        ...     .as_cacheable(ttl=300)  # Cache for 5 minutes
        ... )
    
    Factory Method:
        from_tool(name, tool) -> DirectToolAgent
            Create a new DirectToolAgent with mandatory parameters.
            
            Args:
                name: Unique identifier for the agent
                tool: Tool instance to execute directly
            
            Returns:
                DirectToolAgent instance ready for optional configuration
    
    Chainable Methods (Optional Configuration):
        .with_error_strategy(strategy: ErrorStrategy) -> Self
            Error handling strategy: RAISE (default), RETURN_NONE, LOG_AND_CONTINUE, SILENT
            
        .with_retry(max_attempts: int, initial_delay: float, **kwargs) -> Self
            Configure retry behavior for tool execution failures.
            
        .as_cacheable(ttl: int | None = None) -> Self
            Enable caching of tool results. TTL in seconds (None = no expiration).
    
    Returns:
        Agent instance that returns Tool_Result directly.
    
    Raises:
        Exception: If tool execution fails (with error_strategy=RAISE)
    
    Notes:
        - No LLM call - zero latency and cost overhead
        - Useful for deterministic operations in agent workflows
        - Can be composed with LLM-based agents in pipelines
        - Integrates with ExecutionEvent for monitoring
        - Compatible with state recovery and caching
    """
```

---

## 7. Implementation Plan

### 7.1 Proposed Phases

#### Phase 1: Core Builders (Week 1-2)
**Goal:** Implement the four basic builder classes

**Tasks:**
- [ ] Create `rh_agents/builders.py` module
- [ ] Implement `StructuredAgent.from_model()`
  - Handler generation logic
  - Tool choice enforcement
  - JSON extraction and validation
- [ ] Implement `CompletionAgent.from_prompt()`
  - Simple LLM call wrapper
  - Content extraction
- [ ] Implement `ToolExecutorAgent.from_tools()`
  - Tool execution loop
  - Error handling
  - Result aggregation (concatenate mode)
- [ ] Implement `DirectToolAgent.from_tool()`
  - Direct tool invocation wrapper
- [ ] Unit tests for each builder
- [ ] Type hints and docstrings

**Deliverables:**
- `rh_agents/builders.py` (~300-400 lines)
- `tests/test_builders.py` (~200 lines)
- Basic documentation

#### Phase 2: Result Aggregation Strategies (Week 2)
**Goal:** Implement flexible result combination

**Tasks:**
- [ ] Create `AggregationStrategy` enum (`concatenate`, `list`, `dict`, `first`)
- [ ] Implement concatenate strategy (default)
- [ ] Implement list strategy (preserves structure)
- [ ] Implement dict strategy (keyed by tool name)
- [ ] Implement first strategy (returns first result only)
- [ ] Add `aggregation_strategy` parameter to `ToolExecutorAgent`
- [ ] Tests for each strategy

**Deliverables:**
- Updated `ToolExecutorAgent` with strategy support
- Tests covering all strategies

#### Phase 3: Configuration & Customization (Week 3)
**Goal:** Add configuration options based on design decisions

**Tasks:**
- [ ] Implement LLM parameter overrides (model, temperature, max_tokens)
- [ ] Implement context handling (append to system prompt)
- [ ] Implement error strategy configuration
- [ ] Add parameter validation
- [ ] Documentation for all configuration options

**Deliverables:**
- Enhanced builders with full configuration
- Configuration guide in docs

#### Phase 4: Examples & Refactoring (Week 3-4)
**Goal:** Refactor existing examples and create new ones

**Tasks:**
- [ ] Refactor `examples/index.py` using builders
- [ ] Create `examples/builder_basic.py` - Basic usage
- [ ] Create `examples/builder_advanced.py` - All features
- [ ] Create `examples/builder_comparison.py` - Before/After
- [ ] Update main README with builder examples
- [ ] Create `docs/BUILDERS_GUIDE.md`

**Deliverables:**
- 4 new example files
- Updated documentation
- Migration guide

#### Phase 5: Templates Library (Week 4-5, Optional)
**Goal:** Create pre-built agent templates

**Tasks:**
- [ ] Create `rh_agents/templates.py` module
- [ ] Implement 5-7 template functions:
  - `create_json_extractor()`
  - `create_summarizer()`
  - `create_classifier()`
  - `create_qa_agent()`
  - `create_validator()`
- [ ] Tests for each template
- [ ] Template documentation with examples

**Deliverables:**
- `rh_agents/templates.py` (~150-200 lines)
- Template guide in docs

#### Phase 6: Polish & Documentation (Week 5)
**Goal:** Final testing and documentation

**Tasks:**
- [ ] Integration tests (builders + state recovery + retry + parallel)
- [ ] Performance benchmarks (verify no overhead)
- [ ] Error message improvements
- [ ] API reference documentation
- [ ] Tutorial: "From Manual to Builders"
- [ ] CHANGELOG entry
- [ ] Release notes

**Deliverables:**
- Complete documentation
- Ready for release

### 6.2 Implementation Guidelines

**Coding Standards:**
- Follow existing code style in `rh_agents/`
- Use type hints everywhere
- Write descriptive docstrings (Google style)
- Add inline comments for complex logic

**Testing Requirements:**
- Unit tests for each builder method
- Integration tests with real LLM (using mocks)
- Test error conditions
- Test edge cases (empty tools, missing models, etc.)

**Documentation Requirements:**
- Docstrings for all public methods
- Examples in docstrings
- Separate guide document
- Before/after comparisons

### 6.3 File Structure

```
rh_agents/
├── builders.py          # NEW - Builder classes
├── templates.py         # NEW - Pre-built templates (Phase 5)
├── agents.py            # UNCHANGED
├── core/
│   ├── actors.py        # UNCHANGED
│   └── ...
docs/
├── BUILDERS_GUIDE.md    # NEW - Usage guide
├── BOILERPLATE_REDUCTION_SPEC.md  # This document
examples/
├── builder_basic.py     # NEW
├── builder_advanced.py  # NEW
├── builder_comparison.py # NEW
├── index.py             # REFACTORED
tests/
├── test_builders.py     # NEW
├── test_templates.py    # NEW (Phase 5)
```

---

## 8. Implementation Readiness Verification

### 8.1 Success Criteria

#### 8.1.1 Functional Requirements

✅ **FR1:** `StructuredAgent.from_model()` successfully forces structured output  
✅ **FR2:** `CompletionAgent.from_prompt()` handles simple text completions  
✅ **FR3:** `ToolExecutorAgent.from_tools()` executes multiple tools and aggregates results  
✅ **FR4:** `DirectToolAgent.from_tool()` bypasses LLM and calls tools directly  
✅ **FR5:** All builders support caching, retry, and artifact configuration  
✅ **FR6:** Error handling matches current behavior (raises exceptions by default)  
✅ **FR7:** Context parameter is correctly passed to LLM prompts  

#### 8.1.2 Non-Functional Requirements

✅ **NFR1:** No performance degradation vs. manual implementation  
✅ **NFR2:** Zero breaking changes - existing code continues to work  
✅ **NFR3:** Type safety maintained - all methods properly typed  
✅ **NFR4:** Documentation complete - guides, examples, API reference  
✅ **NFR5:** Test coverage ≥ 80% for new code  
✅ **NFR6:** Examples demonstrate real-world usage  

#### 8.1.3 User Experience Requirements

✅ **UX1:** Reduce boilerplate by ≥70% for common patterns  
✅ **UX2:** Clear error messages when misconfigured  
✅ **UX3:** IDE autocomplete works for all parameters  
✅ **UX4:** Easy migration path from manual → builder  
✅ **UX5:** Debugging is straightforward (verbose mode)  

#### 8.1.4 Metrics

**Before (Manual Implementation):**
- Average agent definition: ~40 lines
- Time to create agent: ~15 minutes
- Common errors: Missing error handling, incorrect tool parsing

**After (Builder Implementation):**
- Average agent definition: ~8 lines (80% reduction)
- Time to create agent: ~3 minutes (80% reduction)
- Common errors: Reduced via built-in validation

**Target Impact:**
- For a typical application with 5 agents:
  - **Lines saved:** ~160 lines
  - **Time saved:** ~60 minutes initial development
  - **Maintenance:** Easier to update (one place vs. five)

---

### 8.2 Design Completeness Checklist

**Core Architecture:**
- ✅ Four builder patterns identified and defined
- ✅ Hybrid API design (mandatory kwargs + chainable optionals)
- ✅ Error handling strategies specified
- ✅ Result aggregation approaches defined
- ✅ Module organization decided (`rh_agents.builders`)
- ✅ Template library scope defined

**API Design:**
- ✅ Factory methods specified for each builder
- ✅ Chainable methods documented with signatures
- ✅ Parameter inheritance model defined (LLM defaults + overrides)
- ✅ Return types specified (Agent instances)
- ✅ Comprehensive docstrings created

**Integration:**
- ✅ ExecutionEvent compatibility maintained
- ✅ EventBus integration preserved
- ✅ State recovery compatibility ensured
- ✅ Caching and retry integration defined
- ✅ No breaking changes confirmed

**Examples:**
- ✅ Complete pipeline example provided (180→60 lines)
- ✅ Before/after comparisons shown
- ✅ Template usage examples included
- ✅ Real-world use case demonstrated

**Documentation:**
- ✅ Design decisions consolidated and approved
- ✅ Implementation phases defined
- ✅ Success criteria established
- ✅ File structure planned

---

### 8.3 Implementation Decisions (Approved)

All implementation questions have been reviewed and decided:

---

#### Decision 8.3.1: Chaining Implementation Pattern ✅

**Approved:** Mutable (Return Self)

**Implementation:**
```python
class AgentBuilder:
    def with_temperature(self, temperature: float) -> Self:
        self._temperature = temperature
        return self  # Modifies instance in place
```

**Rationale:**
- More memory efficient
- Simpler implementation
- Standard Python builder pattern
- Consistent with common Python practices



---

#### Decision 8.3.2: Builder Return Type ✅

**Approved:** Return Agent Directly

**Implementation:**
```python
class StructuredAgent:
    @staticmethod
    async def from_model(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str
    ) -> Agent:
        """Return Agent instance with chainable methods attached."""
        # Create handler that uses ExecutionEvent
        agent = Agent(...)
        # Attach chainable methods dynamically
        agent.with_temperature = lambda t: setattr_and_return(agent, '_temperature', t)
        return agent
```

**Rationale:**
- Simpler API - no `.build()` call needed
- Can use agent immediately or continue chaining
- Cleaner user experience
- Agent is ready to use after factory method


---

#### Decision 8.3.3: ErrorStrategy Enum Location ✅

**Approved:** In core/types.py

**Implementation:**
```python
# rh_agents/core/types.py
from enum import Enum

class ErrorStrategy(str, Enum):
    RAISE = "raise"              # Raise exception immediately (default)
    RETURN_NONE = "return_none"  # Return ExecutionResult with ok=False
    LOG_AND_CONTINUE = "log"     # Log error and return partial results
    SILENT = "silent"            # Suppress errors, return None
```

**Rationale:**
- Reusable across package
- Consistent with other enums (EventType, etc.)
- Clean import path: `from rh_agents.core.types import ErrorStrategy`

**Important Note:** Error strategy should match the current retry/error handling config. No extra logic in core components - keep it simple and compatible with existing patterns.

---

#### Decision 8.3.4: ToolExecutionResult Model Location ✅

**Approved:** In core/result_types.py

**Implementation:**
```python
# rh_agents/core/result_types.py
from pydantic import BaseModel
from typing import Dict, Any

class ToolExecutionResult(BaseModel):
    """Result from executing multiple tools in parallel."""
    results: Dict[str, Any]      # Tool name -> result
    execution_order: list[str]   # Order of execution
    errors: Dict[str, str]       # Tool name -> error message
    
    def get(self, tool_name: str) -> Any:
        """Get result for specific tool."""
        return self.results.get(tool_name)
    
    def first(self) -> Any:
        """Get first successful result."""
        if self.execution_order:
            return self.results[self.execution_order[0]]
        return None
```

**Rationale:**
- Consistent with LLM_Result, Tool_Result, ExecutionResult
- Reusable across package
- Clean separation: result types vs builders

---

#### Decision 8.3.5: Testing Strategy ✅

**Approved:** Both Unit and Integration Tests

**Test Structure:**
```python
# tests/test_builders_unit.py
# Fast unit tests with mocked LLM
class TestBuildersUnit:
    async def test_structured_agent_creation(self):
        llm = MockLLM(response={"tool_calls": [...]})
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=Message,
            output_model=Doctrine,
            system_prompt="Test prompt"
        )
        assert agent.name == "TestAgent"

# tests/test_builders_integration.py
# Integration tests with real OpenAI API (run separately)
class TestBuildersIntegration:
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
    async def test_structured_agent_real_llm(self):
        llm = OpenAILLM()
        agent = await StructuredAgent.from_model(...)
        result = await agent(test_input, "", state)
        assert isinstance(result, Doctrine)
```

**Rationale:**
- Fast unit tests for development (no API costs)
- Integration tests for confidence (run in CI or manually)
- Best of both worlds
- Standard pytest markers for separate execution


---

#### Decision 8.3.6: Transform Function Parameter ✅

**Approved:** Exclude Transform (for v1)

**Implementation:**
```python
# No .with_transform() method in v1
# Users perform transformation after agent call
result = await agent(input, "", state)
transformed = transform_output(result)  # Explicit transformation
```

**Rationale:**
- Simpler API for v1
- More explicit - easier to understand and debug
- Can be added in v2 if users request it
- Focus on core functionality first


---

### 8.4 Critical Implementation Requirements

#### 8.4.1 ExecutionEvent Usage (MANDATORY)

**🚨 CRITICAL:** All builder-generated agent handlers MUST use `ExecutionEvent` for LLM and Tool calls. This is non-negotiable as it provides:

1. **Automatic Logging** - Events are emitted to EventBus for monitoring
2. **Artifact Storage** - Results are stored for state recovery
3. **Retry Integration** - Failed executions can be retried
4. **Event Streaming** - EventPrinter shows execution flow
5. **Caching Support** - Results can be cached at actor level

**Pattern for LLM Calls:**
```python
# Inside generated handler
async def handler(input_data: InputModel, context: str, execution_state: ExecutionState):
    # Create ExecutionEvent for LLM
    llm_event = ExecutionEvent(actor=llm)
    
    # Prepare LLM request
    llm_input = OpenAIRequest(
        system_message=system_prompt + (f"\n\nContext: {context}" if context else ""),
        prompt=input_data.content,
        model=model_name,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        tools=tool_set
    )
    
    # Execute with ExecutionEvent (NOT direct llm call)
    execution_result = await llm_event(llm_input, context, execution_state)
    
    # Handle result
    if not execution_result.ok or execution_result.result is None:
        raise Exception(f"LLM execution failed: {execution_result.erro_message}")
    
    return process_result(execution_result.result)
```

**Pattern for Tool Calls:**
```python
# Inside generated handler for ToolExecutorAgent
async def execute_tool(tool_call):
    tool = tool_set[tool_call.tool_name]
    
    # Create ExecutionEvent for Tool
    tool_event = ExecutionEvent(actor=tool)
    
    # Parse tool arguments
    tool_input = tool.input_model.model_validate_json(tool_call.arguments)
    
    # Execute with ExecutionEvent (NOT direct tool call)
    tool_result = await tool_event(tool_input, context, execution_state)
    
    if not tool_result.ok or tool_result.result is None:
        return {"error": tool_result.erro_message}
    
    return {"result": tool_result.result}

# Parallel execution with ExecutionEvent
tasks = [execute_tool(tc) for tc in response.tools]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Reference Implementation:**
See [rh_agents/agents.py](../rh_agents/agents.py) for current agent implementations:
- `DoctrineReceverAgent.handler` (lines 112-132) - LLM with tool choice
- `StepExecutorAgent.handler` (lines 157-243) - LLM + parallel tool execution
- `ReviewerAgent.handler` (lines 256-280) - Simple LLM completion

**Why This Matters:**
Without ExecutionEvent, builders would create "dead" agents that:
- ❌ Don't emit events (no logging, no monitoring)
- ❌ Can't store artifacts (breaks state recovery)
- ❌ Can't leverage retry mechanism
- ❌ Don't integrate with EventPrinter
- ❌ Can't be cached

**Implementation Checklist:**
- [ ] StructuredAgent handler uses ExecutionEvent for LLM call
- [ ] CompletionAgent handler uses ExecutionEvent for LLM call
- [ ] ToolExecutorAgent handler uses ExecutionEvent for LLM AND each tool call
- [ ] DirectToolAgent handler uses ExecutionEvent for tool call
- [ ] All handlers pass (input, context, execution_state) to ExecutionEvent
- [ ] All handlers check execution_result.ok before proceeding
- [ ] Error messages include execution_result.erro_message

---

### 8.5 Readiness Status

**Current Status:** ✅ **READY FOR IMPLEMENTATION**

**All Prerequisites Complete:**
- ✅ Design decisions approved (Section 3)
- ✅ Implementation decisions approved (Section 8.3)
- ✅ API design complete with docstrings (Section 6)
- ✅ Complete pipeline example provided (Section 5)
- ✅ ExecutionEvent requirements documented (Section 8.4)
- ✅ Success criteria defined (Section 8.1)
- ✅ Phase plan ready (Section 7)

**Ready to Proceed:** **YES** - Phase 1 can begin immediately

**Next Action:** Begin Phase 1 implementation - Core Builders (rh_agents/builders.py)

---

### 8.6 Final Implementation Decisions (Approved)

All remaining implementation questions have been reviewed and decided:

---

#### Decision 8.6.1: Chainable Methods Implementation ✅

**Approved:** Option C - Extend Agent Class

**Implementation:**
```python
class BuilderAgent(Agent):
    """Agent subclass with chainable builder methods."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overrides = {}
    
    def with_temperature(self, temp: float) -> Self:
        self._overrides['temperature'] = temp
        return self
    
    def with_max_tokens(self, tokens: int) -> Self:
        self._overrides['max_tokens'] = tokens
        return self
    
    def with_model(self, model: str) -> Self:
        self._overrides['model'] = model
        return self
    
    # ... additional chainable methods

async def from_model(...) -> BuilderAgent:
    agent = BuilderAgent(...)
    return agent
```

**Rationale:**
- Type hints work perfectly
- IDE autocomplete works
- Still an Agent (isinstance checks work)
- Clean inheritance pattern
- User comment: "Option C is Great"

---

#### Decision 8.6.2: Template Library Timing ✅

**Approved:** Option B - Defer to Phase 5

**Implementation Plan:**
```python
# Phase 1: Core builders only (rh_agents/builders.py)
# - StructuredAgent
# - CompletionAgent
# - ToolExecutorAgent
# - DirectToolAgent

# Phase 5: Add templates after builders are proven (rh_agents/templates.py)
# - create_json_extractor()
# - create_summarizer()
# - create_classifier()
# - create_qa_agent()
# - etc.
```

**Rationale:**
- Focus on core functionality first
- Learn from real usage before creating templates
- Templates can be better informed by user feedback
- Reduces Phase 1 scope and complexity
- Comment: "Do it lastly"

---

#### Decision 8.6.3: Dynamic System Prompt Storage ✅

**Approved:** Option B - Closure Over Function

**Implementation:**
```python
async def from_model(
    name: str,
    llm: LLM,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    system_prompt: str
) -> BuilderAgent:
    # Closure variables to store overrides
    system_prompt_builder = None  # Will hold dynamic prompt function if set
    context_transform = lambda ctx: f"\n\nContext: {ctx}" if ctx else ""
    
    async def handler(input_data, context, execution_state):
        # Access closure variables
        if system_prompt_builder:
            prompt = await system_prompt_builder(input_data, context, execution_state)
        else:
            # Apply context transform to static prompt
            prompt = system_prompt + context_transform(context)
        
        # ... rest of handler logic using ExecutionEvent
    
    agent = BuilderAgent(
        name=name,
        handler=handler,
        # ... other params
    )
    
    # Add chainable method that modifies closure variable
    def with_system_prompt_builder(fn):
        nonlocal system_prompt_builder
        system_prompt_builder = fn
        return agent
    
    agent.with_system_prompt_builder = with_system_prompt_builder
    
    return agent
```

**Rationale:**
- Clean closure pattern
- No attribute pollution on Agent instance
- Function scope is clear and encapsulated
- Easy to reason about
- Works well with dynamic handler generation

---

### 8.7 Phase 1 Scope Analysis

#### What's Needed for Phase 1

**Core Deliverables:**

1. **rh_agents/builders.py** (~400-500 lines)
   - `BuilderAgent` class (extends Agent)
   - `StructuredAgent.from_model()` factory
   - `CompletionAgent.from_prompt()` factory
   - `ToolExecutorAgent.from_tools()` factory
   - `DirectToolAgent.from_tool()` factory
   - All chainable methods implementation

2. **rh_agents/core/types.py** (~15 lines addition)
   - `ErrorStrategy` enum

3. **rh_agents/core/result_types.py** (~30 lines addition)
   - `ToolExecutionResult` class

4. **tests/test_builders_unit.py** (~300 lines)
   - Unit tests for all 4 builders
   - Mock LLM responses
   - Test all chainable methods
   - Test error handling

5. **tests/test_builders_integration.py** (~150 lines)
   - Integration tests with real OpenAI API
   - Marked with @pytest.mark.integration
   - Skip if no API key

6. **examples/builder_basic.py** (~100 lines)
   - Simple examples of each builder
   - Demonstrates core functionality

7. **docs/BUILDERS_GUIDE.md** (~200 lines)
   - User guide
   - Migration from manual agents
   - API reference

**Total Estimated Lines:** ~1,200-1,300 lines of new code

**Estimated Timeline:** 1-2 weeks

---

#### What's NOT in Phase 1

**Deferred to Later Phases:**

❌ **Templates** (Phase 5)
   - `rh_agents/templates.py`
   - `create_json_extractor()`, `create_summarizer()`, etc.
   - Will be created after builders are validated

❌ **Advanced Aggregation Strategies** (Phase 2)
   - Currently only dictionary aggregation
   - Can add list/concatenate/first strategies in Phase 2 if needed

❌ **Result Aggregation Modes** (Phase 2)
   - Phase 1 uses simple dictionary by tool name
   - Enhanced aggregation strategies deferred

❌ **Transform Functions** (None - excluded from v1)
   - `.with_transform()` not included
   - Users transform results externally

❌ **Full Documentation** (Phase 6)
   - Complete API reference
   - Tutorials
   - Migration guides
   - Phase 1 has basic BUILDERS_GUIDE.md only

---

#### Phase 1 Implementation Checklist

**Pre-Implementation:**
- [x] All design decisions finalized
- [x] All implementation decisions finalized
- [x] ExecutionEvent patterns documented
- [x] Success criteria defined
- [x] Phase scope analyzed

**Files to Create:**
- [ ] `rh_agents/builders.py` - Core builders module
- [ ] `tests/test_builders_unit.py` - Unit tests
- [ ] `tests/test_builders_integration.py` - Integration tests
- [ ] `examples/builder_basic.py` - Basic examples
- [ ] `docs/BUILDERS_GUIDE.md` - User guide

**Files to Modify:**
- [ ] `rh_agents/core/types.py` - Add ErrorStrategy
- [ ] `rh_agents/core/result_types.py` - Add ToolExecutionResult
- [ ] `rh_agents/__init__.py` - Export builders (optional)

**Testing Requirements:**
- [ ] All 4 builders tested with unit tests
- [ ] All chainable methods tested
- [ ] ExecutionEvent integration verified
- [ ] Error handling tested
- [ ] Integration tests with real LLM
- [ ] Type hints validated (mypy)

**Documentation Requirements:**
- [ ] Each builder has comprehensive docstring
- [ ] Basic user guide created
- [ ] Code examples work
- [ ] Migration patterns documented

---

#### Critical Implementation Requirements

🚨 **MANDATORY for Phase 1:**

1. **ExecutionEvent Usage**
   - Every handler MUST use ExecutionEvent
   - For LLM calls: `ExecutionEvent(actor=llm)`
   - For tool calls: `ExecutionEvent(actor=tool)`
   - See Section 8.4.1 for patterns

2. **BuilderAgent Class**
   - Extends Agent
   - Has `_overrides` dict
   - All chainable methods return `self`
   - Type hints use `Self` for proper IDE support

3. **Closure-Based Overrides**
   - Dynamic prompts use closure variables
   - `nonlocal` for modifying closure state
   - Clean encapsulation

4. **Error Handling**
   - Default: `ErrorStrategy.RAISE`
   - Check `execution_result.ok` before proceeding
   - Include `execution_result.erro_message` in exceptions

5. **Type Safety**
   - Full type hints on all methods
   - Pydantic models for validation
   - Generic types where appropriate

---

### 8.8 Final Readiness Status

**Implementation Ready:** ✅ **YES - ALL DECISIONS FINALIZED**

**All Decisions Complete:**
- ✅ All design decisions approved (Section 3)
- ✅ All implementation decisions approved (Section 8.3)
- ✅ All final implementation decisions approved (Section 8.6)
- ✅ ExecutionEvent requirements documented (Section 8.4)
- ✅ Phase 1 scope analyzed (Section 8.7)
- ✅ Success criteria defined (Section 8.1)

**Phase 1 Ready:** **YES** - Can begin implementation immediately

**Next Steps:**
1. Create `rh_agents/builders.py` with BuilderAgent class
2. Implement 4 core builders (Structured, Completion, ToolExecutor, DirectTool)
3. Add ErrorStrategy to `core/types.py`
4. Add ToolExecutionResult to `core/result_types.py`
5. Create unit tests (`test_builders_unit.py`)
6. Create integration tests (`test_builders_integration.py`)
7. Create basic example (`builder_basic.py`)
8. Write BUILDERS_GUIDE.md

**Estimated Timeline:** 1-2 weeks for Phase 1 completion

---

## 9. Appendix

### 9.1 Complete Before/After Example

**Before: Manual Implementation (180 lines)**

```python
# examples/index.py - BEFORE
import asyncio
from db import DOC_LIST, DOCS
from rh_agents.agents import OpenAILLM
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import ExecutionState
from rh_agents.openai import OpenAIRequest
from rh_agents.models import Message, AuthorType
from pydantic import BaseModel, Field

# ... 170 more lines of agent definitions
# (DoctrineReceverAgent, StepExecutorAgent, ReviewerAgent)

if __name__ == "__main__":
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools[1:])
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    # ... execution code
```

**After: Builder Implementation (45 lines)**

```python
# examples/index.py - AFTER
import asyncio
from db import DOC_LIST, DOCS
from rh_agents import OpenAILLM
from rh_agents.builders import StructuredAgent, CompletionAgent, ToolExecutorAgent
from rh_agents.models import Message, AuthorType, Doctrine, DoctrineStep, StepResult
from custom_tools import DoctrineTool, ListPecasTool, GetTextoPecaTool

if __name__ == "__main__":
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    
    # Doctrine Receiver: Forces structured output
    doctrine_receiver = StructuredAgent.from_model(
        name="DoctrineReceiver",
        llm=llm,
        input_model=Message,
        output_model=Doctrine,
        system_prompt="Analisa o pedido do usuário e gera um plano estruturado...",
        tools=[DoctrineTool()],
        tool_choice="DoctrineTool",
        is_artifact=True,
        cacheable=True
    )
    
    # Step Executor: Runs tools based on LLM decisions
    step_executor = ToolExecutorAgent.from_tools(
        name="StepExecutor",
        llm=llm,
        input_model=DoctrineStep,
        output_model=StepResult,
        system_prompt="Você é um executor de passos...",
        tools=[ListPecasTool(), GetTextoPecaTool()],
        aggregation_strategy="concatenate"
    )
    
    # Reviewer: Simple completion
    reviewer = CompletionAgent.from_prompt(
        name="Reviewer",
        llm=llm,
        input_model=Doctrine,
        output_model=Message,
        system_prompt="Você é um revisor especializado..."
    )
    
    # Orchestration (unchanged)
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver,
        step_executor_agent=step_executor,
        reviewer_agent=reviewer
    )
    
    # ... execution code (unchanged)
```

**Reduction:** 180 lines → 45 lines (75% reduction)

---

### 9.2 Approved Design Decisions Summary

**All decisions have been reviewed and approved. Here's the final configuration:**

| Decision | Chosen Approach | Key Details |
|----------|----------------|-------------|
| **3.1 Builder Pattern** | Builder/Factory with Hybrid API | Mandatory params as kwargs, optional via chaining |
| **3.2 Error Handling** | Configurable Strategy | Default: `raise`, options: `return_none`, `log`, `silent` |
| **3.3 Tool Execution** | Parallel Only | All tools execute concurrently via asyncio.gather |
| **3.4 Result Aggregation** | Dictionary by Tool Name | With optional `first()` mode |
| **3.5 Context Building** | Append to System Prompt | With optional custom transformer |
| **3.6 System Prompt** | Simple String | With optional builder callback |
| **3.7 LLM Configuration** | Hybrid Inheritance | Inherit from LLM + optional overrides |
| **3.8 Module Organization** | `rh_agents.builders` | Clear separation of concerns |
| **4.1 Templates** | Template Library | 5-10 essential templates |
| **4.2 Debug Mode** | Use Existing EventPrinter | No special support needed |
| **4.3 Async Builders** | Async Only | Consistent with package architecture |
| **4.4 Type Inference** | Explicit Models | No inference, explicit is better |
| **4.5 API Style** | Hybrid | Mandatory as kwargs + chainable optionals |

---

### 9.3 General Open Questions

1. **Naming:** Is `StructuredAgent`, `CompletionAgent`, `ToolExecutorAgent` clear? Alternative suggestions?
   
2. **Backwards Compatibility:** Should we deprecate manual agent creation in future versions?
   
3. **Performance:** Should we benchmark builder-generated agents vs. handwritten ones?
   
4. **Error Messages:** What information should error messages include when builders fail?
   
5. **Testing:** Should we test builders with real OpenAI API or just mocks?

---

### 9.4 Related Documents

- [Architecture Specification](./ARCHITECTURE_DIAGRAMS.md)
- [Retry Mechanism Spec](./RETRY_MECHANISM_SPEC.md)
- [State Recovery Spec](./STATE_RECOVERY_SPEC.md)
- [Parallel Execution Spec](./parallel/PARALLEL_EXECUTION_SPEC.md)
- [Current Implementation](../rh_agents/agents.py)
- [Example Application](../examples/index.py)

---

**Document Status:** ✅ **READY FOR IMPLEMENTATION**  
**Next Action:** Begin Phase 1 - Create rh_agents/builders.py  
**Estimated Implementation Time:** 1-2 weeks for Phase 1  
**Risk Level:** Low (isolated new feature, no breaking changes)

---

## 📝 Status Summary

**Design Status:** ✅ All design decisions finalized and approved  
**Implementation Status:** ✅ All 9 implementation decisions approved  
**Critical Requirements:** ✅ ExecutionEvent usage patterns documented  
**Phase 1 Scope:** ✅ Analyzed and ready  
**Date:** February 18, 2026

**All Approved Decisions:**
- ✅ Hybrid API (mandatory kwargs + chainable optionals)
- ✅ Dictionary-based tool result aggregation
- ✅ Parallel tool execution by default
- ✅ Async-only builders
- ✅ Module: `rh_agents.builders`
- ✅ Mutable chaining (return self)
- ✅ Return Agent directly via BuilderAgent subclass
- ✅ ErrorStrategy in core/types.py
- ✅ ToolExecutionResult in core/result_types.py
- ✅ Both unit and integration tests
- ✅ Exclude transform parameter (v1)
- ✅ **BuilderAgent subclass with chainable methods**
- ✅ **Templates deferred to Phase 5**
- ✅ **Closure-based dynamic prompt storage**
- ✅ **CRITICAL: All handlers must use ExecutionEvent**

**Phase 1 Deliverables:**
- `rh_agents/builders.py` (~400-500 lines)
- `rh_agents/core/types.py` (add ErrorStrategy)
- `rh_agents/core/result_types.py` (add ToolExecutionResult)
- `tests/test_builders_unit.py` (~300 lines)
- `tests/test_builders_integration.py` (~150 lines)
- `examples/builder_basic.py` (~100 lines)
- `docs/BUILDERS_GUIDE.md` (~200 lines)

**Ready to Code:** **YES** - All prerequisites complete, can start immediately.

---

## 🚀 Phase 1 Implementation Quick Reference

### Core Components to Build

**1. BuilderAgent Class** (`rh_agents/builders.py`)
```python
class BuilderAgent(Agent):
    """Extends Agent with chainable builder methods."""
    - __init__: Initialize with _overrides dict
    - with_temperature(float) -> Self
    - with_max_tokens(int) -> Self
    - with_model(str) -> Self
    - with_tools(list[Tool]) -> Self
    - with_tool_choice(str) -> Self
    - with_context_transform(Callable) -> Self
    - with_system_prompt_builder(Callable) -> Self
    - with_error_strategy(ErrorStrategy) -> Self
    - with_retry(**kwargs) -> Self
    - with_first_result_only() -> Self
    - as_artifact() -> Self
    - as_cacheable(ttl=None) -> Self
```

**2. StructuredAgent** (`rh_agents/builders.py`)
```python
class StructuredAgent:
    @staticmethod
    async def from_model(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str
    ) -> BuilderAgent:
        # Implementation:
        # 1. Create closure variables for overrides
        # 2. Generate handler that uses ExecutionEvent for LLM call
        # 3. Handler forces toolchoice for structured output
        # 4. Extracts JSON from tool call
        # 5. Returns BuilderAgent with chainable methods attached
```

**3. CompletionAgent** (`rh_agents/builders.py`)
```python
class CompletionAgent:
    @staticmethod
    async def from_prompt(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str
    ) -> BuilderAgent:
        # Implementation:
        # 1. Create closure variables
        # 2. Generate handler that uses ExecutionEvent for LLM call
        # 3. No tools - simple prompt/response
        # 4. Extracts content from LLM_Result
        # 5. Returns BuilderAgent
```

**4. ToolExecutorAgent** (`rh_agents/builders.py`)
```python
class ToolExecutorAgent:
    @staticmethod
    async def from_tools(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str,
        tools: list[Tool]
    ) -> BuilderAgent:
        # Implementation:
        # 1. Create closure variables
        # 2. Generate handler that:
        #    a. Uses ExecutionEvent for LLM call
        #    b. Iterates tool calls with ExecutionEvent for each
        #    c. Executes tools in parallel (asyncio.gather)
        #    d. Aggregates results into ToolExecutionResult
        # 3. Returns BuilderAgent
```

**5. DirectToolAgent** (`rh_agents/builders.py`)
```python
class DirectToolAgent:
    @staticmethod
    async def from_tool(
        name: str,
        tool: Tool
    ) -> BuilderAgent:
        # Implementation:
        # 1. Generate handler that uses ExecutionEvent for tool
        # 2. No LLM call - direct tool execution
        # 3. Returns BuilderAgent (limited chainable methods)
```

**6. ErrorStrategy Enum** (`rh_agents/core/types.py`)
```python
class ErrorStrategy(str, Enum):
    RAISE = "raise"
    RETURN_NONE = "return_none"
    LOG_AND_CONTINUE = "log"
    SILENT = "silent"
```

**7. ToolExecutionResult** (`rh_agents/core/result_types.py`)
```python
class ToolExecutionResult(BaseModel):
    results: Dict[str, Any]
    execution_order: list[str]
    errors: Dict[str, str]
    
    def get(self, tool_name: str) -> Any: ...
    def first(self) -> Any: ...
```

### Critical Implementation Pattern

**Every handler MUST follow this pattern:**
```python
async def handler(input_data: InputModel, context: str, execution_state: ExecutionState):
    # 1. Create ExecutionEvent
    llm_event = ExecutionEvent(actor=llm)
    
    # 2. Prepare request (apply overrides from closure)
    llm_input = OpenAIRequest(
        system_message=system_prompt + context_transform(context),
        prompt=input_data.content,
        model=overrides.get('model', llm.default_model),
        max_completion_tokens=overrides.get('max_tokens', llm.default_max_tokens),
        temperature=overrides.get('temperature', llm.default_temperature),
        tools=tool_set
    )
    
    # 3. Execute with ExecutionEvent
    execution_result = await llm_event(llm_input, context, execution_state)
    
    # 4. Check result
    if not execution_result.ok or execution_result.result is None:
        raise Exception(f"LLM execution failed: {execution_result.erro_message}")
    
    # 5. Process and return
    return process_result(execution_result.result)
```

### Testing Requirements

**Unit Tests** (`tests/test_builders_unit.py`):
- Test each builder factory method
- Test all chainable methods
- Test handler generation
-Mock LLM responses
- Test error handling
- Verify ExecutionEvent is used

**Integration Tests** (`tests/test_builders_integration.py`):
- Test with real OpenAI API
- Mark with `@pytest.mark.integration`
- Skip if no API key
- Test complete workflow
- Verify artifacts storage
- Verify event emission

### Documentation

**BUILDERS_GUIDE.md** should include:
1. Quick start with each builder
2. Migration guide from manual agents
3. All chainable methods reference
4. Common patterns and examples
5. Troubleshooting
6. Best practices

### Success Criteria

Phase 1 is complete when:
- [ ] All 4 builders implemented and working
- [ ] BuilderAgent class with all chainable methods
- [ ] ErrorStrategy and ToolExecutionResult added
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Basic example works end-to-end
- [ ] BUILDERS_GUIDE.md written
- [ ] Can refactor examples/index.py using builders
- [ ] No breaking changes to existing code
- [ ] Type hints validate with mypy

---

**End of Specification**
