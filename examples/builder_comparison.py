"""
Before/After comparison showing builder pattern benefits.

Compares traditional agent implementation with builder pattern for:
- Structured output parsing
- Simple completions  
- Tool execution workflows
- Configuration complexity
"""

import asyncio
from pydantic import BaseModel, Field
from typing import Optional

from rh_agents.core.actors import Agent, Tool, ToolSet, LLM
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, Tool_Result
from rh_agents.core.types import EventType, ErrorStrategy
from rh_agents.models import Message, AuthorType
from rh_agents.openai import OpenAIRequest
from rh_agents.llms import OpenAILLM
from rh_agents.builders import StructuredAgent, CompletionAgent, ToolExecutorAgent


# ============================================================================
# Example 1: Structured Output Parsing
# ============================================================================

class UserRequest(BaseModel):
    """User input message."""
    content: str


class ParsedTask(BaseModel):
    """Structured task information."""
    task_type: str = Field(description="Type of task")
    priority: str = Field(description="Priority level")
    subject: str = Field(description="Main subject")


print("\n" + "="*80)
print("COMPARISON 1: Structured Output Parsing")
print("="*80)

print("\n" + "-"*80)
print("BEFORE: Traditional Agent (40+ lines)")
print("-"*80)

print("""
class TaskParserAgent_OLD(Agent):
    def __init__(self, llm: LLM):
        PROMPT = "Parse user request into structured task format."
        
        async def handler(input_data: UserRequest, context: str, execution_state: ExecutionState):
            # Create ExecutionEvent for LLM
            llm_event = ExecutionEvent(actor=llm)
            
            # Build system prompt with context
            system_prompt = PROMPT
            if context:
                system_prompt += f"\\n\\nContext: {context}"
            
            # Prepare LLM request with tool choice
            llm_input = OpenAIRequest(
                system_message=system_prompt,
                prompt=input_data.content,
                model='gpt-4o',
                max_completion_tokens=2500,
                temperature=0.7,
                tools=ToolSet(tools=[]),
                tool_choice={"type": "function", "function": {"name": "ParseTool"}}
            )
            
            # Execute with error handling
            execution_result = await llm_event(llm_input, context, execution_state)
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM failed: {execution_result.erro_message}")
            
            result = execution_result.result
            
            # Handle content fallback
            if result.is_content:
                return Message(content=result.content, author=AuthorType.ASSISTANT)
            
            # Extract tool call
            if not (result.is_tool_call and result.tools and result.tools[0]):
                raise Exception("Expected tool call for structured output")
            
            # Parse and validate JSON
            tool_call = result.tools[0]
            return ParsedTask.model_validate_json(tool_call.arguments)
        
        super().__init__(
            name="TaskParser_OLD",
            description=PROMPT,
            input_model=UserRequest,
            output_model=ParsedTask,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools=[])
        )
""")

print("\n" + "-"*80)
print("AFTER: Builder Pattern (5 lines)")
print("-"*80)

print("""
# Just 5 lines - that's it!
parser_NEW = await StructuredAgent.from_model(
    name="TaskParser",
    llm=llm,
    input_model=UserRequest,
    output_model=ParsedTask,
    system_prompt="Parse user request into structured task format."
)
""")

print("\n✅ SAVINGS: 40+ lines → 5 lines (88% reduction)")
print("   - No ExecutionEvent boilerplate")
print("   - No OpenAIRequest construction")
print("   - No tool call extraction")
print("   - No error handling code")
print("   - No JSON validation")


# ============================================================================
# Example 2: Adding Configuration
# ============================================================================

print("\n" + "="*80)
print("COMPARISON 2: Adding Configuration (Temperature, Caching, Retry)")
print("="*80)

print("\n" + "-"*80)
print("BEFORE: Manual configuration (scattered across code)")
print("-"*80)

print("""
# Configuration mixed with implementation
class TaskParserAgent_OLD(Agent):
    def __init__(self, llm: LLM, temperature=0.7, max_tokens=2000, enable_cache=True):
        # Store config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = enable_cache
        
        async def handler(input_data, context, execution_state):
            # ... ExecutionEvent setup ...
            
            # Config buried in request
            llm_input = OpenAIRequest(
                system_message=prompt,
                prompt=input_data.content,
                model='gpt-4o',
                max_completion_tokens=self.max_tokens,  # Config here
                temperature=self.temperature,            # Config here
                ...
            )
            
            # ... rest of implementation ...
        
        super().__init__(
            ...
            cacheable=self.cache_enabled,  # Config here
            retry_config=RetryConfig(      # Config here
                max_attempts=3,
                initial_delay=1.0
            )
        )
""")

print("\n" + "-"*80)
print("AFTER: Chainable configuration (fluent, readable)")
print("-"*80)

print("""
# Configuration is explicit and chainable
parser_NEW = (
    await StructuredAgent.from_model(
        name="TaskParser",
        llm=llm,
        input_model=UserRequest,
        output_model=ParsedTask,
        system_prompt="Parse user request"
    )
    .with_temperature(0.7)
    .with_max_tokens(2000)
    .with_model('gpt-4o')
    .as_cacheable(ttl=300)
    .with_retry(max_attempts=3, initial_delay=1.0)
)
""")

print("\n✅ BENEFITS:")
print("   - Configuration separate from implementation")
print("   - Chainable methods (fluent API)")
print("   - Self-documenting")
print("   - Easy to modify/extend")
print("   - Type-safe with validation")


# ============================================================================
# Example 3: Tool Execution Workflow
# ============================================================================

print("\n" + "="*80)
print("COMPARISON 3: Tool Execution with Parallel Execution")
print("="*80)

print("\n" + "-"*80)
print("BEFORE: Manual tool orchestration (60+ lines)")
print("-"*80)

print("""
class ToolExecutorAgent_OLD(Agent):
    def __init__(self, llm: LLM, tools: list[Tool]):
        PROMPT = "Use available tools to fulfill the request."
        tool_set = ToolSet(tools=tools)
        
        async def handler(input_data, context: str, execution_state: ExecutionState):
            # LLM call with ExecutionEvent
            llm_event = ExecutionEvent(actor=llm)
            llm_input = OpenAIRequest(
                system_message=PROMPT,
                prompt=input_data.content,
                model='gpt-4o',
                max_completion_tokens=2500,
                temperature=1.0,
                tools=tool_set
            )
            
            execution_result = await llm_event(llm_input, context, execution_state)
            if not execution_result.ok:
                raise Exception(f"LLM failed: {execution_result.erro_message}")
            
            response = execution_result.result
            
            # Manual parallel tool execution
            all_results = {}
            execution_order = []
            errors = {}
            
            if response.is_tool_call:
                async def execute_tool(tool_call):
                    tool = tool_set[tool_call.tool_name]
                    if tool is None:
                        return (tool_call.tool_name, None, "Tool not found")
                    
                    try:
                        tool_event = ExecutionEvent(actor=tool)
                        tool_input = tool.input_model.model_validate_json(
                            tool_call.arguments
                        )
                        result = await tool_event(tool_input, context, execution_state)
                        
                        if not result.ok:
                            return (tool_call.tool_name, None, result.erro_message)
                        
                        return (tool_call.tool_name, result.result.output, None)
                    except Exception as e:
                        return (tool_call.tool_name, None, str(e))
                
                # Execute in parallel with asyncio.gather
                tasks = [execute_tool(tc) for tc in response.tools]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for item in results:
                    if isinstance(item, Exception):
                        continue
                    tool_name, output, error = item
                    execution_order.append(tool_name)
                    if error:
                        errors[tool_name] = error
                    else:
                        all_results[tool_name] = output
            
            # Create result object manually
            from rh_agents.core.result_types import ToolExecutionResult
            return ToolExecutionResult(
                results=all_results,
                execution_order=execution_order,
                errors=errors
            )
        
        super().__init__(
            name="ToolExecutor_OLD",
            description=PROMPT,
            input_model=Message,
            output_model=ToolExecutionResult,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=tool_set
        )
""")

print("\n" + "-"*80)
print("AFTER: Builder handles everything (5 lines)")
print("-"*80)

print("""
# Parallel execution, error handling, aggregation - all automatic
executor_NEW = await ToolExecutorAgent.from_tools(
    name="ToolExecutor",
    llm=llm,
    input_model=Message,
    output_model=ToolExecutionResult,
    system_prompt="Use available tools to fulfill the request.",
    tools=[tool1, tool2, tool3]
)
""")

print("\n✅ SAVINGS: 60+ lines → 5 lines (92% reduction)")
print("   - Parallel execution automatic")
print("   - Error aggregation automatic")
print("   - ExecutionEvent management automatic")
print("   - Result formatting automatic")


# ============================================================================
# Example 4: Complete Feature Comparison
# ============================================================================

print("\n" + "="*80)
print("COMPARISON 4: Adding Advanced Features")
print("="*80)

print("\n" + "-"*80)
print("BEFORE: Everything manual")
print("-"*80)

print("""
# Want dynamic prompts? Write custom logic in handler
# Want retry? Add RetryConfig to __init__
# Want different error strategies? Add if/else in handler
# Want result aggregation? Write custom aggregation code
# Want caching? Set cacheable flag
# Total: Scattered across 100+ lines
""")

print("\n" + "-"*80)
print("AFTER: Just chain methods")
print("-"*80)

print("""
# Every feature is a simple method call
agent = (
    await ToolExecutorAgent.from_tools(...)
    .with_system_prompt_builder(custom_prompt_fn)      # Dynamic prompts
    .with_temperature(0.7)                             # LLM config
    .with_max_tokens(2000)
    .with_error_strategy(ErrorStrategy.RETURN_NONE)    # Error handling
    .with_retry(max_attempts=3, initial_delay=1.0)     # Retry logic
    .with_aggregation(AggregationStrategy.LIST)        # Result aggregation
    .as_cacheable(ttl=300)                             # Caching
    .as_artifact()                                     # Artifact storage
)
# Total: 11 readable, type-safe lines
""")

print("\n✅ BENEFITS:")
print("   - Every feature is one line")
print("   - Features compose naturally")
print("   - Self-documenting code")
print("   - Type-safe with validation")
print("   - Easy to add/remove features")


# ============================================================================
# Metrics Summary
# ============================================================================

print("\n" + "="*80)
print("QUANTITATIVE COMPARISON")
print("="*80)

comparison_data = [
    ("Structured Output", 42, 5, "88%"),
    ("Simple Completion", 28, 5, "82%"),
    ("Tool Execution", 65, 5, "92%"),
    ("With Configuration", 55, 11, "80%"),
    ("Full-Featured Agent", 120, 15, "88%"),
]

print("\n{:<25} {:>12} {:>12} {:>15}".format("Pattern", "Before", "After", "Reduction"))
print("-" * 70)
for name, before, after, reduction in comparison_data:
    print("{:<25} {:>9} → {:>9} {:>15}".format(
        name,
        f"{before} lines",
        f"{after} lines",
        reduction
    ))

print("\n" + "="*80)
print("KEY ADVANTAGES OF BUILDER PATTERN")
print("="*80)

print("""
1. MASSIVE BOILERPLATE REDUCTION
   - 80-92% less code to write
   - 40-120 lines → 5-15 lines
   
2. BETTER SEPARATION OF CONCERNS
   - Implementation vs. configuration clearly separated
   - Each concern has its own method
   
3. TYPE SAFETY & VALIDATION
   - All parameters validated at configuration time
   - Clear error messages for invalid config
   - IDE autocomplete support
   
4. SELF-DOCUMENTING CODE
   - Configuration reads like documentation
   - Intent is clear from method names
   - No need to dig through handler code
   
5. EASIER TO MAINTAIN
   - Add feature = add one line
   - Remove feature = remove one line  
   - No scattered changes across codebase
   
6. FASTER DEVELOPMENT
   - Write agents in minutes, not hours
   - Less code = fewer bugs
   - Standard patterns = less thinking
   
7. CONSISTENT PATTERNS
   - All agents use same configuration API
   - Easy to learn once, apply everywhere
   - Team communication improves
""")

print("\n" + "="*80)
print("MIGRATION GUIDE")
print("="*80)

print("""
Step 1: Identify the pattern your agent follows
   - Structured output? → Use StructuredAgent
   - Simple completion? → Use CompletionAgent
   - Tool execution? → Use ToolExecutorAgent
   - Direct tool call? → Use DirectToolAgent

Step 2: Replace __init__ with builder factory
   - from_model() for StructuredAgent
   - from_prompt() for CompletionAgent
   - from_tools() for ToolExecutorAgent
   - from_tool() for DirectToolAgent

Step 3: Move configuration to chainable methods
   - Extract LLM params → .with_temperature(), .with_max_tokens()
   - Extract caching → .as_cacheable()
   - Extract retry → .with_retry()
   - Extract error handling → .with_error_strategy()

Step 4: Remove handler boilerplate
   - No more ExecutionEvent creation
   - No more OpenAIRequest construction
   - No more tool call extraction
   - No more error handling code

Step 5: Test and verify
   - Same inputs should produce same outputs
   - Configuration should match original behavior
   - Test error handling and retry logic

Result: 80-90% less code, same functionality, better maintainability!
""")

print("\n✓ See BUILDERS_GUIDE.md for detailed migration examples")
print("✓ See CONFIGURATION_GUIDE.md for all configuration options")
print()


async def main():
    """Main function placeholder."""
    print("\n[This is a comparison document - no executable code]")
    print("[Run builder_basic.py or builder_advanced.py for working examples]")


if __name__ == "__main__":
    asyncio.run(main())
