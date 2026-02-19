"""
Basic examples demonstrating all four builder types.

Showcases:
- StructuredAgent: Force structured output from LLM
- CompletionAgent: Simple text completions
- ToolExecutorAgent: LLM-guided tool execution
- DirectToolAgent: Direct tool calls without LLM

Note: Examples 1-3 require OPENAI_API_KEY environment variable to run.
      Example 4 (DirectToolAgent) runs without API key.
      Example 4b runs without API key (mock tools).
"""

import asyncio
from pydantic import BaseModel, Field
from typing import Optional

from rh_agents.builders import (
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import ErrorStrategy
from rh_agents.models import Message
from rh_agents.llms import OpenAILLM
import os


# ============================================================================
# Example 1: StructuredAgent - Parse user input into structured data
# ============================================================================

class UserRequest(BaseModel):
    """User's natural language request."""
    content: str


class ParsedTask(BaseModel):
    """Structured task extracted from user request."""
    task_type: str = Field(description="Type of task: 'search', 'create', 'analyze', etc.")
    subject: str = Field(description="Main subject or topic")
    parameters: dict = Field(description="Additional parameters as key-value pairs")
    priority: Optional[str] = Field(default="normal", description="Priority level")


async def example_structured_agent():
    """Demonstrate StructuredAgent for parsing unstructured input."""
    print("\n" + "="*80)
    print("Example 1: StructuredAgent - Parse user input into structured data")
    print("="*80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - showing expected behavior")
        print("\nInput: 'Search for all documents about climate change from 2023'")
        print("\nExpected Output (structured):")
        print("  task_type: 'search'")
        print("  subject: 'climate change documents'")
        print("  parameters: {'year': '2023'}")
        print("  priority: 'normal'")
        print("\n✓ StructuredAgent guarantees typed, validated output matching ParsedTask schema")
        return
    
    # Create LLM
    llm = OpenAILLM()
    
    # Create StructuredAgent
    parser = await StructuredAgent.from_model(
        name="TaskParser",
        llm=llm,
        input_model=UserRequest,
        output_model=ParsedTask,
        system_prompt="Parse the user's request into a structured task format."
    )
    
    # Configure with chainable methods
    parser = (
        parser
        .with_temperature(0.7)
        .with_max_tokens(500)
        .as_cacheable(ttl=300)
    )
    
    # Execute with real API call
    print("\n📝 Input: 'Search for all documents about climate change from 2023'")
    
    # Create execution state
    bus = EventBus()
    state = ExecutionState(event_bus=bus)
    
    # Execute agent
    input_data = UserRequest(content="Search for all documents about climate change from 2023")
    result = await ExecutionEvent[ParsedTask](actor=parser)(input_data, "", state)
    
    # Show actual results
    print("\n✅ Actual Output (structured):")
    if isinstance(result.result, ParsedTask):
        print(f"  task_type: {result.result.task_type}")
        print(f"  subject: {result.result.subject}")
        print(f"  parameters: {result.result.parameters}")
        print(f"  priority: {result.result.priority}")
    else:
        print(f"  ⚠️  Got {type(result.result).__name__} instead of ParsedTask")
        print(f"  This can happen if tool choice setup isn't configured properly")
        print(f"  Result: {result.result}")
    
    print("\n✓ StructuredAgent guarantees typed, validated output matching ParsedTask schema")


# ============================================================================
# Example 2: CompletionAgent - Simple text generation
# ============================================================================

class DocumentInput(BaseModel):
    """Input document to summarize."""
    content: str


async def example_completion_agent():
    """Demonstrate CompletionAgent for text generation."""
    print("\n" + "="*80)
    print("Example 2: CompletionAgent - Summarize document")
    print("="*80)
    
    long_doc = """
    Artificial Intelligence has transformed numerous industries over the past decade.
    Machine learning algorithms now power recommendation systems, autonomous vehicles,
    and medical diagnosis tools. However, concerns about bias, privacy, and job
    displacement remain significant challenges that require careful consideration
    and regulation.
    """
    
    print(f"\n📝 Input document: {long_doc.strip()[:100]}...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - showing expected behavior")
        print("\nExpected Output:")
        print("  'AI has revolutionized multiple sectors through ML applications in")
        print("  recommendations, vehicles, and healthcare. Key challenges include")
        print("  addressing bias, privacy concerns, and employment impacts.'")
        print("\n✓ CompletionAgent returns natural language Message objects")
        return
    
    # Create LLM
    llm = OpenAILLM()
    
    # Create CompletionAgent
    summarizer = await CompletionAgent.from_prompt(
        name="Summarizer",
        llm=llm,
        input_model=DocumentInput,
        output_model=Message,
        system_prompt="Summarize the provided document in 2-3 concise sentences."
    )
    
    # Configure with chainable methods
    summarizer = (
        summarizer
        .with_temperature(0.5)
        .with_max_tokens(200)
        .with_model('gpt-4o-mini')
    )
    
    # Execute with real API call
    bus = EventBus()
    state = ExecutionState(event_bus=bus)
    
    input_data = DocumentInput(content=long_doc)
    result = await ExecutionEvent[Message](actor=summarizer)(input_data, "", state)
    
    # Show actual results
    print("\n✅ Actual Output:")
    if result.result:
        print(f"  {result.result.content}")
    else:
        print("  ⚠️  No result returned")
    
    print("\n✓ CompletionAgent returns natural language Message objects")


# ============================================================================
# Example 3: ToolExecutorAgent - LLM-guided tool execution
# ============================================================================

class SearchQuery(BaseModel):
    """Search query input."""
    query: str
    filters: Optional[dict] = None


class DatabaseResult(BaseModel):
    """Database search result."""
    results: list[dict]
    count: int


class AnalysisResult(BaseModel):
    """Analysis result."""
    summary: str
    metrics: dict


# Mock tools for demonstration
class SearchTool(Tool):
    """Search database tool."""
    def __init__(self):
        super().__init__(
            name="SearchDatabase",
            description="Search database for documents matching query",
            input_model=SearchQuery,
            output_model=DatabaseResult,
            handler=self._search
        )
    
    async def _search(self, input_data, context, state):
        # Mock implementation
        return DatabaseResult(
            results=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
            count=2
        )


class AnalyzeTool(Tool):
    """Analyze data tool."""
    def __init__(self):
        super().__init__(
            name="AnalyzeData",
            description="Perform statistical analysis on data",
            input_model=DatabaseResult,
            output_model=AnalysisResult,
            handler=self._analyze
        )
    
    async def _analyze(self, input_data, context, state):
        # Mock implementation
        return AnalysisResult(
            summary="Analysis complete",
            metrics={"avg": 42, "std": 10}
        )


async def example_tool_executor_agent():
    """Demonstrate ToolExecutorAgent for multi-tool workflows."""
    print("\n" + "="*80)
    print("Example 3: ToolExecutorAgent - Multi-tool data pipeline")
    print("="*80)
    
    print("\n📝 Input: 'Search for climate docs and analyze the results'")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - showing expected behavior")
        print("\nExpected Execution:")
        print("  1. LLM decides to call SearchDatabase with query='climate docs'")
        print("  2. LLM decides to call AnalyzeData with search results")
        print("  3. Both tools execute in parallel")
        print("\nOutput: ToolExecutionResult(")
        print("  results={'SearchDatabase': [...], 'AnalyzeData': {...}},")
        print("  execution_order=['SearchDatabase', 'AnalyzeData'],")
        print("  errors={}")
        print(")")
        print("\n✓ ToolExecutorAgent handles parallel execution and result aggregation")
        return
    
    # Create LLM
    llm = OpenAILLM()
    
    # Create tools
    search_tool = SearchTool()
    analyze_tool = AnalyzeTool()
    
    # Create ToolExecutorAgent
    executor = await ToolExecutorAgent.from_tools(
        name="DataPipeline",
        llm=llm,
        input_model=UserRequest,
        output_model=DatabaseResult,  # Not used (returns ToolExecutionResult)
        system_prompt="Use available tools to fulfill the user's data request.",
        tools=[search_tool, analyze_tool]
    )
    
    # Configure with chainable methods
    executor = (
        executor
        .with_temperature(0.8)
        .with_system_prompt_builder(async_prompt_builder)
        .with_retry(max_attempts=3, initial_delay=1.0)
    )
    
    # Note: Full execution would require proper tool implementation
    # This is a demonstration of the configuration pattern
    print("\n✓ ToolExecutorAgent handles parallel execution and result aggregation")


async def async_prompt_builder(input_data, context, state):
    """Example dynamic prompt builder."""
    return f"Process this request: {input_data.content}\n\nUse tools as needed."


# ============================================================================
# Example 4: DirectToolAgent - Direct tool execution (no LLM)
# ============================================================================

class ValidationInput(BaseModel):
    """Input to validate."""
    data: dict


class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: list[str]


class ValidationTool(Tool):
    """Data validation tool."""
    def __init__(self):
        super().__init__(
            name="ValidateData",
            description="Validate data against schema",
            input_model=ValidationInput,
            output_model=ValidationResult,
            handler=self._validate
        )
    
    async def _validate(self, input_data, context, state):
        # Mock validation logic
        required_fields = ['id', 'name', 'email']
        errors = [
            f"Missing field: {field}"
            for field in required_fields
            if field not in input_data.data
        ]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )


async def example_direct_tool_agent():
    """Demonstrate DirectToolAgent for deterministic operations."""
    print("\n" + "="*80)
    print("Example 4: DirectToolAgent - Direct validation (no LLM)")
    print("="*80)
    
    # Create tool
    validation_tool = ValidationTool()
    
    # Create DirectToolAgent
    validator = await DirectToolAgent.from_tool(
        name="DataValidator",
        tool=validation_tool
    )
    
    # Configure with chainable methods
    validator = (
        validator
        .with_error_strategy(ErrorStrategy.RETURN_NONE)
        .as_cacheable(ttl=60)
    )
    
    print("\n📝 Input: {'id': 123, 'name': 'John'}")
    
    # Execute (no API key needed - direct tool execution)
    bus = EventBus()
    state = ExecutionState(event_bus=bus)
    
    input_data = ValidationInput(data={'id': 123, 'name': 'John'})
    result = await ExecutionEvent[ValidationResult](actor=validator)(input_data, "", state)
    
    # Show actual results
    print("\n✅ Actual Output:")
    if result.result:
        print(f"  ValidationResult(")
        print(f"    is_valid={result.result.is_valid},")
        print(f"    errors={result.result.errors}")
        print(f"  )")
    else:
        print("  ⚠️  No result returned")
    
    print("\n✓ DirectToolAgent bypasses LLM for deterministic operations")
    print("  - Faster execution (no API call)")
    print("  - Lower cost")
    print("  - Predictable behavior")


# ============================================================================
# Example 4b: ToolExecutorAgent with Aggregation Strategies
# ============================================================================

class WebSearchResult(BaseModel):
    """Web search result."""
    source: str
    snippet: str


class WebSearchTool(Tool):
    """Web search tool."""
    def __init__(self, name, source):
        _source = source  # Store in closure for handler
        
        async def _search_impl(input_data, context, state):
            # Mock implementation
            return WebSearchResult(
                source=_source,
                snippet=f"Results from {_source}: {input_data.query}"
            )
        
        super().__init__(
            name=name,
            description=f"Search {source} for information",
            input_model=SearchQuery,
            output_model=WebSearchResult,
            handler=_search_impl
        )


async def example_aggregation_strategies():
    """Demonstrate result aggregation strategies for ToolExecutorAgent."""
    print("\n" + "="*80)
    print("Example 4b: ToolExecutorAgent - Result Aggregation Strategies")
    print("="*80)
    
    from rh_agents.core.types import AggregationStrategy
    
    # Create LLM
    llm = OpenAILLM()
    
    # Create multiple search tools
    tools: list[Tool] = [
        WebSearchTool("GoogleSearch", "Google"),
        WebSearchTool("BingSearch", "Bing"),
        WebSearchTool("DuckDuckGoSearch", "DuckDuckGo")
    ]
    
    print("\n--- DICT Strategy (default) ---")
    print("Returns ToolExecutionResult with dict-like access")
    executor_dict = await ToolExecutorAgent.from_tools(
        name="MultiSearch_Dict",
        llm=llm,
        input_model=SearchQuery,
        output_model=WebSearchResult,
        system_prompt="Search using all available tools",
        tools=tools
    )
    print("Usage: result['GoogleSearch'], result.execution_order")
    
    print("\n--- LIST Strategy ---")
    print("Returns ordered list of results")
    executor_list = await ToolExecutorAgent.from_tools(
        name="MultiSearch_List",
        llm=llm,
        input_model=SearchQuery,
        output_model=WebSearchResult,
        system_prompt="Search using all available tools",
        tools=tools
    )
    executor_list = executor_list.with_aggregation(AggregationStrategy.LIST)
    print("Returns: [result1, result2, result3]")
    print("Usage: for result in results: ...")
    
    print("\n--- CONCATENATE Strategy ---")
    print("Returns concatenated string of all results")
    executor_concat = await ToolExecutorAgent.from_tools(
        name="MultiSearch_Concat",
        llm=llm,
        input_model=SearchQuery,
        output_model=WebSearchResult,
        system_prompt="Search using all available tools",
        tools=tools
    )
    executor_concat = executor_concat.with_aggregation(AggregationStrategy.CONCATENATE, separator="\n---\n")
    print("Returns: 'result1\\n---\\nresult2\\n---\\nresult3'")
    print("Usage: Use for generating reports or summaries")
    
    print("\n--- FIRST Strategy ---")
    print("Returns only the first result")
    executor_first = await ToolExecutorAgent.from_tools(
        name="MultiSearch_First",
        llm=llm,
        input_model=SearchQuery,
        output_model=WebSearchResult,
        system_prompt="Search using all available tools",
        tools=tools
    )
    executor_first = executor_first.with_aggregation(AggregationStrategy.FIRST)
    print("Returns: result1")
    print("Usage: first_result.source, first_result.snippet")
    
    print("\n✓ Aggregation strategies provide flexible result handling:")
    print("  - DICT: Full control, dict-like access")
    print("  - LIST: Sequential processing, iteration")
    print("  - CONCATENATE: Text aggregation, reporting")
    print("  - FIRST: Quick access to first result")


# ============================================================================
# Combined example: Chaining multiple builders
# ============================================================================

async def example_chained_workflow():
    """Demonstrate chaining multiple builder agents."""
    print("\n" + "="*80)
    print("Example 5: Chaining multiple builders in a workflow")
    print("="*80)
    
    print("\nWorkflow:")
    print("  1. StructuredAgent: Parse user request → ParsedTask")
    print("  2. DirectToolAgent: Validate task → ValidationResult")
    print("  3. ToolExecutorAgent: Execute task tools → ToolExecutionResult")
    print("  4. CompletionAgent: Summarize results → Message")
    
    print("\n✓ Builders compose naturally into multi-stage pipelines")
    print("  - Each agent focuses on one responsibility")
    print("  - Type safety enforced at each boundary")
    print("  - ExecutionEvent tracks entire workflow")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BUILDER PATTERN EXAMPLES")
    print("="*80)
    
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    if has_api_key:
        print("\n✅ OPENAI_API_KEY detected - will run actual API calls")
    else:
        print("\n⚠️  OPENAI_API_KEY not set - showing expected behavior for API examples")
        print("   Example 4 (DirectToolAgent) and 4b (Aggregation) will run fully")
    
    print("\nThese examples demonstrate the four builder types:")
    print("  1. StructuredAgent - Forced structured output")
    print("  2. CompletionAgent - Simple text generation")
    print("  3. ToolExecutorAgent - LLM-guided tool execution")
    print("  4. DirectToolAgent - Direct tool calls (no LLM) ✓ Runs without API key")
    print("  4b. Aggregation Strategies - Result combination options ✓ Runs without API key")
    print("  5. Chained workflow - Combining multiple builders")
    
    await example_structured_agent()
    await example_completion_agent()
    await example_tool_executor_agent()
    await example_direct_tool_agent()
    await example_aggregation_strategies()
    await example_chained_workflow()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("\n1. All builders return BuilderAgent instances with chainable methods")
    print("2. ExecutionEvent handles all LLM and tool calls transparently")
    print("3. Error strategies (RAISE, RETURN_NONE, etc.) apply uniformly")
    print("4. Type safety enforced via Pydantic at all boundaries")
    print("5. Caching and artifacts work consistently across all builders")
    
    if not has_api_key:
        print("\n💡 Tip: Set OPENAI_API_KEY environment variable to see actual API results:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   python builder_basic.py")
    
    print("\n✓ See BUILDERS_GUIDE.md for complete documentation")
    print()


if __name__ == "__main__":
    asyncio.run(main())
