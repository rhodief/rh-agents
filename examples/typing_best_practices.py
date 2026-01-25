"""
Enhanced typing example showing best practices with .pyi files
"""
from typing import TYPE_CHECKING
from pydantic import BaseModel
from rh_agents.core.actors import LLM, Tool, Agent, ToolSet
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result

if TYPE_CHECKING:
    # Type checking imports - these provide better hints
    pass


class QueryInput(BaseModel):
    query: str


class AnalysisOutput(BaseModel):
    answer: str
    confidence: float


# Create actors
async def llm_handler(input: QueryInput, context: str, state: ExecutionState) -> LLM_Result:
    return LLM_Result(content=f"Processed: {input.query}", model_name="gpt-4")


async def agent_handler(input: QueryInput, context: str, state: ExecutionState) -> AnalysisOutput:
    return AnalysisOutput(answer="Analysis complete", confidence=0.95)


my_llm = LLM(
    name="query_llm",
    description="Query processing LLM",
    input_model=QueryInput,
    output_model=LLM_Result,
    handler=llm_handler
)

my_agent = Agent(
    name="analysis_agent",
    description="Analysis agent",
    input_model=QueryInput,
    output_model=AnalysisOutput,
    handler=agent_handler,
    tools=ToolSet(tools=[])
)


async def example_with_explicit_types():
    """
    BEST PRACTICE: Use explicit type annotations
    This gives you full type safety and autocomplete
    """
    state = ExecutionState()
    input_data = QueryInput(query="What is AI?")
    
    # Method 1: Explicit type annotation (RECOMMENDED)
    event_llm = ExecutionEvent(actor=my_llm)
    result_llm: ExecutionResult[LLM_Result] = await event_llm(input_data, "", state)
    
    # ✅ Type checker knows result_llm.result is LLM_Result | None
    if result_llm.result:
        content: str = result_llm.result.content  # Full autocomplete!
        model_name: str | None = result_llm.result.model_name
        print(f"LLM ({model_name}): {content}")
    
    # Method 2: Explicit type annotation for agents
    event_agent = ExecutionEvent(actor=my_agent)
    result_agent: ExecutionResult[AnalysisOutput] = await event_agent(input_data, "", state)
    
    # ✅ Type checker knows result_agent.result is AnalysisOutput | None
    if result_agent.result:
        answer: str = result_agent.result.answer  # Full autocomplete!
        confidence: float = result_agent.result.confidence
        print(f"Agent: {answer} (confidence: {confidence})")


async def example_without_explicit_types():
    """
    Without explicit types, you get less type safety
    But runtime behavior is identical
    """
    state = ExecutionState()
    input_data = QueryInput(query="What is AI?")
    
    # No type annotation
    event = ExecutionEvent(actor=my_llm)
    result = await event(input_data, "", state)
    
    # Type checker sees: ExecutionResult[Unknown] or ExecutionResult[Any]
    # Runtime works fine, but IDE won't give autocomplete
    if result.result:
        # Still works at runtime!
        print(f"Result: {result.result}")


async def example_inline_typing():
    """
    You can also annotate inline
    """
    state = ExecutionState()
    input_data = QueryInput(query="What is AI?")
    
    # Inline type annotation
    result = await ExecutionEvent(actor=my_llm)(input_data, "", state)
    
    # Cast to specific type if needed
    if result.result and isinstance(result.result, LLM_Result):
        # Runtime type check + type narrowing
        content: str = result.result.content
        print(f"Content: {content}")


if __name__ == "__main__":
    import asyncio
    
    print("=== Example with explicit types ===")
    asyncio.run(example_with_explicit_types())
    
    print("\n=== Example without explicit types ===")
    asyncio.run(example_without_explicit_types())
    
    print("\n=== Example with inline typing ===")
    asyncio.run(example_inline_typing())
    
    print("\n✅ All examples work! Explicit typing gives better IDE support.")
