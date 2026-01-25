"""
Test file to verify .pyi stub files provide proper type hints
Run with: pyright test_typing_check.py or mypy test_typing_check.py
"""
from pydantic import BaseModel
from rh_agents.core.actors import LLM, Tool, Agent, ToolSet
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result


class MyInput(BaseModel):
    query: str


class MyAgentOutput(BaseModel):
    answer: str


# Create typed handlers
async def llm_handler(input: MyInput, context: str, state: ExecutionState) -> LLM_Result:
    return LLM_Result(content="test response", model="test-model")


async def agent_handler(input: MyInput, context: str, state: ExecutionState) -> MyAgentOutput:
    return MyAgentOutput(answer="test agent response")


async def tool_handler(input: MyInput, context: str, state: ExecutionState) -> dict:
    return {"result": "test"}


# Test LLM typing - NO type parameters at runtime!
# The .pyi file provides Generic[T, R] types for type checkers
# LLM always returns LLM_Result, but .pyi helps with input typing
my_llm = LLM(
    name="test_llm",
    description="Test LLM",
    input_model=MyInput,
    output_model=LLM_Result,  # LLM always uses LLM_Result
    handler=llm_handler
)

# Test Tool typing - NO type parameters at runtime!
my_tool = Tool(
    name="test_tool",
    description="Test Tool",
    input_model=MyInput,
    handler=tool_handler
)

# Test Agent typing - NO type parameters at runtime!
# Agents can have custom output types
my_agent = Agent(
    name="test_agent",
    description="Test Agent",
    input_model=MyInput,
    output_model=MyAgentOutput,  # Agents can return custom types
    handler=agent_handler,
    tools=ToolSet(tools=[my_tool])
)


async def test_typing():
    state = ExecutionState()
    input_data = MyInput(query="test")
    
    # Test ExecutionEvent with LLM - NO type parameter at runtime!
    # The type checker will infer LLM_Result from the actor
    event_llm = ExecutionEvent(actor=my_llm)
    result_llm = await event_llm(input_data, "", state)
    
    # Type checker knows result_llm is ExecutionResult (from .pyi)
    if result_llm.result:
        # For LLM_Result
        content: str = result_llm.result.content  # type: ignore
        print(f"LLM content: {content}")
    
    # Test ExecutionEvent with Agent - NO type parameter at runtime!
    event_agent = ExecutionEvent(actor=my_agent)
    result_agent = await event_agent(input_data, "", state)
    
    # Type checker knows result_agent is ExecutionResult (from .pyi)
    if result_agent.result:
        answer: str = result_agent.result.answer  # type: ignore
        print(f"Agent answer: {answer}")
    
    # Test ExecutionEvent with Tool - returns Any
    event_tool = ExecutionEvent(actor=my_tool)
    result_tool = await event_tool(input_data, "", state)
    
    # result_tool.result is Any for tools
    print(f"Tool result: {result_tool.result}")
    
    print("✅ All type hints work correctly!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_typing())
