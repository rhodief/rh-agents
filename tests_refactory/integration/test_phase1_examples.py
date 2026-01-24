"""
Phase 1 Integration Tests: Verify Examples Work

Tests that key examples work correctly with new import style.
"""
import pytest
import asyncio
from pydantic import BaseModel
from rh_agents import (
    Tool,
    Agent,
    LLM,
    ExecutionState,
    ExecutionEvent,
    ToolSet,
    Tool_Result,
    LLM_Result,
    Message,
    AuthorType
)


class TestInput(BaseModel):
    value: str


class TestExampleBasicUsage:
    """Test basic usage patterns from examples."""
    
    def test_create_tool_with_new_imports(self):
        """Test creating a tool using top-level imports."""
        
        async def handler(input: TestInput, state: ExecutionState, context: str = "") -> Tool_Result:
            return Tool_Result(output=input.value, tool_name="test")
        
        tool = Tool(
            name="TestTool",
            description="A test tool",
            input_model=TestInput,
            handler=handler
        )
        
        assert tool.name == "TestTool"
        assert tool.description == "A test tool"
        assert tool.input_model == TestInput
    
    def test_create_agent_with_new_imports(self):
        """Test creating an agent using top-level imports."""
        
        async def agent_handler(input: Message, state: ExecutionState, context: str = "") -> Message:
            return Message(content="Response", author=AuthorType.ASSISTANT)
        
        agent = Agent(
            name="TestAgent",
            description="A test agent",
            input_model=Message,
            output_model=Message,
            handler=agent_handler,
            tools=ToolSet(tools=[])
        )
        
        assert agent.name == "TestAgent"
        assert agent.tools is not None
    
    def test_create_llm_with_new_imports(self):
        """Test creating an LLM using top-level imports."""
        
        async def llm_handler(input: BaseModel, state: ExecutionState, context: str = "") -> LLM_Result:
            return LLM_Result(content="LLM response")
        
        llm = LLM(
            name="TestLLM",
            description="A test LLM",
            input_model=BaseModel,
            handler=llm_handler
        )
        
        assert llm.name == "TestLLM"
        assert llm.output_model == LLM_Result


class TestExecutionFlow:
    """Test execution flow patterns from examples."""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing a tool with ExecutionState."""
        
        async def handler(input: TestInput, state: ExecutionState, context: str = "") -> Tool_Result:
            return Tool_Result(
                output=f"Processed: {input.value}",
                tool_name="processor"
            )
        
        tool = Tool(
            name="Processor",
            description="Processes input",
            input_model=TestInput,
            handler=handler
        )
        
        state = ExecutionState()
        event = ExecutionEvent(actor=tool)
        
        result = await event(TestInput(value="test"), "", state)
        
        assert result.ok is True
        assert result.result.output == "Processed: test"
    
    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        """Test agent with tools pattern from examples."""
        
        async def tool_handler(input: TestInput, state: ExecutionState, context: str = "") -> Tool_Result:
            return Tool_Result(output=input.value.upper(), tool_name="uppercase")
        
        tool = Tool(
            name="Uppercase",
            description="Converts to uppercase",
            input_model=TestInput,
            handler=tool_handler
        )
        
        async def agent_handler(input: Message, state: ExecutionState, context: str = "") -> Message:
            # Simple agent that just returns a message
            return Message(content="Agent processed", author=AuthorType.ASSISTANT)
        
        agent = Agent(
            name="TestAgent",
            description="Test agent with tools",
            input_model=Message,
            output_model=Message,
            handler=agent_handler,
            tools=ToolSet(tools=[tool])
        )
        
        assert len(agent.tools.tools) == 1
        assert agent.tools["Uppercase"] is not None
        assert agent.tools.get("Uppercase").name == "Uppercase"


class TestStateManagement:
    """Test state management patterns from examples."""
    
    @pytest.mark.asyncio
    async def test_execution_state_creation(self):
        """Test creating ExecutionState as in examples."""
        state = ExecutionState()
        
        assert state.state_id is not None
        assert state.history is not None
        assert state.storage is not None
    
    @pytest.mark.asyncio
    async def test_multiple_tool_executions(self):
        """Test multiple tool executions sharing state."""
        
        async def tool1_handler(input: TestInput, context: str, state: ExecutionState) -> Tool_Result:
            state.storage.set("tool1_ran", "True")
            return Tool_Result(output="tool1", tool_name="tool1")
        
        async def tool2_handler(input: TestInput, context: str, state: ExecutionState) -> Tool_Result:
            tool1_ran = state.storage.get("tool1_ran")
            return Tool_Result(output=f"tool2 (tool1_ran={tool1_ran})", tool_name="tool2")
        
        tool1 = Tool(
            name="Tool1",
            description="First tool",
            input_model=TestInput,
            handler=tool1_handler
        )
        
        tool2 = Tool(
            name="Tool2",
            description="Second tool",
            input_model=TestInput,
            handler=tool2_handler
        )
        
        state = ExecutionState()
        
        result1 = await ExecutionEvent(actor=tool1)(TestInput(value="test"), "", state)
        assert result1.ok is True
        
        result2 = await ExecutionEvent(actor=tool2)(TestInput(value="test"), "", state)
        assert result2.ok is True
        assert "tool1_ran=True" in result2.result.output
