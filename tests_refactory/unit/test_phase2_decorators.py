"""
Phase 2 Unit Tests: Decorator API

Tests the @tool and @agent decorators for simplified actor creation.
"""
import pytest
from pydantic import BaseModel
from rh_agents import tool_decorator, agent_decorator, ExecutionState
from rh_agents.core.result_types import Tool_Result, LLM_Result
from rh_agents.core.actors import Tool, Agent


class SimpleInput(BaseModel):
    value: str


class SimpleOutput(BaseModel):
    result: str


class TestToolDecorator:
    """Test the @tool decorator."""
    
    def test_tool_decorator_basic(self):
        """Test creating a tool with decorator."""
        @tool_decorator(name="test_tool", description="A test tool")
        async def my_tool(input: SimpleInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output=f"processed: {input.value}", tool_name="test_tool")
        
        assert isinstance(my_tool, Tool)
        assert my_tool.name == "test_tool"
        assert my_tool.description == "A test tool"
        assert my_tool.input_model == SimpleInput
        assert my_tool.cacheable is False
        assert my_tool.version == "1.0.0"
    
    def test_tool_decorator_defaults(self):
        """Test tool decorator uses function name and docstring as defaults."""
        @tool_decorator()
        async def calculate(input: SimpleInput, context: str, state: ExecutionState) -> Tool_Result:
            """Performs calculations."""
            return Tool_Result(output="calculated", tool_name="calculate")
        
        assert calculate.name == "calculate"
        assert "Performs calculations" in calculate.description
    
    def test_tool_decorator_with_cacheable(self):
        """Test creating cacheable tool."""
        @tool_decorator(name="cached_tool", cacheable=True, version="2.0.0")
        async def my_tool(input: SimpleInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="cached_tool")
        
        assert my_tool.cacheable is True
        assert my_tool.version == "2.0.0"
    
    def test_tool_decorator_validates_signature(self):
        """Test that decorator validates handler signature."""
        with pytest.raises(ValueError, match="must have at least one parameter"):
            @tool_decorator(name="bad_tool")
            async def bad_tool() -> Tool_Result:
                return Tool_Result(output="result", tool_name="bad_tool")
    
    def test_tool_decorator_requires_type_annotation(self):
        """Test that decorator requires input parameter to be type-annotated."""
        with pytest.raises(ValueError, match="must be type-annotated"):
            @tool_decorator(name="bad_tool")
            async def bad_tool(input, context: str, state: ExecutionState) -> Tool_Result:
                return Tool_Result(output="result", tool_name="bad_tool")


class TestAgentDecorator:
    """Test the @agent decorator."""
    
    def test_agent_decorator_basic(self):
        """Test creating an agent with decorator."""
        @agent_decorator(name="test_agent", description="A test agent")
        async def my_agent(input: SimpleInput, context: str, state: ExecutionState) -> SimpleOutput:
            return SimpleOutput(result=f"processed: {input.value}")
        
        assert isinstance(my_agent, Agent)
        assert my_agent.name == "test_agent"
        assert my_agent.description == "A test agent"
        assert my_agent.input_model == SimpleInput
        assert my_agent.cacheable is False
    
    def test_agent_decorator_with_tools_and_llm(self):
        """Test agent decorator with tools and LLM."""
        from rh_agents import LLM, ToolSet
        
        # Create a simple tool
        @tool_decorator(name="helper_tool")
        async def helper(input: SimpleInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="help", tool_name="helper_tool")
        
        # Create a mock LLM
        mock_llm = LLM(
            name="mock_llm",
            description="Mock LLM",
            input_model=SimpleInput,
            handler=lambda x, c, s: None,
            cacheable=True
        )
        
        @agent_decorator(name="agent_with_tools", tools=[helper], llm=mock_llm)
        async def my_agent(input: SimpleInput, context: str, state: ExecutionState) -> SimpleOutput:
            return SimpleOutput(result="done")
        
        assert len(my_agent.tools) == 1
        assert my_agent.llm == mock_llm
    
    def test_agent_decorator_defaults(self):
        """Test agent decorator uses function name and docstring."""
        @agent_decorator()
        async def process_data(input: SimpleInput, context: str, state: ExecutionState) -> SimpleOutput:
            """Processes incoming data."""
            return SimpleOutput(result="processed")
        
        assert process_data.name == "process_data"
        assert "Processes incoming data" in process_data.description
    
    def test_agent_decorator_validates_signature(self):
        """Test that decorator validates handler signature."""
        with pytest.raises(ValueError, match="must have at least one parameter"):
            @agent_decorator(name="bad_agent")
            async def bad_agent() -> SimpleOutput:
                return SimpleOutput(result="bad")
