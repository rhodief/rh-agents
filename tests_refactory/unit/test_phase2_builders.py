"""
Phase 2 Unit Tests: Builder Pattern API

Tests the fluent builder pattern for constructing complex actors.
"""
import pytest
from pydantic import BaseModel
from rh_agents import AgentBuilder, ToolBuilder, ExecutionState, LLM, Tool
from rh_agents.core.result_types import Tool_Result


class BuilderInput(BaseModel):
    value: int


class BuilderOutput(BaseModel):
    result: int


class TestToolBuilder:
    """Test ToolBuilder fluent API."""
    
    def test_build_basic_tool(self):
        """Test building a basic tool with required fields."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output=input.value * 2, tool_name="doubler")
        
        tool = (
            ToolBuilder()
            .name("doubler")
            .description("Doubles a number")
            .input_model(BuilderInput)
            .handler(handler)
            .build()
        )
        
        assert isinstance(tool, Tool)
        assert tool.name == "doubler"
        assert tool.description == "Doubles a number"
        assert tool.input_model == BuilderInput
        assert tool.handler == handler
        assert tool.cacheable is False
        assert tool.version == "1.0.0"
    
    def test_build_cacheable_tool(self):
        """Test building a cacheable tool with TTL."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="cached", tool_name="test")
        
        tool = (
            ToolBuilder()
            .name("cache_tool")
            .description("Cacheable tool")
            .input_model(BuilderInput)
            .handler(handler)
            .cacheable(True, ttl=3600)
            .build()
        )
        
        assert tool.cacheable is True
        assert tool.cache_ttl == 3600
    
    def test_build_with_version(self):
        """Test building tool with custom version."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool = (
            ToolBuilder()
            .name("versioned_tool")
            .description("Tool with version")
            .input_model(BuilderInput)
            .handler(handler)
            .version("2.1.0")
            .build()
        )
        
        assert tool.version == "2.1.0"
    
    def test_build_with_output_model(self):
        """Test building tool with output model."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> BuilderOutput:
            return BuilderOutput(result=input.value)
        
        tool = (
            ToolBuilder()
            .name("typed_tool")
            .description("Tool with output model")
            .input_model(BuilderInput)
            .output_model(BuilderOutput)
            .handler(handler)
            .build()
        )
        
        assert tool.output_model == BuilderOutput
    
    def test_build_fails_without_name(self):
        """Test that build fails if name is missing."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        builder = (
            ToolBuilder()
            .description("Description")
            .input_model(BuilderInput)
            .handler(handler)
        )
        
        with pytest.raises(ValueError, match="name is required"):
            builder.build()
    
    def test_build_fails_without_handler(self):
        """Test that build fails if handler is missing."""
        builder = (
            ToolBuilder()
            .name("tool")
            .description("Description")
            .input_model(BuilderInput)
        )
        
        with pytest.raises(ValueError, match="handler is required"):
            builder.build()
    
    def test_fluent_api_returns_builder(self):
        """Test that all builder methods return self for chaining."""
        builder = ToolBuilder()
        
        assert builder.name("test") is builder
        assert builder.description("desc") is builder
        assert builder.input_model(BuilderInput) is builder
        assert builder.version("1.0.0") is builder
        assert builder.cacheable(True) is builder


class TestAgentBuilder:
    """Test AgentBuilder fluent API."""
    
    def test_build_basic_agent(self):
        """Test building a basic agent with required fields."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> BuilderOutput:
            return BuilderOutput(result=input.value)
        
        agent = (
            AgentBuilder()
            .name("processor")
            .description("Processes data")
            .input_model(BuilderInput)
            .handler(handler)
            .build()
        )
        
        assert agent.name == "processor"
        assert agent.description == "Processes data"
        assert agent.input_model == BuilderInput
        assert agent.handler == handler
        assert len(agent.tools) == 0
        assert agent.llm is None
    
    def test_build_with_llm(self):
        """Test building agent with LLM."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=42)
        
        mock_llm = LLM(
            name="mock_llm",
            description="Mock",
            input_model=BuilderInput,
            handler=lambda x, c, s: None,
            cacheable=True
        )
        
        agent = (
            AgentBuilder()
            .name("agent_with_llm")
            .description("Has LLM")
            .input_model(BuilderInput)
            .handler(handler)
            .with_llm(mock_llm)
            .build()
        )
        
        assert agent.llm == mock_llm
    
    def test_build_with_tools(self):
        """Test building agent with multiple tools."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=0)
        
        async def tool_handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool1 = Tool(name="tool1", description="Tool 1", input_model=BuilderInput, handler=tool_handler)
        tool2 = Tool(name="tool2", description="Tool 2", input_model=BuilderInput, handler=tool_handler)
        
        agent = (
            AgentBuilder()
            .name("agent_with_tools")
            .description("Has tools")
            .input_model(BuilderInput)
            .handler(handler)
            .with_tools([tool1, tool2])
            .build()
        )
        
        assert len(agent.tools) == 2
        assert agent.tools["tool1"] == tool1
        assert agent.tools["tool2"] == tool2
    
    def test_add_single_tool(self):
        """Test adding tools one at a time."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=0)
        
        async def tool_handler(input: BuilderInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool1 = Tool(name="tool1", description="Tool 1", input_model=BuilderInput, handler=tool_handler)
        tool2 = Tool(name="tool2", description="Tool 2", input_model=BuilderInput, handler=tool_handler)
        
        agent = (
            AgentBuilder()
            .name("agent")
            .description("Agent")
            .input_model(BuilderInput)
            .handler(handler)
            .add_tool(tool1)
            .add_tool(tool2)
            .build()
        )
        
        assert len(agent.tools) == 2
    
    def test_build_with_caching(self):
        """Test building cacheable agent."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=0)
        
        agent = (
            AgentBuilder()
            .name("cached_agent")
            .description("Cacheable")
            .input_model(BuilderInput)
            .handler(handler)
            .cacheable(True)
            .build()
        )
        
        assert agent.cacheable is True
    
    def test_build_with_output_model(self):
        """Test building agent with output model."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState) -> BuilderOutput:
            return BuilderOutput(result=0)
        
        agent = (
            AgentBuilder()
            .name("typed_agent")
            .description("Has output type")
            .input_model(BuilderInput)
            .output_model(BuilderOutput)
            .handler(handler)
            .build()
        )
        
        assert agent.output_model == BuilderOutput
    
    def test_build_with_conditions(self):
        """Test building agent with pre/post conditions."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=0)
        
        def precondition(state):
            return True
        
        def postcondition(result):
            return True
        
        agent = (
            AgentBuilder()
            .name("conditional_agent")
            .description("Has conditions")
            .input_model(BuilderInput)
            .handler(handler)
            .add_precondition(precondition)
            .add_postcondition(postcondition)
            .build()
        )
        
        assert len(agent.preconditions) == 1
        assert len(agent.postconditions) == 1
    
    def test_build_fails_without_required_fields(self):
        """Test that build fails if required fields are missing."""
        async def handler(input: BuilderInput, context: str, state: ExecutionState):
            return BuilderOutput(result=0)
        
        # Missing name
        with pytest.raises(ValueError, match="name is required"):
            AgentBuilder().description("d").input_model(BuilderInput).handler(handler).build()
        
        # Missing description
        with pytest.raises(ValueError, match="description is required"):
            AgentBuilder().name("n").input_model(BuilderInput).handler(handler).build()
        
        # Missing input_model
        with pytest.raises(ValueError, match="input_model is required"):
            AgentBuilder().name("n").description("d").handler(handler).build()
        
        # Missing handler
        with pytest.raises(ValueError, match="handler is required"):
            AgentBuilder().name("n").description("d").input_model(BuilderInput).build()
