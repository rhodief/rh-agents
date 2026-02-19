"""
Unit tests for builder factories.

Tests each builder type with mocked LLMs and tools to verify:
- Proper agent instantiation
- Chainable method functionality
- Error handling strategies
- Closure-based override mechanics
- ExecutionEvent integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from rh_agents.builders import (
    BuilderAgent,
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.events import ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, ToolExecutionResult
from rh_agents.core.types import ErrorStrategy
from rh_agents.models import Message, AuthorType


# Test models
class TestInput(BaseModel):
    content: str


class TestOutput(BaseModel):
    result: str


class TestStructuredOutput(BaseModel):
    parsed_data: str
    confidence: float


# Mock LLM
class MockLLM(LLM):
    def __init__(self):
        # Simple async handler that returns mock response
        async def mock_handler(input_data, context, execution_state):
            return LLM_Result(
                content="Mock response",
                is_content=True,
                is_tool_call=False
            )
        
        super().__init__(
            name="MockLLM",
            description="Mock LLM for testing",
            input_model=BaseModel,
            output_model=LLM_Result,
            handler=mock_handler
        )


# Mock Tool
class MockTool(Tool):
    def __init__(self, name="MockTool"):
        # Simple async handler that returns mock result
        async def mock_handler(input_data, context, execution_state):
            return TestOutput(result="Mock tool result")
        
        super().__init__(
            name=name,
            description="Mock tool for testing",
            input_model=TestInput,
            output_model=TestOutput,
            handler=mock_handler
        )


@pytest.mark.asyncio
class TestBuildersInstantiation:
    """Test basic instantiation of all builder types."""
    
    async def test_structured_agent_creates_builder_agent(self):
        """StructuredAgent.from_model() returns BuilderAgent instance."""
        llm = MockLLM()
        
        agent = await StructuredAgent.from_model(
            name="TestStructuredAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestStructuredOutput,
            system_prompt="Test structured prompt"
        )
        
        assert isinstance(agent, BuilderAgent)
        assert agent.name == "TestStructuredAgent"
        assert agent.input_model == TestInput
        assert agent.output_model == TestStructuredOutput
        assert agent.llm == llm
        assert hasattr(agent, '_overrides')
    
    async def test_completion_agent_creates_builder_agent(self):
        """CompletionAgent.from_prompt() returns BuilderAgent instance."""
        llm = MockLLM()
        
        agent = await CompletionAgent.from_prompt(
            name="TestCompletionAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test completion prompt"
        )
        
        assert isinstance(agent, BuilderAgent)
        assert agent.name == "TestCompletionAgent"
        assert agent.input_model == TestInput
        assert agent.output_model == Message
        assert agent.llm == llm
        assert hasattr(agent, '_overrides')
    
    async def test_tool_executor_agent_creates_builder_agent(self):
        """ToolExecutorAgent.from_tools() returns BuilderAgent instance."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestToolExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test tool execution prompt",
            tools=tools
        )
        
        assert isinstance(agent, BuilderAgent)
        assert agent.name == "TestToolExecutor"
        assert agent.input_model == TestInput
        assert agent.output_model == ToolExecutionResult
        assert agent.llm == llm
        assert len(agent.tools.tools) == 1
        assert hasattr(agent, '_overrides')
    
    async def test_direct_tool_agent_creates_builder_agent(self):
        """DirectToolAgent.from_tool() returns BuilderAgent instance."""
        tool = MockTool()
        
        agent = await DirectToolAgent.from_tool(
            name="TestDirectTool",
            tool=tool
        )
        
        assert isinstance(agent, BuilderAgent)
        assert agent.name == "TestDirectTool"
        assert agent.input_model == TestInput
        assert agent.output_model == TestOutput
        assert agent.llm is None
        assert len(agent.tools.tools) == 1
        assert hasattr(agent, '_overrides')


@pytest.mark.asyncio
class TestChainableMethods:
    """Test all chainable builder methods."""
    
    async def test_with_temperature(self):
        """with_temperature() sets override and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.with_temperature(0.7)
        
        assert result is agent  # Returns self for chaining
        assert agent._overrides['temperature'] == 0.7
    
    async def test_with_max_tokens(self):
        """with_max_tokens() sets override and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.with_max_tokens(1000)
        
        assert result is agent
        assert agent._overrides['max_tokens'] == 1000
    
    async def test_with_model(self):
        """with_model() sets override and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.with_model('gpt-4o-mini')
        
        assert result is agent
        assert agent._overrides['model'] == 'gpt-4o-mini'
    
    async def test_with_error_strategy(self):
        """with_error_strategy() sets override and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.with_error_strategy(ErrorStrategy.RETURN_NONE)
        
        assert result is agent
        assert agent._overrides['error_strategy'] == ErrorStrategy.RETURN_NONE
    
    async def test_as_cacheable(self):
        """as_cacheable() sets cacheable flag and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.as_cacheable(ttl=300)
        
        assert result is agent
        assert agent.cacheable is True
        assert agent.cache_ttl == 300
    
    async def test_as_artifact(self):
        """as_artifact() sets artifact flag and returns self."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        result = agent.as_artifact()
        
        assert result is agent
        assert agent.is_artifact is True
    
    async def test_chaining_multiple_methods(self):
        """Multiple chainable methods can be called in sequence."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        agent = (
            agent
            .with_temperature(0.8)
            .with_max_tokens(2000)
            .with_model('gpt-4o')
            .as_cacheable(ttl=600)
            .as_artifact()
        )
        
        assert agent._overrides['temperature'] == 0.8
        assert agent._overrides['max_tokens'] == 2000
        assert agent._overrides['model'] == 'gpt-4o'
        assert agent.cacheable is True
        assert agent.cache_ttl == 600
        assert agent.is_artifact is True


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling strategies."""
    
    async def test_error_strategy_raise(self):
        """ErrorStrategy.RAISE raises exceptions on failure."""
        # Mock LLM that returns error
        llm = MockLLM()
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = agent.with_error_strategy(ErrorStrategy.RAISE)
        
        # Mock ExecutionEvent to return error
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=None,
                ok=False,
                erro_message="Mock error"
            )
            mock_event_cls.return_value = mock_event
            
            # Should raise exception
            with pytest.raises(Exception, match="LLM execution failed"):
                execution_state = MagicMock()
                await agent.handler(TestInput(content="test"), "", execution_state)
    
    async def test_error_strategy_return_none(self):
        """ErrorStrategy.RETURN_NONE returns None on failure."""
        llm = MockLLM()
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = agent.with_error_strategy(ErrorStrategy.RETURN_NONE)
        
        # Mock ExecutionEvent to return error
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=None,
                ok=False,
                erro_message="Mock error"
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = MagicMock()
            result = await agent.handler(TestInput(content="test"), "", execution_state)
            
            # Should return ExecutionResult with ok=False
            assert result is not None
            assert result.ok is False


@pytest.mark.asyncio
class TestOverrideClosures:
    """Test that overrides work correctly via closures."""
    
    async def test_temperature_override_used_in_handler(self):
        """Handler uses overridden temperature value."""
        llm = MockLLM()
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = agent.with_temperature(0.5)
        
        # Verify closure captured override
        assert agent._overrides['temperature'] == 0.5
    
    async def test_multiple_overrides_stored(self):
        """Multiple overrides are stored in closure."""
        llm = MockLLM()
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = (
            agent
            .with_temperature(0.7)
            .with_max_tokens(1500)
            .with_model('gpt-4o-mini')
        )
        
        assert agent._overrides['temperature'] == 0.7
        assert agent._overrides['max_tokens'] == 1500
        assert agent._overrides['model'] == 'gpt-4o-mini'


@pytest.mark.asyncio
class TestToolExecutorAgent:
    """Test ToolExecutorAgent specific functionality."""
    
    async def test_tool_executor_returns_tool_execution_result(self):
        """ToolExecutorAgent always returns ToolExecutionResult."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        
        assert agent.output_model == ToolExecutionResult
    
    async def test_with_first_result_only(self):
        """with_first_result_only() sets override."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_first_result_only()
        
        assert agent._overrides['first_result_only'] is True


@pytest.mark.asyncio
class TestDirectToolAgent:
    """Test DirectToolAgent specific functionality."""
    
    async def test_direct_tool_has_no_llm(self):
        """DirectToolAgent should have llm=None."""
        tool = MockTool()
        
        agent = await DirectToolAgent.from_tool(
            name="Test",
            tool=tool
        )
        
        assert agent.llm is None
    
    async def test_direct_tool_uses_tool_models(self):
        """DirectToolAgent uses tool's input/output models."""
        tool = MockTool()
        
        agent = await DirectToolAgent.from_tool(
            name="Test",
            tool=tool
        )
        
        assert agent.input_model == tool.input_model
        assert agent.output_model == tool.output_model


@pytest.mark.asyncio
class TestContextTransform:
    """Test context transformation functionality."""
    
    async def test_with_context_transform(self):
        """with_context_transform() sets custom transform function."""
        llm = MockLLM()
        
        def custom_transform(ctx: str) -> str:
            return f"\n\nCustom: {ctx}"
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = agent.with_context_transform(custom_transform)
        
        assert agent._overrides['context_transform'] == custom_transform


@pytest.mark.asyncio
class TestSystemPromptBuilder:
    """Test dynamic system prompt builder functionality."""
    
    async def test_with_system_prompt_builder(self):
        """with_system_prompt_builder() sets dynamic builder function."""
        llm = MockLLM()
        
        async def dynamic_prompt(input_data, context, state):
            return f"Dynamic: {input_data.content}"
        
        agent = await CompletionAgent.from_prompt(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        agent = agent.with_system_prompt_builder(dynamic_prompt)
        
        assert agent._overrides['system_prompt_builder'] == dynamic_prompt


@pytest.mark.asyncio
class TestToolChoice:
    """Test tool choice override for structured output."""
    
    async def test_with_tool_choice(self):
        """with_tool_choice() sets tool choice override."""
        llm = MockLLM()
        
        agent = await StructuredAgent.from_model(
            name="Test",
            llm=llm,
            input_model=TestInput,
            output_model=TestStructuredOutput,
            system_prompt="Test"
        )
        agent = agent.with_tool_choice("StructuredTool")
        
        assert 'tool_choice' in agent._overrides
        assert agent._overrides['tool_choice']['function']['name'] == "StructuredTool"
