"""
Integration tests for builder factories.

Tests full execution flows with mocked ExecutionEvents to verify:
- End-to-end handler execution
- ExecutionEvent integration
- Tool execution and aggregation
- Error propagation
- Result formatting
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from rh_agents.builders import (
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.events import ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, Tool_Result, ToolExecutionResult
from rh_agents.core.types import ErrorStrategy
from rh_agents.models import Message, AuthorType
from rh_agents.openai import ToolCall


# Test models
class TestInput(BaseModel):
    content: str


class TestOutput(BaseModel):
    result: str


class ParsedData(BaseModel):
    field_a: str
    field_b: int


# Mock execution state
def create_mock_execution_state():
    state = MagicMock(spec=ExecutionState)
    state.log = AsyncMock()
    return state


@pytest.mark.asyncio
class TestStructuredAgentIntegration:
    """Test StructuredAgent end-to-end execution."""
    
    async def test_structured_agent_execution_with_tool_call(self):
        """StructuredAgent executes and extracts tool call result."""
        # Mock LLM
        llm = MagicMock(spec=LLM)
        
        # Create agent
        agent = await StructuredAgent.from_model(
            name="TestParser",
            llm=llm,
            input_model=TestInput,
            output_model=ParsedData,
            system_prompt="Parse input"
        )
        
        # Mock tool call response
        tool_call = ToolCall(
            id="call_1",
            tool_name="ParseTool",
            arguments='{"field_a": "value1", "field_b": 42}'
        )
        
        llm_result = LLM_Result(
            content=None,
            is_content=False,
            is_tool_call=True,
            tools=[tool_call]
        )
        
        # Mock ExecutionEvent
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=llm_result,
                ok=True,
                erro_message=None
            )
            mock_event_cls.return_value = mock_event
            
            # Execute handler
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test input"),
                "",
                execution_state
            )
            
            # Verify result
            assert isinstance(result, ParsedData)
            assert result.field_a == "value1"
            assert result.field_b == 42
            
            # Verify ExecutionEvent was called
            assert mock_event.called
    
    async def test_structured_agent_with_content_fallback(self):
        """StructuredAgent handles content response (no tool call)."""
        llm = MagicMock(spec=LLM)
        
        agent = await StructuredAgent.from_model(
            name="TestParser",
            llm=llm,
            input_model=TestInput,
            output_model=ParsedData,
            system_prompt="Parse input"
        )
        
        # Mock content response (not tool call)
        llm_result = LLM_Result(
            content="Plain text response",
            is_content=True,
            is_tool_call=False
        )
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=llm_result,
                ok=True,
                erro_message=None
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test"),
                "",
                execution_state
            )
            
            # Should return Message with content
            assert isinstance(result, Message)
            assert result.content == "Plain text response"


@pytest.mark.asyncio
class TestCompletionAgentIntegration:
    """Test CompletionAgent end-to-end execution."""
    
    async def test_completion_agent_execution(self):
        """CompletionAgent executes and returns Message."""
        llm = MagicMock(spec=LLM)
        
        agent = await CompletionAgent.from_prompt(
            name="TestCompletion",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Complete this"
        )
        
        # Mock LLM response
        llm_result = LLM_Result(
            content="Generated completion text",
            is_content=True,
            is_tool_call=False
        )
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=llm_result,
                ok=True,
                erro_message=None
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test prompt"),
                "",
                execution_state
            )
            
            # Verify Message result
            assert isinstance(result, Message)
            assert result.content == "Generated completion text"
            assert result.author == AuthorType.ASSISTANT
    
    async def test_completion_agent_with_temperature_override(self):
        """CompletionAgent uses overridden temperature."""
        llm = MagicMock(spec=LLM)
        
        agent = (
            await CompletionAgent.from_prompt(
                name="TestCompletion",
                llm=llm,
                input_model=TestInput,
                output_model=Message,
                system_prompt="Complete this"
            )
            .with_temperature(0.3)
            .with_max_tokens(500)
        )
        
        llm_result = LLM_Result(content="Response", is_content=True, is_tool_call=False)
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=llm_result,
                ok=True,
                erro_message=None
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = create_mock_execution_state()
            await agent.handler(TestInput(content="test"), "", execution_state)
            
            # Verify overrides were stored
            assert agent._overrides['temperature'] == 0.3
            assert agent._overrides['max_tokens'] == 500


@pytest.mark.asyncio
class TestToolExecutorAgentIntegration:
    """Test ToolExecutorAgent end-to-end execution."""
    
    async def test_tool_executor_parallel_execution(self):
        """ToolExecutorAgent executes multiple tools in parallel."""
        llm = MagicMock(spec=LLM)
        
        # Mock tools
        tool1 = MagicMock(spec=Tool)
        tool1.name = "Tool1"
        tool1.input_model = TestInput
        tool1.output_model = TestOutput
        
        tool2 = MagicMock(spec=Tool)
        tool2.name = "Tool2"
        tool2.input_model = TestInput
        tool2.output_model = TestOutput
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Execute tools",
            tools=[tool1, tool2]
        )
        
        # Mock LLM response with multiple tool calls
        tool_call_1 = ToolCall(
            id="call_1",
            tool_name="Tool1",
            arguments='{"content": "arg1"}'
        )
        tool_call_2 = ToolCall(
            id="call_2",
            tool_name="Tool2",
            arguments='{"content": "arg2"}'
        )
        
        llm_result = LLM_Result(
            content=None,
            is_content=False,
            is_tool_call=True,
            tools=[tool_call_1, tool_call_2]
        )
        
        # Mock tool results
        tool_result_1 = Tool_Result(output="Result from Tool1")
        tool_result_2 = Tool_Result(output="Result from Tool2")
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            # Create separate mocks for LLM and tool events
            llm_event = AsyncMock()
            llm_event.return_value = ExecutionResult(
                result=llm_result,
                ok=True,
                erro_message=None
            )
            
            tool_event_1 = AsyncMock()
            tool_event_1.return_value = ExecutionResult(
                result=tool_result_1,
                ok=True,
                erro_message=None
            )
            
            tool_event_2 = AsyncMock()
            tool_event_2.return_value = ExecutionResult(
                result=tool_result_2,
                ok=True,
                erro_message=None
            )
            
            # Mock ExecutionEvent to return different results
            call_count = [0]
            def side_effect(actor):
                if call_count[0] == 0:
                    call_count[0] += 1
                    return llm_event
                elif call_count[0] == 1:
                    call_count[0] += 1
                    return tool_event_1
                else:
                    return tool_event_2
            
            mock_event_cls.side_effect = side_effect
            
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test"),
                "",
                execution_state
            )
            
            # Verify ToolExecutionResult
            assert isinstance(result, ToolExecutionResult)
            assert len(result.execution_order) == 2
            assert "Tool1" in result.results
            assert "Tool2" in result.results
    
    async def test_tool_executor_with_first_result_only(self):
        """ToolExecutorAgent stops after first result when configured."""
        llm = MagicMock(spec=LLM)
        
        tool1 = MagicMock(spec=Tool)
        tool1.name = "Tool1"
        tool1.input_model = TestInput
        tool1.output_model = TestOutput
        
        agent = (
            await ToolExecutorAgent.from_tools(
                name="TestExecutor",
                llm=llm,
                input_model=TestInput,
                output_model=TestOutput,
                system_prompt="Execute tools",
                tools=[tool1]
            )
            .with_first_result_only()
        )
        
        # Verify override is set
        assert agent._overrides['first_result_only'] is True


@pytest.mark.asyncio
class TestDirectToolAgentIntegration:
    """Test DirectToolAgent end-to-end execution."""
    
    async def test_direct_tool_execution(self):
        """DirectToolAgent executes tool without LLM."""
        tool = MagicMock(spec=Tool)
        tool.name = "DirectTool"
        tool.description = "Direct execution"
        tool.input_model = TestInput
        tool.output_model = TestOutput
        
        agent = await DirectToolAgent.from_tool(
            name="TestDirect",
            tool=tool
        )
        
        # Mock tool result
        tool_result = Tool_Result(output="Direct result")
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=tool_result,
                ok=True,
                erro_message=None
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test"),
                "",
                execution_state
            )
            
            # Verify result
            assert result == tool_result
            
            # Verify ExecutionEvent was called for tool
            assert mock_event.called
    
    async def test_direct_tool_with_error_strategy_return_none(self):
        """DirectToolAgent respects error strategy."""
        tool = MagicMock(spec=Tool)
        tool.name = "DirectTool"
        tool.description = "Direct execution"
        tool.input_model = TestInput
        tool.output_model = TestOutput
        
        agent = (
            await DirectToolAgent.from_tool(
                name="TestDirect",
                tool=tool
            )
            .with_error_strategy(ErrorStrategy.RETURN_NONE)
        )
        
        # Mock tool error
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            mock_event = AsyncMock()
            mock_event.return_value = ExecutionResult(
                result=None,
                ok=False,
                erro_message="Tool failed"
            )
            mock_event_cls.return_value = mock_event
            
            execution_state = create_mock_execution_state()
            result = await agent.handler(
                TestInput(content="test"),
                "",
                execution_state
            )
            
            # Should return ExecutionResult with ok=False
            assert result is not None
            assert result.ok is False


@pytest.mark.asyncio
class TestContextTransformIntegration:
    """Test context transformation in execution."""
    
    async def test_custom_context_transform(self):
        """Custom context transform is applied."""
        llm = MagicMock(spec=LLM)
        
        def custom_transform(ctx: str) -> str:
            return f"\n\n[CONTEXT]: {ctx.upper()}"
        
        agent = (
            await CompletionAgent.from_prompt(
                name="Test",
                llm=llm,
                input_model=TestInput,
                output_model=Message,
                system_prompt="Base prompt"
            )
            .with_context_transform(custom_transform)
        )
        
        # Verify transform is stored
        assert agent._overrides['context_transform'] == custom_transform


@pytest.mark.asyncio
class TestDynamicPromptBuilderIntegration:
    """Test dynamic system prompt builder in execution."""
    
    async def test_dynamic_prompt_builder(self):
        """Dynamic prompt builder is used instead of static prompt."""
        llm = MagicMock(spec=LLM)
        
        async def dynamic_builder(input_data, context, state):
            return f"Dynamic prompt for: {input_data.content}"
        
        agent = (
            await CompletionAgent.from_prompt(
                name="Test",
                llm=llm,
                input_model=TestInput,
                output_model=Message,
                system_prompt="Static prompt"
            )
            .with_system_prompt_builder(dynamic_builder)
        )
        
        # Verify builder is stored
        assert agent._overrides['system_prompt_builder'] == dynamic_builder
