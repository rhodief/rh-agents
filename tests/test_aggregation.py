"""
Unit tests for Phase 2: Aggregation Strategies.

Tests the new aggregation functionality added to ToolExecutorAgent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from rh_agents.builders import ToolExecutorAgent
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.events import ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, LLM_Tool_Call, Tool_Result, ToolExecutionResult
from rh_agents.core.types import AggregationStrategy
from rh_agents.models import Message


# Test models
class TestInput(BaseModel):
    content: str


class TestOutput(BaseModel):
    result: str


# Mock LLM
class MockLLM(LLM):
    def __init__(self):
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
    def __init__(self, name="MockTool", return_value="Mock result"):
        # Store return_value before calling super().__init__
        _return_value = return_value
        
        async def mock_handler(input_data, context, execution_state):
            return TestOutput(result=_return_value)
        
        super().__init__(
            name=name,
            description="Mock tool for testing",
            input_model=TestInput,
            output_model=TestOutput,
            handler=mock_handler
        )


@pytest.mark.asyncio
class TestAggregationStrategies:
    """Test different aggregation strategies for ToolExecutorAgent."""
    
    async def test_dict_aggregation_default(self):
        """DICT aggregation (default) returns ToolExecutionResult."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "result1"), MockTool("Tool2", "result2")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        
        # Verify output model is ToolExecutionResult (default)
        assert agent.output_model == ToolExecutionResult
    
    async def test_list_aggregation(self):
        """LIST aggregation returns list of results in execution order."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "result1"), MockTool("Tool2", "result2")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_aggregation(AggregationStrategy.LIST)
        
        # Verify aggregation is set
        assert agent._overrides['aggregation_strategy'] == AggregationStrategy.LIST
    
    async def test_concatenate_aggregation(self):
        """CONCATENATE aggregation joins results with separator."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "result1"), MockTool("Tool2", "result2")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_aggregation(AggregationStrategy.CONCATENATE, separator=" | ")
        
        # Verify aggregation and separator are set
        assert agent._overrides['aggregation_strategy'] == AggregationStrategy.CONCATENATE
        assert agent._overrides['aggregation_separator'] == " | "
    
    async def test_first_aggregation(self):
        """FIRST aggregation returns only first successful result."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "result1"), MockTool("Tool2", "result2")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_aggregation(AggregationStrategy.FIRST)
        
        # Verify aggregation is set
        assert agent._overrides['aggregation_strategy'] == AggregationStrategy.FIRST


@pytest.mark.asyncio
class TestToolExecutionResultMethods:
    """Test ToolExecutionResult aggregation methods."""
    
    def test_to_list(self):
        """to_list() returns results as list in execution order."""
        result = ToolExecutionResult(
            results={"Tool1": "result1", "Tool2": "result2", "Tool3": "result3"},
            execution_order=["Tool1", "Tool2", "Tool3"],
            errors={}
        )
        
        as_list = result.to_list()
        assert as_list == ["result1", "result2", "result3"]
    
    def test_to_list_with_errors(self):
        """to_list() skips tools that had errors."""
        result = ToolExecutionResult(
            results={"Tool1": "result1", "Tool3": "result3"},
            execution_order=["Tool1", "Tool2", "Tool3"],  # Tool2 failed
            errors={"Tool2": "Some error"}
        )
        
        as_list = result.to_list()
        assert as_list == ["result1", "result3"]
    
    def test_to_concatenated_default_separator(self):
        """to_concatenated() joins results with default separator."""
        result = ToolExecutionResult(
            results={"Tool1": "result1", "Tool2": "result2"},
            execution_order=["Tool1", "Tool2"],
            errors={}
        )
        
        concatenated = result.to_concatenated()
        assert concatenated == "result1\n\nresult2"
    
    def test_to_concatenated_custom_separator(self):
        """to_concatenated() respects custom separator."""
        result = ToolExecutionResult(
            results={"Tool1": "result1", "Tool2": "result2", "Tool3": "result3"},
            execution_order=["Tool1", "Tool2", "Tool3"],
            errors={}
        )
        
        concatenated = result.to_concatenated(separator=" | ")
        assert concatenated == "result1 | result2 | result3"
    
    def test_to_concatenated_non_string_results(self):
        """to_concatenated() converts non-strings to strings."""
        result = ToolExecutionResult(
            results={"Tool1": 42, "Tool2": {"key": "value"}, "Tool3": [1, 2, 3]},
            execution_order=["Tool1", "Tool2", "Tool3"],
            errors={}
        )
        
        concatenated = result.to_concatenated(separator="\n")
        assert "42" in concatenated
        assert "{'key': 'value'}" in concatenated or "key" in concatenated
        assert "[1, 2, 3]" in concatenated or "1" in concatenated
    
    def test_to_dict(self):
        """to_dict() returns results dictionary."""
        result = ToolExecutionResult(
            results={"Tool1": "result1", "Tool2": "result2"},
            execution_order=["Tool1", "Tool2"],
            errors={}
        )
        
        as_dict = result.to_dict()
        assert as_dict == {"Tool1": "result1", "Tool2": "result2"}
        assert as_dict is result.results  # Same reference
    
    def test_first_method(self):
        """first() returns first result in execution order."""
        result = ToolExecutionResult(
            results={"Tool2": "result2", "Tool1": "result1", "Tool3": "result3"},
            execution_order=["Tool1", "Tool2", "Tool3"],
            errors={}
        )
        
        first = result.first()
        assert first == "result1"  # First by execution_order, not dict order
    
    def test_first_method_empty(self):
        """first() returns None for empty results."""
        result = ToolExecutionResult(
            results={},
            execution_order=[],
            errors={}
        )
        
        first = result.first()
        assert first is None


@pytest.mark.asyncio
class TestAggregationIntegration:
    """Integration tests for aggregation with mocked execution."""
    
    async def test_list_aggregation_integration(self):
        """Test LIST aggregation with mocked tool execution."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "A"), MockTool("Tool2", "B")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_aggregation(AggregationStrategy.LIST)
        
        # Mock LLM to return tool calls
        tool_call_1 = LLM_Tool_Call(tool_name="Tool1", arguments='{"content": "test"}')
        tool_call_2 = LLM_Tool_Call(tool_name="Tool2", arguments='{"content": "test"}')
        
        llm_result = LLM_Result(
            content="",
            tools=[tool_call_1, tool_call_2]
        )
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            # LLM event returns tool calls
            llm_event = AsyncMock()
            llm_event.return_value = ExecutionResult(result=llm_result, ok=True, erro_message=None)
            
            # Tool events return results
            tool_event_1 = AsyncMock()
            tool_event_1.return_value = ExecutionResult(
                result=Tool_Result(output="A", tool_name="Tool1"),
                ok=True,
                erro_message=None
            )
            
            tool_event_2 = AsyncMock()
            tool_event_2.return_value = ExecutionResult(
                result=Tool_Result(output="B", tool_name="Tool2"),
                ok=True,
                erro_message=None
            )
            
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
            
            execution_state = MagicMock(spec=ExecutionState)
            result = await agent.handler(TestInput(content="test"), "", execution_state)
            
            # Result should be a list
            assert isinstance(result, list)
            assert len(result) == 2
            assert "A" in str(result)
            assert "B" in str(result)
    
    async def test_concatenate_aggregation_integration(self):
        """Test CONCATENATE aggregation with mocked tool execution."""
        llm = MockLLM()
        tools = [MockTool("Tool1", "Hello"), MockTool("Tool2", "World")]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestExecutor",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        agent = agent.with_aggregation(AggregationStrategy.CONCATENATE, separator=" ")
        
        # Mock setup similar to above
        tool_call_1 = LLM_Tool_Call(tool_name="Tool1", arguments='{"content": "test"}')
        tool_call_2 = LLM_Tool_Call(tool_name="Tool2", arguments='{"content": "test"}')
        
        llm_result = LLM_Result(
            content="",
            tools=[tool_call_1, tool_call_2]
        )
        
        with patch('rh_agents.builders.ExecutionEvent') as mock_event_cls:
            llm_event = AsyncMock()
            llm_event.return_value = ExecutionResult(result=llm_result, ok=True, erro_message=None)
            
            tool_event_1 = AsyncMock()
            tool_event_1.return_value = ExecutionResult(
                result=Tool_Result(output="Hello", tool_name="Tool1"),
                ok=True,
                erro_message=None
            )
            
            tool_event_2 = AsyncMock()
            tool_event_2.return_value = ExecutionResult(
                result=Tool_Result(output="World", tool_name="Tool2"),
                ok=True,
                erro_message=None
            )
            
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
            
            execution_state = MagicMock(spec=ExecutionState)
            result = await agent.handler(TestInput(content="test"), "", execution_state)
            
            # Result should be concatenated string
            assert isinstance(result, str)
            # Should contain both results joined with space
            assert "Hello" in result
            assert "World" in result
