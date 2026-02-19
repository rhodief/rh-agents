"""
Tests for Phase 3: Configuration & Customization.

Tests parameter validation and configuration options for all builder types.
"""

import pytest
from pydantic import BaseModel

from rh_agents.builders import (
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.types import ErrorStrategy, AggregationStrategy
from rh_agents.core.result_types import LLM_Result
from rh_agents.models import Message


# Mock models for testing
class TestInput(BaseModel):
    content: str


class TestOutput(BaseModel):
    result: str


# Mock LLM
class MockLLM(LLM):
    def __init__(self):
        async def mock_handler(input_data, context, execution_state):
            return LLM_Result(
                content="Mock response"
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
    def __init__(self):
        async def mock_handler(input_data, context, execution_state):
            return TestOutput(result="Mock result")
        
        super().__init__(
            name="MockTool",
            description="Mock tool for testing",
            input_model=TestInput,
            output_model=TestOutput,
            handler=mock_handler
        )


@pytest.mark.asyncio
class TestParameterValidation:
    """Test parameter validation in chainable methods."""
    
    async def test_temperature_validation_min(self):
        """Temperature must be >= 0.0."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            agent.with_temperature(-0.1)
    
    async def test_temperature_validation_max(self):
        """Temperature must be <= 2.0."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            agent.with_temperature(2.1)
    
    async def test_temperature_validation_valid(self):
        """Valid temperature values should work."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        # Should not raise
        agent.with_temperature(0.0)
        agent.with_temperature(1.0)
        agent.with_temperature(2.0)
    
    async def test_max_tokens_validation_min(self):
        """max_tokens must be >= 1."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        with pytest.raises(ValueError, match="max_tokens must be between 1 and 128000"):
            agent.with_max_tokens(0)
    
    async def test_max_tokens_validation_max(self):
        """max_tokens must be <= 128000."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        with pytest.raises(ValueError, match="max_tokens must be between 1 and 128000"):
            agent.with_max_tokens(128001)
    
    async def test_max_tokens_validation_valid(self):
        """Valid max_tokens values should work."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        # Should not raise
        agent.with_max_tokens(1)
        agent.with_max_tokens(1000)
        agent.with_max_tokens(128000)
    
    async def test_retry_max_attempts_validation_min(self):
        """max_attempts must be >= 1."""
        tool = MockTool()
        agent = await DirectToolAgent.from_tool(
            name="TestAgent",
            tool=tool
        )
        
        with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
            agent.with_retry(max_attempts=0)
    
    async def test_retry_max_attempts_validation_max(self):
        """max_attempts must be <= 10."""
        tool = MockTool()
        agent = await DirectToolAgent.from_tool(
            name="TestAgent",
            tool=tool
        )
        
        with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
            agent.with_retry(max_attempts=11)
    
    async def test_retry_initial_delay_validation(self):
        """initial_delay must be positive."""
        tool = MockTool()
        agent = await DirectToolAgent.from_tool(
            name="TestAgent",
            tool=tool
        )
        
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            agent.with_retry(max_attempts=3, initial_delay=0)
        
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            agent.with_retry(max_attempts=3, initial_delay=-1.0)
    
    async def test_retry_validation_valid(self):
        """Valid retry values should work."""
        tool = MockTool()
        agent = await DirectToolAgent.from_tool(
            name="TestAgent",
            tool=tool
        )
        
        # Should not raise
        agent.with_retry(max_attempts=1, initial_delay=0.1)
        agent.with_retry(max_attempts=5, initial_delay=1.0)
        agent.with_retry(max_attempts=10, initial_delay=5.0)
    
    async def test_cacheable_ttl_validation(self):
        """TTL must be non-negative."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            agent.as_cacheable(ttl=-1)
    
    async def test_cacheable_ttl_valid(self):
        """Valid TTL values should work."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        # Should not raise
        agent.as_cacheable(ttl=0)
        agent.as_cacheable(ttl=300)
        agent.as_cacheable(ttl=None)  # No expiration


@pytest.mark.asyncio
class TestConfigurationOptions:
    """Test all configuration options work correctly."""
    
    async def test_llm_parameter_overrides(self):
        """Test LLM parameter overrides (model, temperature, max_tokens)."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        agent = (
            agent
            .with_model("gpt-4o-mini")
            .with_temperature(0.5)
            .with_max_tokens(1000)
        )
        
        # Check overrides are stored
        assert agent._overrides['model'] == "gpt-4o-mini"
        assert agent._overrides['temperature'] == 0.5
        assert agent._overrides['max_tokens'] == 1000
    
    async def test_error_strategy_configuration(self):
        """Test error strategy configuration."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        agent = agent.with_error_strategy(ErrorStrategy.RETURN_NONE)
        
        assert agent._overrides['error_strategy'] == ErrorStrategy.RETURN_NONE
    
    async def test_context_transform(self):
        """Test context transform function."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        def custom_transform(ctx):
            return f"\n\nCustom context: {ctx}"
        
        agent = agent.with_context_transform(custom_transform)
        
        assert agent._overrides['context_transform'] == custom_transform
    
    async def test_system_prompt_builder(self):
        """Test dynamic system prompt builder."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        async def custom_prompt(input_data, context, state):
            return f"Dynamic prompt: {input_data.content}"
        
        agent = agent.with_system_prompt_builder(custom_prompt)
        
        assert agent._overrides['system_prompt_builder'] == custom_prompt
    
    async def test_artifact_configuration(self):
        """Test artifact configuration."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        agent = agent.as_artifact()
        
        assert agent.is_artifact is True
    
    async def test_cacheable_configuration(self):
        """Test caching configuration."""
        llm = MockLLM()
        agent = await CompletionAgent.from_prompt(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=Message,
            system_prompt="Test"
        )
        
        agent = agent.as_cacheable(ttl=600)
        
        assert agent.cacheable is True
        assert agent.cache_ttl == 600
    
    async def test_retry_configuration(self):
        """Test retry configuration."""
        tool = MockTool()
        agent = await DirectToolAgent.from_tool(
            name="TestAgent",
            tool=tool
        )
        
        agent = agent.with_retry(max_attempts=5, initial_delay=2.0)
        
        assert agent.retry_config is not None
        assert agent.retry_config.max_attempts == 5
        assert agent.retry_config.initial_delay == 2.0
    
    async def test_aggregation_configuration(self):
        """Test aggregation strategy configuration."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        
        agent = agent.with_aggregation(AggregationStrategy.LIST)
        
        assert agent._overrides['aggregation_strategy'] == AggregationStrategy.LIST
    
    async def test_first_result_only_configuration(self):
        """Test first_result_only configuration."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        
        agent = agent.with_first_result_only()
        
        assert agent._overrides['first_result_only'] is True


@pytest.mark.asyncio
class TestMethodChaining:
    """Test method chaining works correctly."""
    
    async def test_multiple_configurations(self):
        """Test chaining multiple configuration methods."""
        llm = MockLLM()
        agent = await StructuredAgent.from_model(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test"
        )
        
        # Chain multiple methods
        configured = (
            agent
            .with_model("gpt-4o")
            .with_temperature(0.7)
            .with_max_tokens(2000)
            .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)
            .as_artifact()
            .as_cacheable(ttl=300)
        )
        
        # Verify all configurations applied
        assert configured._overrides['model'] == "gpt-4o"
        assert configured._overrides['temperature'] == 0.7
        assert configured._overrides['max_tokens'] == 2000
        assert configured._overrides['error_strategy'] == ErrorStrategy.LOG_AND_CONTINUE
        assert configured.is_artifact is True
        assert configured.cacheable is True
        assert configured.cache_ttl == 300
    
    async def test_tool_executor_full_chain(self):
        """Test full configuration chain for ToolExecutorAgent."""
        llm = MockLLM()
        tools = [MockTool()]
        
        agent = await ToolExecutorAgent.from_tools(
            name="TestAgent",
            llm=llm,
            input_model=TestInput,
            output_model=TestOutput,
            system_prompt="Test",
            tools=tools
        )
        
        configured = (
            agent
            .with_temperature(0.8)
            .with_aggregation(AggregationStrategy.CONCATENATE, separator=" | ")
            .with_first_result_only()
            .with_retry(max_attempts=3, initial_delay=1.0)
            .as_cacheable()
        )
        
        assert configured._overrides['temperature'] == 0.8
        assert configured._overrides['aggregation_strategy'] == AggregationStrategy.CONCATENATE
        assert configured._overrides['aggregation_separator'] == " | "
        assert configured._overrides['first_result_only'] is True
        assert configured.retry_config.max_attempts == 3
        assert configured.cacheable is True

