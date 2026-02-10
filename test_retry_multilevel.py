"""
Tests for Phase 2: Multi-Level Retry Configuration

Tests the precedence order:
1. Event-level config (ExecutionEvent.retry_config)
2. Actor-level config (BaseActor.retry_config)
3. ExecutionState type-specific config (execution_state.retry_config_by_actor_type)
4. ExecutionState global default (execution_state.default_retry_config)
5. Built-in defaults (from get_default_retry_config_by_actor_type)
"""
import pytest
from pydantic import BaseModel
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import EventType
from rh_agents.core.retry import RetryConfig, BackoffStrategy


class SimpleInput(BaseModel):
    value: int = 0


class TestMultiLevelConfigPrecedence:
    """Test the precedence order of retry configurations."""
    
    @pytest.mark.asyncio
    async def test_event_level_overrides_all(self):
        """Event-level config takes highest precedence."""
        # Setup: Define configs at all levels
        state = ExecutionState()
        state.default_retry_config = RetryConfig(max_attempts=10, initial_delay=10.0)
        state.retry_config_by_actor_type = {
            EventType.TOOL_CALL: RetryConfig(max_attempts=8, initial_delay=8.0)
        }
        
        async def handler(input_data, context, state):
            return "success"
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=handler,
            retry_config=RetryConfig(max_attempts=5, initial_delay=5.0)
        )
        
        # Event-level config should win
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(max_attempts=3, initial_delay=3.0)
        )
        
        # Track which config was used
        events_received = []
        state.event_bus.subscribe(lambda e: events_received.append(e))
        
        # Test with a failing handler to trigger retry
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        actor.handler = failing_handler
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Should use event-level max_attempts=3
        assert fail_count[0] == 3
        assert result.ok is False
    
    @pytest.mark.asyncio
    async def test_actor_level_overrides_state_and_defaults(self):
        """Actor-level config used when no event-level config."""
        state = ExecutionState()
        state.default_retry_config = RetryConfig(max_attempts=10, initial_delay=10.0)
        state.retry_config_by_actor_type = {
            EventType.TOOL_CALL: RetryConfig(max_attempts=8, initial_delay=8.0)
        }
        
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=failing_handler,
            retry_config=RetryConfig(max_attempts=5, initial_delay=0.01, jitter=False)
        )
        
        # No event-level config, so actor config should be used
        event = ExecutionEvent(actor=actor)
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Should use actor-level max_attempts=5
        assert fail_count[0] == 5
        assert result.ok is False
    
    @pytest.mark.asyncio
    async def test_state_type_specific_overrides_global_and_defaults(self):
        """ExecutionState type-specific config used when no event or actor config."""
        state = ExecutionState()
        state.default_retry_config = RetryConfig(max_attempts=10, initial_delay=10.0)
        state.retry_config_by_actor_type = {
            EventType.TOOL_CALL: RetryConfig(max_attempts=7, initial_delay=0.01, jitter=False)
        }
        
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        # No actor-level config
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=failing_handler
        )
        
        # No event-level config
        event = ExecutionEvent(actor=actor)
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Should use type-specific max_attempts=7
        assert fail_count[0] == 7
        assert result.ok is False
    
    @pytest.mark.asyncio
    async def test_state_global_default_overrides_builtin_defaults(self):
        """ExecutionState global default used when no event, actor, or type-specific config."""
        state = ExecutionState()
        state.default_retry_config = RetryConfig(max_attempts=6, initial_delay=0.01, jitter=False)
        
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=failing_handler
        )
        
        event = ExecutionEvent(actor=actor)
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Should use global default max_attempts=6
        assert fail_count[0] == 6
        assert result.ok is False
    
    @pytest.mark.asyncio
    async def test_builtin_defaults_as_fallback(self):
        """Built-in defaults used when no other config is provided."""
        state = ExecutionState()
        # No default_retry_config or retry_config_by_actor_type set
        
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=failing_handler
        )
        
        event = ExecutionEvent(actor=actor)
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Should use built-in default for TOOL_CALL (max_attempts=3 by default)
        assert fail_count[0] == 3
        assert result.ok is False
    
    @pytest.mark.asyncio
    async def test_disabled_config_skips_to_next_level(self):
        """Disabled config at higher level falls back to next level."""
        state = ExecutionState()
        state.default_retry_config = RetryConfig(max_attempts=5, initial_delay=0.01, jitter=False)
        
        fail_count = [0]
        async def failing_handler(input_data, context, state):
            fail_count[0] += 1
            raise ConnectionError("Test error")
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=failing_handler,
            retry_config=RetryConfig(enabled=False)  # Disabled!
        )
        
        event = ExecutionEvent(actor=actor)
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Actor config is disabled, should fall back to global default (5 attempts)
        assert fail_count[0] == 5
        assert result.ok is False


class TestMultiLevelConfigByEventType:
    """Test type-specific configuration for different event types."""
    
    @pytest.mark.asyncio
    async def test_different_configs_for_different_types(self):
        """Different event types can have different retry configs."""
        state = ExecutionState()
        state.retry_config_by_actor_type = {
            EventType.TOOL_CALL: RetryConfig(max_attempts=5, initial_delay=0.01, jitter=False),
            EventType.LLM_CALL: RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False),
            EventType.AGENT_CALL: RetryConfig(enabled=False)
        }
        
        # Test TOOL_CALL
        tool_fail_count = [0]
        async def tool_handler(input_data, context, state):
            tool_fail_count[0] += 1
            raise ConnectionError("Tool error")
        
        tool_actor = BaseActor(
            name="tool",
            description="Tool",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=tool_handler
        )
        
        tool_event = ExecutionEvent(actor=tool_actor)
        tool_result = await tool_event(input_data="test", extra_context={}, execution_state=state)
        
        assert tool_fail_count[0] == 5  # TOOL_CALL config
        assert tool_result.ok is False
        
        # Test LLM_CALL
        llm_fail_count = [0]
        async def llm_handler(input_data, context, state):
            llm_fail_count[0] += 1
            raise ConnectionError("LLM error")
        
        llm_actor = BaseActor(
            name="llm",
            description="LLM",
            input_model=SimpleInput,
            event_type=EventType.LLM_CALL,
            handler=llm_handler
        )
        
        llm_event = ExecutionEvent(actor=llm_actor)
        llm_result = await llm_event(input_data="test", extra_context={}, execution_state=state)
        
        assert llm_fail_count[0] == 2  # LLM_CALL config
        assert llm_result.ok is False
        
        # Test AGENT_CALL (disabled)
        agent_fail_count = [0]
        async def agent_handler(input_data, context, state):
            agent_fail_count[0] += 1
            raise ConnectionError("Agent error")
        
        agent_actor = BaseActor(
            name="agent",
            description="Agent",
            input_model=SimpleInput,
            event_type=EventType.AGENT_CALL,
            handler=agent_handler
        )
        
        agent_event = ExecutionEvent(actor=agent_actor)
        agent_result = await agent_event(input_data="test", extra_context={}, execution_state=state)
        
        assert agent_fail_count[0] == 1  # No retry (disabled)
        assert agent_result.ok is False


class TestDecoratorRetryConfig:
    """Test that decorators properly accept and apply retry config."""
    
    @pytest.mark.asyncio
    async def test_tool_decorator_with_retry_config(self):
        """@tool decorator accepts retry_config parameter."""
        from rh_agents.decorators import tool
        from rh_agents.core.result_types import Tool_Result
        
        call_count = [0]
        
        @tool(
            name="flaky_tool",
            description="A flaky tool",
            retry_config=RetryConfig(max_attempts=4, initial_delay=0.01, jitter=False)
        )
        async def flaky_tool(input: SimpleInput, context: str, state: ExecutionState) -> Tool_Result:
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Flaky error")
            return Tool_Result(output="success", tool_name="flaky_tool")
        
        # Tool should have retry_config set
        assert flaky_tool.retry_config is not None
        assert flaky_tool.retry_config.max_attempts == 4
        
        # Create event and execute
        state = ExecutionState()
        event = ExecutionEvent(actor=flaky_tool)
        result = await event(input_data=SimpleInput(value=1), extra_context={}, execution_state=state)
        
        # Should succeed on attempt 3
        assert call_count[0] == 3
        assert result.ok is True
    
    @pytest.mark.asyncio
    async def test_agent_decorator_with_retry_config(self):
        """@agent decorator accepts retry_config parameter."""
        from rh_agents.decorators import agent
        
        call_count = [0]
        
        @agent(
            name="flaky_agent",
            description="A flaky agent",
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False)
        )
        async def flaky_agent(input: SimpleInput, context: str, state: ExecutionState) -> SimpleInput:
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("Agent error")
            return SimpleInput(value=input.value + 1)
        
        # Agent should have retry_config set
        assert flaky_agent.retry_config is not None
        assert flaky_agent.retry_config.max_attempts == 2
        
        # Create event and execute
        state = ExecutionState()
        event = ExecutionEvent(actor=flaky_agent)
        result = await event(input_data=SimpleInput(value=1), extra_context={}, execution_state=state)
        
        # Should succeed on attempt 2
        assert call_count[0] == 2
        assert result.ok is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
