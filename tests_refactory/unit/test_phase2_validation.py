"""
Phase 2 Unit Tests: Validation Utilities

Tests the validation helpers for actors and execution state.
"""
import pytest
from pydantic import BaseModel
from rh_agents import (
    validate_actor,
    validate_state,
    validate_handler_signature,
    ActorValidationError,
    StateValidationError,
    Tool,
    Agent,
    LLM,
    ExecutionState,
    ToolSet
)
from rh_agents.core.result_types import Tool_Result


class DummyInput(BaseModel):
    data: str


class TestValidateActor:
    """Test validate_actor function."""
    
    def test_validate_valid_tool(self):
        """Test that valid tool passes validation."""
        async def handler(input: DummyInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool = Tool(
            name="valid_tool",
            description="A valid tool",
            input_model=DummyInput,
            handler=handler
        )
        
        # Should not raise
        validate_actor(tool)
    
    def test_validate_empty_name(self):
        """Test that empty name fails validation."""
        async def handler(input: DummyInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool = Tool(
            name="",
            description="Description",
            input_model=DummyInput,
            handler=handler
        )
        
        with pytest.raises(ActorValidationError, match="name is required"):
            validate_actor(tool)
    
    def test_validate_missing_description(self):
        """Test that missing description fails validation."""
        async def handler(input: DummyInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool = Tool(
            name="test_tool",
            description="",
            input_model=DummyInput,
            handler=handler
        )
        
        with pytest.raises(ActorValidationError, match="description is required"):
            validate_actor(tool)
    
    def test_validate_missing_input_model(self):
        """Test that missing input_model fails validation at construction."""
        async def handler(input: DummyInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        # Pydantic validates input_model at construction time, so we can't create an invalid Tool
        # This tests that Pydantic catches the error
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            Tool(
                name="test_tool",
                description="Description",
                input_model=None,
                handler=handler
            )
    
    def test_validate_non_async_handler(self):
        """Test that non-async handler fails validation."""
        def sync_handler(input: DummyInput, context: str, state: ExecutionState):
            return Tool_Result(output="result", tool_name="test")
        
        tool = Tool(
            name="test_tool",
            description="Description",
            input_model=DummyInput,
            handler=sync_handler
        )
        
        with pytest.raises(ActorValidationError, match="must be an async function"):
            validate_actor(tool)
    
    def test_validate_invalid_version_format(self):
        """Test that invalid version format fails validation."""
        async def handler(input: DummyInput, context: str, state: ExecutionState) -> Tool_Result:
            return Tool_Result(output="result", tool_name="test")
        
        tool = Tool(
            name="test_tool",
            description="Description",
            input_model=DummyInput,
            handler=handler,
            version="1.0"  # Should be X.Y.Z
        )
        
        with pytest.raises(ActorValidationError, match="version must be in format"):
            validate_actor(tool)
    
    def test_validate_agent_without_tools(self):
        """Test that agent without tools fails validation at construction."""
        async def handler(input: DummyInput, context: str, state: ExecutionState):
            return {"result": "done"}
        
        # Pydantic validates tools at construction time
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            Agent(
                name="test_agent",
                description="Description",
                input_model=DummyInput,
                handler=handler,
                tools=None
            )


class TestValidateState:
    """Test validate_state function."""
    
    def test_validate_valid_state(self):
        """Test that valid state passes validation."""
        state = ExecutionState()
        
        # Should not raise
        validate_state(state)
    
    def test_validate_missing_state_id(self):
        """Test that missing state_id fails validation."""
        state = ExecutionState()
        state.state_id = ""
        
        with pytest.raises(StateValidationError, match="state_id is required"):
            validate_state(state)
    
    def test_validate_none_execution_stack(self):
        """Test that None execution_stack fails validation."""
        state = ExecutionState()
        state.execution_stack = None
        
        with pytest.raises(StateValidationError, match="execution_stack cannot be None"):
            validate_state(state)
    
    def test_validate_none_history(self):
        """Test that None history fails validation."""
        state = ExecutionState()
        state.history = None
        
        with pytest.raises(StateValidationError, match="history cannot be None"):
            validate_state(state)


class TestValidateHandlerSignature:
    """Test validate_handler_signature function."""
    
    def test_validate_correct_signature(self):
        """Test that correct signature passes validation."""
        async def handler(input_data: DummyInput, context: str, state: ExecutionState):
            return "result"
        
        # Should not raise
        validate_handler_signature(handler, "test_actor")
    
    def test_validate_sync_function_fails(self):
        """Test that sync function fails validation."""
        def sync_handler(input_data: DummyInput, context: str, state: ExecutionState):
            return "result"
        
        with pytest.raises(ActorValidationError, match="must be async function"):
            validate_handler_signature(sync_handler, "test_actor")
    
    def test_validate_insufficient_parameters(self):
        """Test that insufficient parameters fails validation."""
        async def bad_handler(input_data: DummyInput):
            return "result"
        
        with pytest.raises(ActorValidationError, match="must accept at least 2 parameters"):
            validate_handler_signature(bad_handler, "test_actor")
    
    def test_validate_warns_on_parameter_names(self):
        """Test that non-standard parameter names generate warnings."""
        import warnings
        
        async def handler(inp: DummyInput, ctx: str, st: ExecutionState):
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_handler_signature(handler, "test_actor")
            
            # Should have warnings about parameter names
            assert len(w) > 0
            assert "parameter" in str(w[0].message).lower()
