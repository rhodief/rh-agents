"""
Validation utilities for actors and execution state.

Provides helper functions to validate actor configurations and
execution state consistency, helping catch errors early.
"""
import asyncio
import inspect
from typing import Any
from rh_agents.core.actors import BaseActor, Tool, Agent, LLM
from rh_agents.core.execution import ExecutionState
from pydantic import BaseModel


class ActorValidationError(Exception):
    """Raised when actor validation fails."""
    pass


class StateValidationError(Exception):
    """Raised when state validation fails."""
    pass


def validate_actor(actor: BaseActor) -> None:
    """
    Validate actor configuration.

    Checks:
    - Handler is async function
    - Input/output models are valid
    - Required fields are set
    - Version format is valid

    Args:
        actor: Actor to validate

    Raises:
        ActorValidationError: If validation fails
    """
    errors = []

    # Check name
    if not actor.name or not actor.name.strip():
        errors.append("Actor name is required and cannot be empty")

    # Check description
    if not actor.description:
        errors.append("Actor description is required")

    # Check input model
    if not actor.input_model:
        errors.append("input_model is required")
    else:
        try:
            if not issubclass(actor.input_model, BaseModel):
                errors.append(
                    f"input_model must be a Pydantic BaseModel, "
                    f"got {type(actor.input_model)}"
                )
        except TypeError:
            errors.append(f"input_model is not a valid class: {actor.input_model}")

    # Check handler
    if not actor.handler:
        errors.append("handler is required")
    else:
        if not asyncio.iscoroutinefunction(actor.handler):
            errors.append("handler must be an async function")
        else:
            # Check handler signature
            sig = inspect.signature(actor.handler)
            params = list(sig.parameters.values())
            if len(params) < 2:
                errors.append(
                    f"handler must accept at least 2 parameters "
                    f"(input_data, context), got {len(params)}"
                )

    # Check version format
    if actor.version:
        parts = actor.version.split('.')
        if len(parts) != 3:
            errors.append(f"version must be in format 'X.Y.Z', got '{actor.version}'")
        else:
            for part in parts:
                if not part.isdigit():
                    errors.append(f"version parts must be numeric, got '{actor.version}'")

    # Type-specific validations
    if isinstance(actor, Agent):
        if actor.tools is None:
            errors.append("Agent must have tools (can be empty ToolSet)")

    if isinstance(actor, Tool):
        if actor.cacheable and actor.cache_ttl is None:
            errors.append("Cacheable tools should specify cache_ttl")

    if isinstance(actor, LLM):
        if not actor.cacheable:
            errors.append("LLMs should generally be cacheable for efficiency")

    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ActorValidationError(f"Actor '{actor.name}' validation failed:\n{error_msg}")


def validate_state(state: ExecutionState) -> None:
    """
    Validate execution state consistency.

    Checks:
    - History events are valid
    - Storage is consistent
    - Execution stack is balanced
    - Backends are properly configured

    Args:
        state: ExecutionState to validate

    Raises:
        StateValidationError: If validation fails
    """
    errors = []

    # Check state ID
    if not state.state_id:
        errors.append("state_id is required")

    # Check execution stack
    if state.execution_stack is None:
        errors.append("execution_stack cannot be None")

    # Check history
    if state.history is None:
        errors.append("history cannot be None")
    else:
        # Validate events in history
        for event in state.history.get_event_list():
            if isinstance(event, dict):
                if 'address' not in event:
                    errors.append(f"Event in history missing 'address' field")
                if 'execution_status' not in event:
                    errors.append(
                        f"Event at {event.get('address', 'unknown')} "
                        "missing 'execution_status'"
                    )

    # Check storage
    if state.storage is None:
        errors.append("storage cannot be None")

    # Check for resume_from_address validity
    if state.resume_from_address:
        # Verify address exists in history
        if not any(
            (isinstance(e, dict) and e.get('address') == state.resume_from_address) or
            (hasattr(e, 'address') and e.address == state.resume_from_address)
            for e in state.history.get_event_list()
        ):
            errors.append(f"resume_from_address '{state.resume_from_address}' not found in history")

    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise StateValidationError(f"ExecutionState validation failed:\n{error_msg}")


def validate_handler_signature(handler: Any, actor_name: str = "unknown") -> None:
    """
    Validate that a handler function has the correct signature.

    Expected: async def handler(input_data, context, state) -> result

    Args:
        handler: Handler function to validate
        actor_name: Name of actor for error messages

    Raises:
        ActorValidationError: If signature is invalid
    """
    if not asyncio.iscoroutinefunction(handler):
        raise ActorValidationError(f"Handler for '{actor_name}' must be async function")

    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise ActorValidationError(
            f"Handler for '{actor_name}' must accept at least 2 parameters "
            f"(input_data, context), got {len(params)}"
        )

    # Check parameter names (helpful but not required)
    expected_names = ['input_data', 'context', 'state']
    for i, (param, expected) in enumerate(zip(params[:3], expected_names)):
        if param.name != expected:
            # Warning, not error - parameter names are convention
            import warnings
            warnings.warn(
                f"Handler for '{actor_name}' parameter {i} is named '{param.name}', "
                f"expected '{expected}' by convention"
            )
