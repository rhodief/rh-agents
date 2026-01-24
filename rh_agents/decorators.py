"""
Decorator-based API for creating actors.

Provides a more Pythonic and concise way to define tools, agents, and LLMs
using decorators, similar to FastAPI or Flask.
"""
from typing import Callable, TypeVar, Any, Awaitable
from functools import wraps
from pydantic import BaseModel
from rh_agents.core.actors import Tool, Agent, LLM, ToolSet
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import Tool_Result, LLM_Result

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


def tool(
    name: str | None = None,
    description: str | None = None,
    cacheable: bool = False,
    version: str = "1.0.0"
) -> Callable[[F], Tool]:
    """
    Decorator to create a Tool from an async function.

    Usage:
        @tool(name="calculator", description="Performs calculations")
        async def calculate(
            input: CalculatorArgs, context: str, state: ExecutionState
        ) -> Tool_Result:
            return Tool_Result(output=input.a + input.b, tool_name="calculator")

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        cacheable: Whether results should be cached
        version: Tool version for cache invalidation
    """
    def decorator(func: F) -> Tool:
        import inspect

        # Extract function metadata
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        # Get type hints to determine input model
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError(
                f"Tool handler {func.__name__} must have at least "
                "one parameter (input_data)"
            )

        input_param = params[0]
        if input_param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Tool handler {func.__name__} first parameter must be "
                "type-annotated"
            )

        input_model = input_param.annotation

        # Wrap function to match expected signature
        @wraps(func)
        async def handler(input_data: BaseModel, context: str, state: ExecutionState) -> Any:
            return await func(input_data, context, state)

        return Tool(
            name=tool_name,
            description=tool_description,
            input_model=input_model,
            handler=handler,
            cacheable=cacheable,
            version=version
        )

    return decorator


def agent(
    name: str | None = None,
    description: str | None = None,
    tools: list[Tool] | None = None,
    llm: LLM | None = None,
    cacheable: bool = False
) -> Callable[[F], Agent]:
    """
    Decorator to create an Agent from an async function.

    Usage:
        @agent(name="DoctrineAgent", tools=[tool1, tool2], llm=my_llm)
        async def handle_doctrine(input: Message, context: str, state: ExecutionState) -> Doctrine:
            # Agent logic here
            return result

    Args:
        name: Agent name (defaults to function name)
        description: Agent description (defaults to function docstring)
        tools: List of tools available to the agent
        llm: LLM instance for the agent
        cacheable: Whether results should be cached
    """
    def decorator(func: F) -> Agent:
        import inspect

        agent_name = name or func.__name__
        agent_description = description or func.__doc__ or f"Agent: {agent_name}"

        # Get type hints
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError(f"Agent handler {func.__name__} must have at least one parameter")

        input_param = params[0]
        if input_param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Agent handler {func.__name__} first parameter must be "
                "type-annotated"
            )

        input_model = input_param.annotation
        output_model = (
            sig.return_annotation
            if sig.return_annotation != inspect.Signature.empty
            else None
        )

        @wraps(func)
        async def handler(input_data: BaseModel, context: str, state: ExecutionState) -> Any:
            return await func(input_data, context, state)

        return Agent(
            name=agent_name,
            description=agent_description,
            input_model=input_model,
            output_model=output_model,  # type: ignore[arg-type]
            handler=handler,
            tools=ToolSet(tools=tools or []),
            llm=llm,
            cacheable=cacheable
        )

    return decorator
