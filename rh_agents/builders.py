"""
Builder pattern for constructing complex actors.

Provides fluent API for building agents with many optional parameters,
making construction more readable and maintainable.
"""
from typing import Callable, Awaitable, Any
from pydantic import BaseModel
from rh_agents.core.actors import Agent, LLM, Tool, ToolSet
from rh_agents.core.execution import ExecutionState


class AgentBuilder:
    """
    Fluent builder for Agent instances.

    Usage:
        agent = (
            AgentBuilder()
            .name("MyAgent")
            .description("Does cool stuff")
            .with_llm(my_llm)
            .with_tools([tool1, tool2])
            .cacheable(True)
            .build()
        )
    """

    def __init__(self):
        self._name: str | None = None
        self._description: str | None = None
        self._input_model: type[BaseModel] | None = None
        self._output_model: type[BaseModel] | None = None
        self._handler: Callable[[BaseModel, str, ExecutionState], Awaitable[Any]] | None = None
        self._tools: list[Tool] = []
        self._llm: LLM | None = None
        self._cacheable: bool = False
        self._version: str = "1.0.0"
        self._preconditions: list[Callable] = []
        self._postconditions: list[Callable] = []

    def name(self, name: str) -> 'AgentBuilder':
        """Set agent name."""
        self._name = name
        return self

    def description(self, description: str) -> 'AgentBuilder':
        """Set agent description."""
        self._description = description
        return self

    def input_model(self, model: type[BaseModel]) -> 'AgentBuilder':
        """Set input model."""
        self._input_model = model
        return self

    def output_model(self, model: type[BaseModel]) -> 'AgentBuilder':
        """Set output model."""
        self._output_model = model
        return self

    def handler(self, handler: Callable) -> 'AgentBuilder':
        """Set handler function."""
        self._handler = handler
        return self

    def with_llm(self, llm: LLM) -> 'AgentBuilder':
        """Attach an LLM to the agent."""
        self._llm = llm
        return self

    def with_tools(self, tools: list[Tool]) -> 'AgentBuilder':
        """Set tools available to the agent."""
        self._tools = tools
        return self

    def add_tool(self, tool: Tool) -> 'AgentBuilder':
        """Add a single tool to the agent."""
        self._tools.append(tool)
        return self

    def cacheable(self, cacheable: bool = True) -> 'AgentBuilder':
        """Set whether agent results should be cached."""
        self._cacheable = cacheable
        return self

    def version(self, version: str) -> 'AgentBuilder':
        """Set agent version."""
        self._version = version
        return self

    def add_precondition(self, condition: Callable) -> 'AgentBuilder':
        """Add a precondition check."""
        self._preconditions.append(condition)
        return self

    def add_postcondition(self, condition: Callable) -> 'AgentBuilder':
        """Add a postcondition check."""
        self._postconditions.append(condition)
        return self

    def build(self) -> Agent:
        """
        Build the Agent instance.

        Returns:
            Configured Agent

        Raises:
            ValueError: If required fields are missing
        """
        if not self._name:
            raise ValueError("Agent name is required")
        if not self._description:
            raise ValueError("Agent description is required")
        if not self._input_model:
            raise ValueError("Agent input_model is required")
        if not self._handler:
            raise ValueError("Agent handler is required")

        return Agent(
            name=self._name,
            description=self._description,
            input_model=self._input_model,
            output_model=self._output_model,
            handler=self._handler,
            tools=ToolSet(tools=self._tools),
            llm=self._llm,
            cacheable=self._cacheable,
            version=self._version,
            preconditions=self._preconditions,
            postconditions=self._postconditions
        )


class ToolBuilder:
    """
    Fluent builder for Tool instances.

    Usage:
        tool = (
            ToolBuilder()
            .name("MyTool")
            .description("Does something")
            .input_model(MyInput)
            .handler(my_handler)
            .cacheable(True)
            .build()
        )
    """

    def __init__(self):
        self._name: str | None = None
        self._description: str | None = None
        self._input_model: type[BaseModel] | None = None
        self._output_model: type[BaseModel] | None = None
        self._handler: Callable | None = None
        self._cacheable: bool = False
        self._version: str = "1.0.0"
        self._cache_ttl: int | None = None

    def name(self, name: str) -> 'ToolBuilder':
        """Set tool name."""
        self._name = name
        return self

    def description(self, description: str) -> 'ToolBuilder':
        """Set tool description."""
        self._description = description
        return self

    def input_model(self, model: type[BaseModel]) -> 'ToolBuilder':
        """Set input model."""
        self._input_model = model
        return self

    def output_model(self, model: type[BaseModel]) -> 'ToolBuilder':
        """Set output model."""
        self._output_model = model
        return self

    def handler(self, handler: Callable) -> 'ToolBuilder':
        """Set handler function."""
        self._handler = handler
        return self

    def cacheable(self, cacheable: bool = True, ttl: int | None = None) -> 'ToolBuilder':
        """Set caching configuration."""
        self._cacheable = cacheable
        if ttl is not None:
            self._cache_ttl = ttl
        return self

    def version(self, version: str) -> 'ToolBuilder':
        """Set tool version."""
        self._version = version
        return self

    def build(self) -> Tool:
        """Build the Tool instance."""
        if not self._name:
            raise ValueError("Tool name is required")
        if not self._description:
            raise ValueError("Tool description is required")
        if not self._input_model:
            raise ValueError("Tool input_model is required")
        if not self._handler:
            raise ValueError("Tool handler is required")

        return Tool(
            name=self._name,
            description=self._description,
            input_model=self._input_model,
            output_model=self._output_model,
            handler=self._handler,
            cacheable=self._cacheable,
            version=self._version,
            cache_ttl=self._cache_ttl
        )
