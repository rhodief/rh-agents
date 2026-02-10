import asyncio
from typing import Any, Callable, Coroutine, Union
from pydantic import BaseModel, Field
from rh_agents.core.result_types import LLM_Result
from rh_agents.core.types import EventType
from rh_agents.core.execution import ExecutionState

# Type alias for async handlers
AsyncHandler = Callable[..., Coroutine[Any, Any, Any]]


class BaseActor(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel]
    handler: AsyncHandler
    output_model: Union[type[BaseModel], None] = None
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType
    cacheable: bool = Field(default=False, description="Whether execution results should be cached")
    version: str = Field(default="1.0.0", description="Actor version for cache invalidation")
    cache_ttl: Union[int, None] = Field(default=None, description="Cache TTL in seconds, None means no expiration")
    is_artifact: bool = Field(default=False, description="Whether this actor produces artifacts that should be stored separately")
    # Type: RetryConfig (from rh_agents.core.retry) - using Any to avoid circular import
    retry_config: Union[Any, None] = Field(default=None, description="Actor-level retry configuration")
    
    async def run_preconditions(self, input_data, extra_context, execution_state: ExecutionState):
        """Run all precondition checks before execution (async)"""
        for precondition in self.preconditions:
            if asyncio.iscoroutinefunction(precondition):
                await precondition(input_data, extra_context, execution_state)
            else:
                precondition(input_data, extra_context, execution_state)

    async def run_postconditions(self, result, extra_context, execution_state: ExecutionState):
        for postcondition in self.postconditions:
            if asyncio.iscoroutinefunction(postcondition):
                await postcondition(result, extra_context, execution_state)
            else:
                postcondition(result, extra_context, execution_state)


class Tool(BaseActor):
    """
    Tool actor for executing specific functions.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, cache_ttl, is_artifact
    """
    # Override defaults
    event_type: EventType = EventType.TOOL_CALL
    cacheable: bool = Field(default=False, description="Tools are not cacheable by default (side effects)")
    
    def model_post_init(self, __context) -> None:
        if self.output_model is None:
            from rh_agents.core.result_types import Tool_Result
            self.output_model = Tool_Result
        super().model_post_init(__context) if hasattr(super(), 'model_post_init') else None
    
class ToolSet(BaseModel):
    """Collection of tools with efficient name-based lookup."""
    tools: list[Tool] = Field(default_factory=list)
    
    @property
    def by_name(self) -> dict[str, Tool]:
        """Get tools as a name-indexed dictionary."""
        return {tool.name: tool for tool in self.tools}
    
    def __iter__(self):
        """Iterate over tools."""
        return iter(self.tools)
    
    def __getitem__(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self.by_name.get(name)
    
    def get(self, name: str) -> Tool | None:
        """Get tool by name, returns None if not found."""
        return self.by_name.get(name)
    
    def __len__(self) -> int:
        """Return number of tools."""
        return len(self.tools)


class LLM(BaseActor):
    """
    LLM actor for language model calls.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, is_artifact
    """
    # Override defaults
    output_model: type[LLM_Result] = LLM_Result
    event_type: EventType = EventType.LLM_CALL
    cacheable: bool = Field(default=True, description="LLM calls are cacheable by default")
    cache_ttl: int | None = Field(default=3600, description="Default 1 hour TTL for LLM results")    
    
    

class Agent(BaseActor):
    """
    Agent actor for orchestrating tools and LLMs.
    
    Inherits from BaseActor:
        name, description, input_model, handler, output_model,
        preconditions, postconditions, version, cache_ttl, is_artifact
        
    Additional fields:
        tools: Collection of tools available to the agent
        llm: Optional LLM for the agent to use
    """
    # Override defaults
    event_type: EventType = EventType.AGENT_CALL
    cacheable: bool = Field(default=False, description="Agents are not cacheable by default")
    
    # Agent-specific fields
    tools: ToolSet
    llm: LLM | None = None
    
    
