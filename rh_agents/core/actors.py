import asyncio
from typing import Any, Callable, Coroutine, Generic, TypeVar, Union
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
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: Union[type[BaseModel], None] = None
    handler: AsyncHandler
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.TOOL_CALL
    
    def model_post_init(self, __context) -> None:
        if self.output_model is None:
            from rh_agents.core.result_types import Tool_Result
            self.output_model = Tool_Result
        super().model_post_init(__context) if hasattr(super(), 'model_post_init') else None
    
class ToolSet(BaseModel):
    tools: list[Tool] = Field(default_factory=list)
    __tools: dict[str, Tool] = {}

    def __init__(self, tools: Union[list[Tool], None] = None, **data):
        if tools is None:
            tools = []
        super().__init__(tools=tools, **data)
        self.__tools = {tool.name: tool for tool in self.tools}

    def __iter__(self):
        for name, tool in self.__tools.items():
            yield name, tool

    def get_tool_list(self) -> list[Tool]:
        return list(self.__tools.values())
    
    def __getitem__(self, key: str) -> Tool | None:
        return self.__tools.get(key)
    
    def get(self, key: str) -> Union[Tool, None]:
        return self.__tools.get(key, None)


T = TypeVar('T', bound=Any)

class LLM(BaseActor, Generic[T]):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[LLM_Result] = LLM_Result
    handler: Callable[[T, str, ExecutionState], Coroutine[Any, Any, LLM_Result]]
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.LLM_CALL    
    
    

class Agent(BaseActor):
    name: str
    description: str
    handler: AsyncHandler
    preconditions: list[Callable] = []
    postconditions: list[Callable] = []
    event_type: EventType = EventType.AGENT_CALL
    tools: ToolSet
    llm: LLM | None = None
    
    
