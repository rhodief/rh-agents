from typing import TypeVar, Generic, Awaitable, Callable, Any
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result
from rh_agents.core.types import EventType

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')

class BaseActor:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    handler: Callable[..., Awaitable[Any]]
    preconditions: list[Callable]
    postconditions: list[Callable]
    event_type: EventType
    cacheable: bool
    version: str
    cache_ttl: int | None
    is_artifact: bool
    
    async def run_preconditions(self, input_data: Any, extra_context: str, execution_state: ExecutionState) -> None: ...
    async def run_postconditions(self, result: Any, extra_context: str, execution_state: ExecutionState) -> None: ...

class LLM(BaseActor, Generic[T, R]):
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        **kwargs
    ) -> None: ...

class Tool(BaseActor, Generic[T]):
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        handler: Callable[[T, str, ExecutionState], Awaitable[Any]],
        **kwargs
    ) -> None: ...

class Agent(BaseActor, Generic[T, R]):
    tools: ToolSet
    llm: LLM | None
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        **kwargs
    ) -> None: ...

class ToolSet(BaseModel):
    tools: list[Tool]
    @property
    def by_name(self) -> dict[str, Tool]: ...
    def __iter__(self): ...
    def __getitem__(self, name: str) -> Tool | None: ...
    def get(self, name: str) -> Tool | None: ...
    def __len__(self) -> int: ...
