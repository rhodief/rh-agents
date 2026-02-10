from typing import TypeVar, Generic, Awaitable, Callable, Any, overload
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, Tool_Result
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
    retry_config: Any | None
    
    async def run_preconditions(self, input_data: Any, extra_context: str, execution_state: ExecutionState) -> None: ...
    async def run_postconditions(self, result: Any, extra_context: str, execution_state: ExecutionState) -> None: ...

class LLM(BaseActor, Generic[T, R]):
    output_model: type[R]
    
    @overload
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        *,
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = True,
        version: str = "1.0.0",
        cache_ttl: int | None = 3600,
        is_artifact: bool = False
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = True,
        version: str = "1.0.0",
        cache_ttl: int | None = 3600,
        is_artifact: bool = False,
        **kwargs: Any
    ) -> None: ...

class Tool(BaseActor, Generic[T]):
    output_model: type[Tool_Result]
    
    @overload
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        handler: Callable[[T, str, ExecutionState], Awaitable[Any]],
        *,
        output_model: type[BaseModel] | None = None,
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = False,
        version: str = "1.0.0",
        cache_ttl: int | None = None,
        is_artifact: bool = False
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[T],
        handler: Callable[[T, str, ExecutionState], Awaitable[Any]],
        output_model: type[BaseModel] | None = None,
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = False,
        version: str = "1.0.0",
        cache_ttl: int | None = None,
        is_artifact: bool = False,
        **kwargs: Any
    ) -> None: ...

class Agent(BaseActor, Generic[T, R]):
    tools: ToolSet
    llm: LLM[Any, Any] | None
    output_model: type[R]
    
    @overload
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        *,
        tools: ToolSet | None = None,
        llm: LLM[Any, Any] | None = None,
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = False,
        version: str = "1.0.0",
        cache_ttl: int | None = None,
        is_artifact: bool = False
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        tools: ToolSet | None = None,
        llm: LLM[Any, Any] | None = None,
        preconditions: list[Callable] = [],
        postconditions: list[Callable] = [],
        cacheable: bool = False,
        version: str = "1.0.0",
        cache_ttl: int | None = None,
        is_artifact: bool = False,
        **kwargs: Any
    ) -> None: ...

class ToolSet(BaseModel):
    tools: list[Tool[Any]]
    @property
    def by_name(self) -> dict[str, Tool[Any]]: ...
    def __iter__(self) -> Any: ...
    def __getitem__(self, name: str) -> Tool[Any] | None: ...
    def get(self, name: str) -> Tool[Any] | None: ...
    def __len__(self) -> int: ...
