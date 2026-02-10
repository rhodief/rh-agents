from typing import TypeVar, Generic, Awaitable, Any, overload
from pydantic import BaseModel
from rh_agents.core.actors import BaseActor, LLM, Tool, Agent
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import ExecutionStatus

T = TypeVar('T')
R = TypeVar('R')
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT')

class ExecutionResult(Generic[T]):
    result: T | None
    execution_time: float | None
    ok: bool
    erro_message: str | None
    
    def __init__(
        self,
        *,
        result: T | None = None,
        execution_time: float | None = None,
        ok: bool = True,
        erro_message: str | None = None
    ) -> None: ...

class ExecutionEvent(Generic[T]):
    actor: BaseActor
    datetime: str
    address: str
    execution_time: float | None
    execution_status: ExecutionStatus
    message: str | None
    detail: str | None
    tag: str
    max_detail_length: int
    from_cache: bool
    result: Any | None
    is_replayed: bool
    skip_republish: bool
    group_id: str | None
    parallel_index: int | None
    is_parallel: bool
    
    retry_config: Any |  None
    retry_attempt: int
    is_retry: bool
    original_error: str | None
    retry_delay: float | None
    
    # Overload for LLM actors with known output type
    @overload
    def __init__(
        self,
        *,
        actor: LLM[InputT, R],
        datetime: str | None = None,
        address: str = "",
        execution_time: float | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.STARTED,
        message: str | None = None,
        detail: str | None = None,
        tag: str = "",
        max_detail_length: int = 500,
        from_cache: bool = False,
        result: Any | None = None,
        is_replayed: bool = False,
        skip_republish: bool = False,
        group_id: str | None = None,
        parallel_index: int | None = None,
        is_parallel: bool = False
    ) -> None: ...
    
    # Overload for Agent actors with known output type
    @overload
    def __init__(
        self,
        *,
        actor: Agent[InputT, R],
        datetime: str | None = None,
        address: str = "",
        execution_time: float | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.STARTED,
        message: str | None = None,
        detail: str | None = None,
        tag: str = "",
        max_detail_length: int = 500,
        from_cache: bool = False,
        result: Any | None = None,
        is_replayed: bool = False,
        skip_republish: bool = False,
        group_id: str | None = None,
        parallel_index: int | None = None,
        is_parallel: bool = False
    ) -> None: ...
    
    # Overload for Tool actors (return Any)
    @overload
    def __init__(
        self,
        *,
        actor: Tool[InputT],
        datetime: str | None = None,
        address: str = "",
        execution_time: float | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.STARTED,
        message: str | None = None,
        detail: str | None = None,
        tag: str = "",
        max_detail_length: int = 500,
        from_cache: bool = False,
        result: Any | None = None,
        is_replayed: bool = False,
        skip_republish: bool = False,
        group_id: str | None = None,
        parallel_index: int | None = None,
        is_parallel: bool = False
    ) -> None: ...
    
    # Generic fallback for any BaseActor
    @overload
    def __init__(
        self,
        *,
        actor: BaseActor,
        datetime: str | None = None,
        address: str = "",
        execution_time: float | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.STARTED,
        message: str | None = None,
        detail: str | None = None,
        tag: str = "",
        max_detail_length: int = 500,
        from_cache: bool = False,
        result: Any | None = None,
        is_replayed: bool = False,
        skip_republish: bool = False,
        group_id: str | None = None,
        parallel_index: int | None = None,
        is_parallel: bool = False
    ) -> None: ...
    
    def start_timer(self) -> None: ...
    def stop_timer(self) -> None: ...
    def model_dump_json(self) -> str: ...
    
    # Overload for __call__ with proper return type inference
    async def __call__(
        self: ExecutionEvent[R],
        input_data: Any,
        extra_context: str,
        execution_state: ExecutionState
    ) -> ExecutionResult[R]: ...
