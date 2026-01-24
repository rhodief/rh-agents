from typing import TypeVar, Generic, Awaitable, Any
from pydantic import BaseModel
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import ExecutionStatus

T = TypeVar('T')

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
    
    async def __call__(
        self,
        input_data: Any,
        extra_context: str,
        execution_state: ExecutionState
    ) -> ExecutionResult[T]: ...
