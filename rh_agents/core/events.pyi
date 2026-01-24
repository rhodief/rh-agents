from typing import TypeVar, Generic, Awaitable
from pydantic import BaseModel
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState

T = TypeVar('T')

class ExecutionResult(Generic[T]):
    result: T | None
    execution_time: float | None
    ok: bool
    erro_message: str | None

class ExecutionEvent(Generic[T]):
    actor: BaseActor
    async def __call__(
        self, 
        input_data: Any, 
        extra_context: str, 
        execution_state: ExecutionState
    ) -> ExecutionResult[T]: ...
