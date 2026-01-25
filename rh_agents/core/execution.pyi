from typing import Any, Callable, Awaitable, TYPE_CHECKING, Optional
from pydantic import BaseModel
from rh_agents.core.parallel import ErrorStrategy, ParallelExecutionManager
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.state_recovery import ReplayMode

if TYPE_CHECKING:
    from rh_agents.core.events import ExecutionEvent

class ExecutionStore(BaseModel):
    data: dict[str, str]
    artifacts: dict[str, Any]
    
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def get_artifact(self, key: str) -> Any | None: ...
    def set_artifact(self, key: str, value: Any) -> None: ...
    def has_artifact(self, key: str) -> bool: ...

class HistorySet(BaseModel):
    events: list[Any]
    
    def add(self, event: "ExecutionEvent[Any]" | dict[str, Any]) -> None: ...
    def __getitem__(self, address: str) -> "ExecutionEvent[Any]" | dict[str, Any]: ...
    def has_completed_event(self, address: str) -> bool: ...
    def get_event_list(self) -> list["ExecutionEvent[Any]" | dict[str, Any]]: ...

class EventBus:
    subscribers: list[Callable[["ExecutionEvent[Any]"], Awaitable[None]]]
    
    def subscribe(self, handler: Callable[["ExecutionEvent[Any]"], Awaitable[None] | None]) -> None: ...
    async def publish(self, event: "ExecutionEvent[Any]") -> None: ...

class ExecutionState(BaseModel):
    state_id: str
    execution_stack: list[str]
    history: HistorySet
    storage: ExecutionStore
    event_bus: EventBus
    resume_from_address: str | None
    replay_mode: ReplayMode
    state_backend: Any | None
    artifact_backend: Any | None
    parallel_manager: Any | None
    
    def __init__(
        self,
        *,
        state_id: str | None = None,
        execution_stack: list[str] | None = None,
        history: HistorySet | None = None,
        storage: ExecutionStore | None = None,
        event_bus: EventBus | None = None,
        resume_from_address: str | None = None,
        replay_mode: ReplayMode = ReplayMode.NORMAL,
        state_backend: Any | None = None,
        artifact_backend: Any | None = None,
        parallel_manager: Any | None = None
    ) -> None: ...
    
    def push_context(self, name: str) -> None: ...
    def pop_context(self) -> str | None: ...
    def get_current_address(self, event_type: EventType) -> str: ...
    def should_skip_event(self, address: str) -> bool: ...
    async def add_event(self, event: "ExecutionEvent[Any]", status: ExecutionStatus) -> None: ...
    def add_step_result(self, step_index: int, value: Any) -> None: ...
    def get_steps_result(self, step_index: list[int]) -> list[Any]: ...
    def get_last_step_result(self) -> Any | None: ...
    def get_all_steps_results(self) -> dict[int, Any]: ...
    def parallel(
        self,
        max_workers: int = 5,
        error_strategy: Optional['ErrorStrategy'] = None,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ) -> 'ParallelExecutionManager':...
