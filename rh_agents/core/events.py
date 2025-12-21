from __future__ import annotations
import asyncio
import datetime
from time import time
from typing import Self, Union, TypeVar, Generic, Any
from pydantic import BaseModel, Field
from rh_agents.core.types import ExecutionStatus
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState

T = TypeVar('T', bound=Any)
OutputT = TypeVar('OutputT', bound=BaseModel)

class ExecutionResult(BaseModel, Generic[T]):
    result: Union[T, None] = Field(default=None, description="Result of the execution")
    execution_time: Union[float, None] = Field(default=None, description="Execution time in seconds")
    ok: bool = Field(default=True, description="Indicates if the execution was successful")
    erro_message: Union[str, None] = Field(default=None, description="Error message if execution failed")

class ExecutionEvent(BaseModel, Generic[OutputT]):
    actor: BaseActor
    datetime: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="Timestamp of the event in milliseconds since epoch")
    address: str = Field(default="", description="Address of the agent triggering the event on exectution tree")
    execution_time: Union[float, None] = Field(default=None, description="Execution time in seconds")
    execution_status: ExecutionStatus = Field(default = ExecutionStatus.STARTED, description="Status of the execution event")
    message: Union[str, None] = Field(default=None, description="Optional message associated with the event")
    
    def start_timer(self):
        self._start_time = time()
    
    def stop_timer(self):
        if hasattr(self, '_start_time'):
            self.execution_time = time() - self._start_time
        else:
            self.execution_time = None
    
    async def __call__(self, input_data, extra_context, execution_state: ExecutionState) -> ExecutionResult[OutputT]:
        """
        Execute the wrapped actor with full lifecycle management (async only).
        """
        execution_state.push_context(self.actor.name)
        try:
            # Run preconditions
            await self.actor.run_preconditions(input_data, extra_context, execution_state)

            # Start timer and mark as started
            self.start_timer()
            execution_state.add_event(self, ExecutionStatus.STARTED)
            # Enforce async handler
            if not asyncio.iscoroutinefunction(self.actor.handler):
                raise TypeError(f"Handler for actor '{self.actor.name}' must be async.")
            result = await self.actor.handler(input_data, execution_state)
            # Run postconditions
            await self.actor.run_postconditions(result, extra_context, execution_state)

            # Stop timer and mark as completed
            self.stop_timer()
            execution_state.add_event(self, ExecutionStatus.COMPLETED)
            return result

        except Exception as e:
            # Stop timer, mark as failed and capture error message
            self.stop_timer()
            self.message = str(e)
            execution_state.add_event(self, ExecutionStatus.FAILED)
            raise

        finally:
            execution_state.pop_context()
    
    def __call_sync__(self, input_data, execution_state: ExecutionState):
        raise NotImplementedError("Synchronous execution is not supported. Use 'await' on the event.")