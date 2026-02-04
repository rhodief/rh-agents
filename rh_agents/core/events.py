from __future__ import annotations
import asyncio
import datetime
from time import time
from typing import Self, Union, Any, Generic, TypeVar, Dict
from pydantic import BaseModel, Field, field_serializer
from rh_agents.core.types import ExecutionStatus, InterruptSignal
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState

T = TypeVar('T')


class ExecutionResult(BaseModel, Generic[T]):
    result: Union[T, None] = Field(default=None, description="Result of the execution")
    execution_time: Union[float, None] = Field(default=None, description="Execution time in seconds")
    ok: bool = Field(default=True, description="Indicates if the execution was successful")
    erro_message: Union[str, None] = Field(default=None, description="Error message if execution failed")

class ExecutionEvent(BaseModel, Generic[T]):
    actor: BaseActor
    datetime: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="Timestamp of the event in milliseconds since epoch")
    address: str = Field(default="", description="Address of the agent triggering the event on exectution tree")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata to store relevant information for the event")
    execution_time: Union[float, None] = Field(default=None, description="Execution time in seconds")
    execution_status: ExecutionStatus = Field(default = ExecutionStatus.STARTED, description="Status of the execution event")
    message: Union[str, None] = Field(default=None, description="Optional message associated with the event")
    detail: Union[str, None] = Field(default=None, description="Optional detailed information about the event")
    tag: str = Field(default="", description="Optional tag for categorizing the event")
    max_detail_length: int = Field(default=500, description="Maximum length of detail string")
    from_cache: bool = Field(default=False, description="Whether the result was recovered from cache")
    
    # State recovery fields
    result: Union[Any, None] = Field(default=None, description="Actual result of the execution (for replay)")
    is_replayed: bool = Field(default=False, description="True if event was recovered from restored state")
    skip_republish: bool = Field(default=False, description="If True, skip publishing to event bus (replay control)")
    
    # Parallel execution fields
    group_id: Union[str, None] = Field(default=None, description="Parallel group ID if part of parallel execution")
    parallel_index: Union[int, None] = Field(default=None, description="Index within parallel group")
    is_parallel: bool = Field(default=False, description="True if event is part of parallel group")
    
    @field_serializer('actor')
    def serialize_actor(self, actor: BaseActor) -> dict:
        """Serialize actor to a JSON-safe dict with only relevant fields."""
        return {
            "name": actor.name,
            "event_type": actor.event_type.value,
        }
    
    
    def start_timer(self):
        self._start_time = time()
    
    def stop_timer(self):
        start_time = getattr(self, '_start_time', None)
        if start_time is not None:
            self.execution_time = time() - start_time
        else:
            self.execution_time = None
    
    def _serialize_detail(self, data: Any) -> str:
        """Serialize data to string and truncate if needed."""
        try:
            if isinstance(data, BaseModel):
                serialized = data.model_dump_json(indent=2)
            elif isinstance(data, (dict, list)):
                import json
                serialized = json.dumps(data, indent=2, default=str)
            else:
                serialized = str(data)
            
            if len(serialized) > self.max_detail_length:
                return serialized[:self.max_detail_length] + "..."
            return serialized
        except Exception:
            return str(data)[:self.max_detail_length]
    

    
    async def __call__(self, input_data, extra_context, execution_state: ExecutionState) -> ExecutionResult[T]:
        """
        Execute the wrapped actor with replay awareness and state recovery support.
        
        Flow:
        1. Check if this event should be skipped during replay (already executed)
        2. If skipped, return cached result from history
        3. Otherwise, execute normally and store result for future replay
        """
        execution_state.push_context(f'{self.actor.name}{"::" + self.tag if self.tag else ""}')
        
        try:
            # INTERRUPT CHECK 1: Before any processing
            await execution_state.check_interrupt()
            
            current_address = execution_state.get_current_address(self.actor.event_type)
            
            # PHASE 2: REPLAY LOGIC
            # Check if we should skip this event (already executed in restored state)
            if execution_state.should_skip_event(current_address):
                # Event already executed - retrieve result from history
                existing_event = execution_state.history[current_address]
                
                # Mark as replayed
                self.is_replayed = True
                self.from_cache = True  # Keep for backward compatibility
                self.execution_time = 0.0
                
                # Decide whether to republish based on replay mode
                from rh_agents.core.state_recovery import ReplayMode
                if execution_state.replay_mode == ReplayMode.REPUBLISH_ALL:
                    self.skip_republish = False
                else:
                    self.skip_republish = True  # Don't republish old events
                
                # Copy event details from history (handle both dict and object)
                if isinstance(existing_event, dict):
                    self.detail = f"[REPLAYED] {existing_event.get('detail', '')}"
                    stored_result = existing_event.get('result')
                else:
                    self.detail = f"[REPLAYED] {getattr(existing_event, 'detail', '')}"
                    stored_result = getattr(existing_event, 'result', None)
                
                self.message = "Recovered from restored state"
                
                # Publish recovery event if needed
                await execution_state.add_event(self, ExecutionStatus.RECOVERED)  # type: ignore[arg-type]
                
                # Return the stored result
                if stored_result is not None:
                    # If it's a dict (from deserialization), try to reconstruct the model
                    if isinstance(stored_result, dict):
                        # Try to reconstruct as the output model if available
                        if self.actor.output_model:
                            try:
                                stored_result = self.actor.output_model.model_validate(stored_result)
                            except Exception:
                                # If reconstruction fails, use dict as-is
                                pass
                    
                    return ExecutionResult(
                        result=stored_result,  # type: ignore[arg-type]
                        execution_time=0.0,
                        ok=True
                    )
                else:
                    # If no result found, log warning but continue to re-execute
                    print(f"Warning: No result found for replayed event at {current_address}, re-executing...")
            
            # NOT IN REPLAY MODE or EVENT NOT FOUND or VALIDATION MODE
            # INTERRUPT CHECK 2: Before preconditions
            await execution_state.check_interrupt()
            
            # Run preconditions
            await self.actor.run_preconditions(input_data, extra_context, execution_state)

            # INTERRUPT CHECK 3: After preconditions, before execution
            await execution_state.check_interrupt()
            
            # Start timer and mark as started with input details
            self.start_timer()
            self.detail = self._serialize_detail(input_data)
            await execution_state.add_event(self, ExecutionStatus.STARTED)  # type: ignore[arg-type]
            
            # Enforce async handler
            if not asyncio.iscoroutinefunction(self.actor.handler):
                raise TypeError(f"Handler for actor '{self.actor.name}' must be async.")
            result = await self.actor.handler(input_data, extra_context, execution_state)
            
            # INTERRUPT CHECK 4: After handler execution
            await execution_state.check_interrupt()
            
            # Run postconditions
            await self.actor.run_postconditions(result, extra_context, execution_state)

            # Stop timer and mark as completed with result details
            self.stop_timer()
            self.detail = self._serialize_detail(result)
            
            # PHASE 2: Store result in event for replay
            self.result = result
            
            # Store result as artifact if actor produces artifacts
            if self.actor.is_artifact and execution_state.artifact_backend is not None:
                from rh_agents.state_backends import compute_artifact_id
                artifact_id = compute_artifact_id(result)
                execution_state.storage.set_artifact(artifact_id, result)
                execution_state.artifact_backend.save_artifact(artifact_id, result)
            
            await execution_state.add_event(self, ExecutionStatus.COMPLETED)  # type: ignore[arg-type]
            
            execution_result = ExecutionResult(
                result=result,
                execution_time=self.execution_time,
                ok=True
            )
            
            # VALIDATION MODE: Compare with historical result if it exists
            from rh_agents.core.state_recovery import ReplayMode
            if execution_state.replay_mode == ReplayMode.VALIDATION:
                if execution_state.history.has_completed_event(current_address):
                    historical_event = execution_state.history[current_address]
                    # Handle both ExecutionEvent objects and dict representations
                    historical_result = None
                    if isinstance(historical_event, dict):
                        historical_result = historical_event.get('result')
                    else:
                        historical_result = getattr(historical_event, 'result', None)
                    
                    if historical_result is not None and historical_result != result:
                        print(f"WARNING: Validation mismatch at {current_address}")
                        print(f"  Historical: {historical_result}")
                        print(f"  Current: {result}")
            
            return execution_result

        except Exception as e:
            # Check if it's an interrupt exception
            from rh_agents.core.exceptions import ExecutionInterrupted
            
            if isinstance(e, ExecutionInterrupted):
                # Handle interruption gracefully
                self.stop_timer()
                self.message = e.message
                self.detail = f"Interrupted: {e.reason.value}"
                await execution_state.add_event(self, ExecutionStatus.INTERRUPTED)  # type: ignore[arg-type]
                
                return ExecutionResult(
                    result=None,
                    execution_time=self.execution_time,
                    ok=False,
                    erro_message=e.message
                )
            else:
                # Stop timer, mark as failed and capture error message
                self.stop_timer()
                self.message = str(e)
                await execution_state.add_event(self, ExecutionStatus.FAILED)  # type: ignore[arg-type]
                return ExecutionResult(
                    result=None,
                    execution_time=self.execution_time,
                    ok=False,
                    erro_message=str(e)
                )

        finally:
            execution_state.pop_context()
    
    def __call_sync__(self, input_data, execution_state: ExecutionState):
        raise NotImplementedError("Synchronous execution is not supported. Use 'await' on the event.")