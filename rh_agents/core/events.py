from __future__ import annotations
import asyncio
import datetime
from time import time
from typing import Self, Union, TypeVar, Generic, Any
from pydantic import BaseModel, Field, field_serializer
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
    detail: Union[str, None] = Field(default=None, description="Optional detailed information about the event")
    tag: str = Field(default="", description="Optional tag for categorizing the event")
    max_detail_length: int = Field(default=500, description="Maximum length of detail string")
    from_cache: bool = Field(default=False, description="Whether the result was recovered from cache")
    
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
        if hasattr(self, '_start_time'):
            self.execution_time = time() - self._start_time
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
    
    async def __try_retrieve_from_cache(self, input_data: Any, execution_state: ExecutionState) -> Union[ExecutionResult[OutputT], None]:
        """
        Attempt to retrieve cached result if caching is enabled.
        For artifacts, checks ExecutionState storage first, then cache backend.
        Returns cached ExecutionResult if found, None otherwise.
        """
        if not self.actor.cacheable:
            return None
        
        from rh_agents.core.cache import compute_cache_key
        
        # Compute cache key based on address, input, actor name and version
        temp_address = execution_state.get_current_address(self.actor.event_type)
        cache_key, input_hash = compute_cache_key(
            temp_address,
            input_data,
            self.actor.name,
            self.actor.version
        )
        
        # For artifacts, check ExecutionState storage first for fast access
        if self.actor.is_artifact:
            artifact = execution_state.storage.get_artifact(cache_key)
            if artifact is not None:
                # Artifact hit in memory! Prepare for return
                self.from_cache = True
                self.execution_time = 0.0
                self.detail = f"[ARTIFACT:MEMORY] {self._serialize_detail(artifact.result)}"
                self.message = f"Recovered from artifact storage (in-memory)"
                await execution_state.add_event(self, ExecutionStatus.RECOVERED)
                return artifact
            
            # Not in memory, check cache backend for persistence
            if execution_state.cache_backend is not None:
                cached = execution_state.cache_backend.get(cache_key)
                if cached is not None:
                    # Reconstruct the result object using actor's output_model if it's a dict
                    if isinstance(cached.result.result, dict) and self.actor.output_model is not None:
                        try:
                            cached.result.result = self.actor.output_model(**cached.result.result)
                        except Exception:
                            pass  # If reconstruction fails, use dict as-is
                    
                    # Artifact hit in cache! Store in ExecutionState for future access
                    execution_state.storage.set_artifact(cache_key, cached.result)
                    self.from_cache = True
                    self.execution_time = 0.0
                    self.detail = f"[ARTIFACT:CACHE] {self._serialize_detail(cached.result.result)}"
                    self.message = f"Recovered from artifact cache (saved at {cached.cached_at})"
                    await execution_state.add_event(self, ExecutionStatus.RECOVERED)
                    return cached.result
            
            # Artifact not found in either location
            return None
        
        # For regular actors, try to get from cache backend
        if execution_state.cache_backend is None:
            return None
            
        cached = execution_state.cache_backend.get(cache_key)
        if cached is None:
            return None
        
        # Reconstruct the result object using actor's output_model if it's a dict
        if isinstance(cached.result.result, dict) and self.actor.output_model is not None:
            try:
                cached.result.result = self.actor.output_model(**cached.result.result)
            except Exception:
                pass  # If reconstruction fails, use dict as-is
        
        # Cache hit! Prepare for return
        self.from_cache = True
        self.execution_time = 0.0
        self.detail = f"[CACHED] {self._serialize_detail(cached.result.result)}"
        self.message = f"Recovered from cache (saved at {cached.cached_at})"
        await execution_state.add_event(self, ExecutionStatus.RECOVERED)
        
        return cached.result
    
    def __store_result_in_cache(self, input_data: Any, execution_result: ExecutionResult[OutputT], execution_state: ExecutionState):
        """
        Store execution result in cache if caching is enabled.
        For artifacts, stores in BOTH ExecutionState storage (for fast access) AND cache backend (for persistence).
        """
        if not self.actor.cacheable:
            return
        
        from rh_agents.core.cache import CachedResult, compute_cache_key
        
        temp_address = execution_state.get_current_address(self.actor.event_type)
        cache_key, input_hash = compute_cache_key(
            temp_address,
            input_data,
            self.actor.name,
            self.actor.version
        )
        
        # For artifacts, store in BOTH ExecutionState storage AND cache backend
        if self.actor.is_artifact:
            # Store in ExecutionState for fast within-session access
            execution_state.storage.set_artifact(cache_key, execution_result)
            
            # Also store in cache backend for cross-execution persistence
            if execution_state.cache_backend is not None:
                cached_result = CachedResult(
                    result=execution_result,
                    input_hash=input_hash,
                    cache_key=cache_key,
                    actor_name=self.actor.name,
                    actor_version=self.actor.version
                )
                execution_state.cache_backend.set(
                    cache_key,
                    cached_result,
                    ttl=self.actor.cache_ttl
                )
            return
        
        # For regular results, store in cache backend (requires cache_backend)
        if execution_state.cache_backend is None:
            return
            
        cached_result = CachedResult(
            result=execution_result,
            input_hash=input_hash,
            cache_key=cache_key,
            actor_name=self.actor.name,
            actor_version=self.actor.version
        )
        
        execution_state.cache_backend.set(
            cache_key,
            cached_result,
            ttl=self.actor.cache_ttl
        )
    
    async def __call__(self, input_data, extra_context, execution_state: ExecutionState) -> ExecutionResult[OutputT]:
        """
        Execute the wrapped actor with full lifecycle management and caching support (async only).
        """
        execution_state.push_context(f'{self.actor.name}{"::" + self.tag if self.tag else ""}')
        
        try:
            # Try to retrieve from cache
            cached_result = await self.__try_retrieve_from_cache(input_data, execution_state)
            if cached_result is not None:
                return cached_result
            
            # Run preconditions
            await self.actor.run_preconditions(input_data, extra_context, execution_state)

            # Start timer and mark as started with input details
            self.start_timer()
            self.detail = self._serialize_detail(input_data)
            await execution_state.add_event(self, ExecutionStatus.STARTED)
            
            # Enforce async handler
            if not asyncio.iscoroutinefunction(self.actor.handler):
                raise TypeError(f"Handler for actor '{self.actor.name}' must be async.")
            result = await self.actor.handler(input_data, extra_context, execution_state)
            
            # Run postconditions
            await self.actor.run_postconditions(result, extra_context, execution_state)

            # Stop timer and mark as completed with result details
            self.stop_timer()
            self.detail = self._serialize_detail(result)
            await execution_state.add_event(self, ExecutionStatus.COMPLETED)
            
            execution_result = ExecutionResult[OutputT](
                result=result,
                execution_time=self.execution_time,
                ok=True
            )
            
            # Store result in cache
            self.__store_result_in_cache(input_data, execution_result, execution_state)
            
            return execution_result

        except Exception as e:
            # Stop timer, mark as failed and capture error message
            self.stop_timer()
            self.message = str(e)
            await execution_state.add_event(self, ExecutionStatus.FAILED)
            return ExecutionResult[OutputT](
                result=None,
                execution_time=self.execution_time,
                ok=False,
                erro_message=str(e)
            )

        finally:
            execution_state.pop_context()
    
    def __call_sync__(self, input_data, execution_state: ExecutionState):
        raise NotImplementedError("Synchronous execution is not supported. Use 'await' on the event.")