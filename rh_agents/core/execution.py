from __future__ import annotations
import asyncio
import uuid
import logging
from collections.abc import AsyncGenerator
from typing import Any, Callable, Union, Optional, Awaitable, AsyncGenerator, TYPE_CHECKING
from pydantic import BaseModel, Field, field_serializer
if TYPE_CHECKING:
    from rh_agents.core.events import ExecutionEvent
    from rh_agents.core.state_backend import StateBackend, ArtifactBackend
    from rh_agents.core.state_recovery import StateSnapshot, StateStatus, StateMetadata
    from rh_agents.core.parallel import ParallelExecutionManager, ErrorStrategy
from rh_agents.core.types import EventType, ExecutionStatus, InterruptReason, InterruptSignal
from rh_agents.core.state_recovery import ReplayMode
import inspect

# Type alias for interrupt checker function
InterruptChecker = Union[
    Callable[[], bool],                                    # Sync function returning bool
    Callable[[], Awaitable[bool]],                        # Async function returning bool
    Callable[[], InterruptSignal],                        # Sync function returning signal details
    Callable[[], Awaitable[InterruptSignal]],             # Async function returning signal details
    Callable[[], Optional[InterruptSignal]],              # Sync function returning signal or None
    Callable[[], Awaitable[Optional[InterruptSignal]]]    # Async function returning signal or None
]

class ExecutionStore(BaseModel):
    data: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    
    def get(self, key: str) -> str | None:
        return self.data.get(key, None)
    
    def set(self, key: str, value: str):
        self.data[key] = value
    
    def get_artifact(self, key: str) -> Any | None:
        """Retrieve an artifact by key."""
        return self.artifacts.get(key, None)
    
    def set_artifact(self, key: str, value: Any):
        """Store an artifact by key."""
        self.artifacts[key] = value
    
    def has_artifact(self, key: str) -> bool:
        """Check if an artifact exists."""
        return key in self.artifacts


class HistorySet(BaseModel):
    events: list[Any] = Field(default_factory=list)  # ExecutionEvent | dict at runtime
    __events: dict[str, Any] = {}

    def __init__(self, events: list[Union['ExecutionEvent', dict[str, Any]]] | None = None, **data):  # type: ignore[name-defined]
        if events is None:
            events = []
        super().__init__(events=events, **data)
        # Handle both ExecutionEvent objects and dicts (from deserialization)
        self.__events = {}
        for event in self.events:
            if isinstance(event, dict):
                # Store dict as-is, use 'address' key
                address = event.get('address', '')
                if address:
                    self.__events[address] = event
            else:
                # It's an ExecutionEvent object
                self.__events[event.address] = event

    def __iter__(self):
        for address, event in self.__events.items():
            yield address, event

    def get_event_list(self) -> list[Union['ExecutionEvent', dict[str, Any]]]:
        return list(self.__events.values())
    
    def __getitem__(self, key: str) -> Union['ExecutionEvent', dict[str, Any]]:
        return self.__events[key]
    
    def __setitem__(self, key: str, value: Union['ExecutionEvent', dict[str, Any]]):
        """Store event and update latest event lookup for the address"""
        self.__events[key] = value
        self.events.append(value)
        
    def add(self, event: Union['ExecutionEvent', dict[str, Any]]):
        """Add event to history, keeping all events while updating latest lookup"""
        if isinstance(event, dict):
            address = event.get('address', '')
        else:
            address = event.address
        self.__setitem__(address, event)
    
    def get_event_result(self, address: str) -> Optional[Any]:
        """Get the result from a completed event at given address."""
        if address in self.__events:
            event = self.__events[address]
            # Handle both dict and ExecutionEvent objects
            if isinstance(event, dict):
                if event.get('execution_status') == ExecutionStatus.COMPLETED.value:
                    return event.get('result')
            else:
                if hasattr(event, 'execution_status') and event.execution_status == ExecutionStatus.COMPLETED:
                    return getattr(event, 'result', None)
        return None
    
    def has_completed_event(self, address: str) -> bool:
        """Check if event at address completed successfully."""
        if address not in self.__events:
            return False
        event = self.__events[address]
        # Handle both dict and ExecutionEvent objects
        if isinstance(event, dict):
            return event.get('execution_status') == ExecutionStatus.COMPLETED.value
        else:
            return hasattr(event, 'execution_status') and event.execution_status == ExecutionStatus.COMPLETED


class EventBus(BaseModel):
    subscribers: list[Callable] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)  # ExecutionEvent at runtime
    queue: asyncio.Queue = Field(default_factory=asyncio.Queue)

    model_config = {"arbitrary_types_allowed": True}

    def subscribe(self, handler: Callable):
        self.subscribers.append(handler)

    async def publish(self, event: Union[ExecutionEvent, Any]):  # Accept ExecutionEvent or InterruptEvent
        self.events.append(event)

        for handler in self.subscribers:
            model_copy_fn = getattr(event, "model_copy", None)
            event_copy = (
                model_copy_fn()
                if model_copy_fn is not None
                else event
            )

            result = handler(event_copy)
            if asyncio.iscoroutine(result):
                await result
        
        # Yield control to allow other tasks (like stream generators) to process the event
        await asyncio.sleep(0)

    async def stream(self) -> AsyncGenerator[Union['ExecutionEvent', Any], None]:
        """
        Stream events from the queue.
        
        This generator yields execution events as they are published. It automatically
        terminates when an InterruptEvent is received, allowing for graceful shutdown
        of streaming endpoints (SSE, WebSocket, etc.).
        
        Yields:
            ExecutionEvent or InterruptEvent: Events from the queue
        
        Raises:
            asyncio.CancelledError: If the stream task is cancelled
        
        Example:
            ```python
            # In streaming API endpoint
            async def stream_events():
                async for event in state.event_bus.stream():
                    if isinstance(event, InterruptEvent):
                        # Interrupt received, stream will terminate
                        break
                    yield format_event(event)
            ```
        """
        from rh_agents.core.types import InterruptEvent
        
        try:
            while True:
                event = await self.queue.get()
                
                # Check for interrupt event - terminate stream
                if isinstance(event, InterruptEvent):
                    logging.info("🛑 Interrupt signal received, terminating event stream...")
                    yield event  # Yield the interrupt event so handlers can process it
                    break
                
                yield event
        except asyncio.CancelledError:
            logging.info("🛑 Event stream cancelled")
            raise



class ExecutionState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    # State recovery fields
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this execution state")
    replay_mode: ReplayMode = Field(default=ReplayMode.NORMAL, description="How to handle replay of already-executed events")
    resume_from_address: Optional[str] = Field(default=None, description="Address to resume execution from (replay all events after this)")
    _resume_point_reached: bool = False  # Internal flag: True after resume_from_address is cleared
    
    # Core state components
    storage: ExecutionStore = Field(default_factory=lambda: ExecutionStore())
    current_execution: Any = None  # ExecutionEvent at runtime
    history: HistorySet = Field(default_factory=HistorySet)
    execution_stack: list[str] = Field(default_factory=list, description="Stack tracking current execution path (agent/tool names)")
    
    # Runtime components (excluded from serialization)
    event_bus: EventBus = Field(default_factory=EventBus, exclude=True)
    state_backend: Any = Field(default=None, exclude=True)  # StateBackend at runtime
    artifact_backend: Any = Field(default=None, exclude=True)  # ArtifactBackend at runtime
    
    # Interrupt management (excluded from serialization)
    is_interrupted: bool = Field(default=False, exclude=True)
    interrupt_signal: Optional[InterruptSignal] = Field(default=None, exclude=True)
    interrupt_checker: Optional[InterruptChecker] = Field(default=None, exclude=True)
    active_generators: set[asyncio.Task] = Field(default_factory=set, exclude=True)
    
    def get_current_address(self, event_type: EventType) -> str:
        """Build address from execution stack + event type, e.g. 'doctrine_agent::tool_call'"""
        if not self.execution_stack:
            return event_type.value
        return "::" .join(self.execution_stack) + "::" + event_type.value
    
    def push_context(self, name: str):
        """Enter a new execution level (agent, tool, llm)"""
        self.execution_stack.append(name)
    
    def pop_context(self) -> str | None:
        """Exit current execution level"""
        if self.execution_stack:
            return self.execution_stack.pop()
        return None
    
    async def add_event(self, event: 'ExecutionEvent', status: ExecutionStatus):
        """Add event to history and conditionally publish to event bus."""
        event.address = self.get_current_address(event.actor.event_type)
        event.execution_status = status
        self.history.add(event)
        
        # Only publish if not skipped (for replay control)
        if not getattr(event, 'skip_republish', False):
            await self.event_bus.publish(event)
        
    def add_step_result(self, step_index: int, value: Any):
        """Store step result in execution storage with key 'step_{index}'"""
        self.storage.set(f"step::{step_index}", str(value))
        
    def get_steps_result(self, step_index: list[int]) -> list[Any]:
        """Retrieve step result from execution storage with key 'step_{index}'"""
        results = []
        for index in step_index:
            item = self.storage.get(f"step::{index}")
            if item is not None:
                results.append(item)
        return results
    
    def get_last_step_result(self) -> Any | None:
        """Retrieve last step result from execution storage with key 'step_{index}'"""
        step_keys = [key for key in self.storage.data.keys() if key.startswith("step::")]
        if not step_keys:
            return None
        last_key = max(step_keys, key=lambda k: int(k.split("::")[1]))
        return self.storage.get(last_key)
            
    def get_all_steps_results(self) -> dict[str, str]:
        """Retrieve all stored execution results"""
        return self.storage.data.copy()
    
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
    ) -> 'ParallelExecutionManager':
        """
        Create a parallel execution context for running independent events concurrently.
        
        This method returns a ParallelExecutionManager that can be used as an async
        context manager to execute multiple events in parallel with controlled concurrency.
        
        Args:
            max_workers: Maximum number of concurrent workers (default: 5)
            error_strategy: How to handle errors - ErrorStrategy.FAIL_SLOW (default) 
                          or ErrorStrategy.FAIL_FAST
            timeout: Optional timeout in seconds for the entire parallel group
            name: Optional human-readable name for the parallel group
            max_retries: Number of retry attempts for failed tasks (default: 0)
            retry_delay: Base delay between retries in seconds (default: 1.0)
            circuit_breaker_threshold: Failures before circuit opens (default: 5)
            circuit_breaker_timeout: Seconds before circuit half-opens (default: 60.0)
        
        Returns:
            ParallelExecutionManager instance ready to use as context manager
        
        Example:
            ```python
            # Basic parallel execution
            async with execution_state.parallel(max_workers=5) as p:
                p.add(event1(input1, context, state))
                p.add(event2(input2, context, state))
                p.add(event3(input3, context, state))
                results = await p.gather()
            
            # With error handling and retries
            async with execution_state.parallel(
                max_workers=3,
                error_strategy=ErrorStrategy.FAIL_FAST,
                timeout=30.0,
                name="Document Processing",
                max_retries=3,
                retry_delay=1.0
            ) as p:
                for doc in documents:
                    p.add(process_document(doc, context, state))
                results = await p.gather()
            ```
        """
        from rh_agents.core.parallel import ParallelExecutionManager, ErrorStrategy
        
        # Use default error strategy if not provided
        if error_strategy is None:
            error_strategy = ErrorStrategy.FAIL_SLOW
        
        return ParallelExecutionManager(
            execution_state=self,  # type: ignore[arg-type]
            max_workers=max_workers,
            error_strategy=error_strategy,
            timeout=timeout,
            name=name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_timeout=circuit_breaker_timeout
        )
    
    def should_skip_event(self, address: str) -> bool:
        """
        Check if event should be skipped during replay.
        
        Logic:
        - VALIDATION mode: Never skip (re-execute everything)
        - With resume_from_address: Skip until we reach that address, then never skip again
        - After resume point reached: Never skip (creating new timeline)
        - Normal replay: Skip if event completed successfully in history
        
        Args:
            address: Event address to check
            
        Returns:
            True if event should be skipped, False if it should execute
        """
        if self.replay_mode == ReplayMode.VALIDATION:
            return False  # Always execute for validation
        
        # If resume point was already reached, don't skip anything (new timeline)
        if self._resume_point_reached:
            return False
        
        # If resume_from_address is set, skip everything before it
        if self.resume_from_address:
            # Check if we've reached the resume point
            if address == self.resume_from_address:
                # Mark that we've reached it and DON'T skip this event
                self._resume_point_reached = True
                return False  # Execute this event (and continue executing after)
            
            # Check if current address is a PREFIX of the resume address
            # This means we need to execute this parent event to reach the nested resume point
            # Remove trailing ::xxx from address and check if resume starts with it
            addr_base = address.rsplit("::", 1)[0] if "::" in address else address
            if self.resume_from_address.startswith(addr_base + "::"):
                return False  # Don't skip - need to execute to reach nested resume point
            
            # Also check exact prefix match (for addresses without :: at the end)
            if self.resume_from_address.startswith(address + "::"):
                return False
            
            # Skip if we haven't reached resume point yet AND event exists in history
            return self.history.has_completed_event(address)
        
        # If we've reached the resume point, don't skip anymore (force re-execution)
        if self._resume_point_reached:
            return False
        
        # Normal replay: skip if already completed
        return self.history.has_completed_event(address)
    
    def to_snapshot(
        self,
        status: Optional["StateStatus"] = None,
        metadata: Optional["StateMetadata"] = None
    ) -> "StateSnapshot":
        """
        Create a snapshot of current execution state for persistence.
        
        This includes:
        - Core state (history, storage, stack)
        - Artifact references (artifacts stored separately)
        - Metadata and timestamps
        
        Runtime components (event_bus, backends) are excluded.
        
        Args:
            status: Execution status (default: RUNNING)
            metadata: Optional metadata to attach
            
        Returns:
            StateSnapshot ready for persistence
        """
        from datetime import datetime
        from rh_agents.core.state_recovery import StateSnapshot, StateStatus, StateMetadata
        from rh_agents.state_backends import compute_artifact_id
        
        # Serialize core state (exclude runtime components)
        state_dict = self.model_dump(
            exclude={'event_bus'}
        )
        
        # Extract artifact references and save artifacts separately
        artifact_refs = {}
        for key, artifact in self.storage.artifacts.items():
            artifact_id = compute_artifact_id(artifact)
            artifact_refs[key] = artifact_id
            
            # Save artifact to backend if available
            if self.artifact_backend:
                self.artifact_backend.save_artifact(artifact_id, artifact)
        
        # Create snapshot
        return StateSnapshot(
            state_id=self.state_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status=status or StateStatus.RUNNING,
            metadata=metadata or StateMetadata(),
            execution_state=state_dict,
            artifact_refs=artifact_refs
        )
    
    @classmethod
    def from_snapshot(
        cls,
        snapshot: "StateSnapshot",
        state_backend: Optional["StateBackend"] = None,
        artifact_backend: Optional["ArtifactBackend"] = None,
        event_bus: Optional[EventBus] = None,
        replay_mode: ReplayMode = ReplayMode.NORMAL,
        resume_from_address: Optional[str] = None
    ) -> "ExecutionState":
        """
        Restore ExecutionState from a snapshot.
        
        Reconstructs:
        - Core state from serialized data
        - Artifacts from artifact backend
        - Runtime components (event bus, backends)
        
        Note: Events in history are kept as dicts (not fully reconstructed)
        since we only need result/status data for replay, not the actor/handler.
        
        Args:
            snapshot: StateSnapshot to restore from
            state_backend: Backend for state persistence
            artifact_backend: Backend for artifact storage
            event_bus: Event bus instance (creates new if None)
            replay_mode: How to handle replay
            resume_from_address: Optional address to resume from
            
        Returns:
            Restored ExecutionState ready for execution
        """
        # Deserialize core state - events will be dicts, not ExecutionEvent objects
        # This is fine because HistorySet now handles both
        state = cls(**snapshot.execution_state)
        
        # Restore artifacts from backend
        if artifact_backend:
            for key, artifact_id in snapshot.artifact_refs.items():
                artifact = artifact_backend.load_artifact(artifact_id)
                if artifact:
                    state.storage.artifacts[key] = artifact
        
        # Reconstruct runtime components
        state.event_bus = event_bus or EventBus()
        state.state_backend = state_backend
        state.artifact_backend = artifact_backend
        state.replay_mode = replay_mode
        state.resume_from_address = resume_from_address
        
        return state
    
    def save_checkpoint(
        self,
        status: Optional["StateStatus"] = None,
        metadata: Optional["StateMetadata"] = None
    ) -> bool:
        """
        Save current state to backend as a checkpoint.
        
        Args:
            status: Execution status (default: RUNNING)
            metadata: Optional metadata to attach
            
        Returns:
            True if successful, False if no backend configured or save failed
        """
        if not self.state_backend:
            return False
        
        snapshot = self.to_snapshot(status, metadata)
        return self.state_backend.save_state(snapshot)
    
    @classmethod
    def load_from_state_id(
        cls,
        state_id: str,
        state_backend: "StateBackend",
        artifact_backend: Optional["ArtifactBackend"] = None,
        event_bus: Optional[EventBus] = None,
        replay_mode: ReplayMode = ReplayMode.NORMAL,
        resume_from_address: Optional[str] = None
    ) -> Optional["ExecutionState"]:
        """
        Load ExecutionState by its unique ID.
        
        Convenience method that loads snapshot and restores state.
        
        Args:
            state_id: Unique state identifier
            state_backend: Backend to load from
            artifact_backend: Backend for artifact loading
            event_bus: Event bus instance (creates new if None)
            replay_mode: How to handle replay
            resume_from_address: Optional address to resume from
            
        Returns:
            Restored ExecutionState if found, None otherwise
        """
        snapshot = state_backend.load_state(state_id)
        if not snapshot:
            return None
        
        return cls.from_snapshot(
            snapshot,
            state_backend,
            artifact_backend,
            event_bus,
            replay_mode,
            resume_from_address
        )
    
    # ===== Interrupt Management Methods =====
    
    def request_interrupt(
        self,
        reason: InterruptReason = InterruptReason.USER_CANCELLED,
        message: Optional[str] = None,
        triggered_by: Optional[str] = None,
        save_checkpoint: bool = True
    ) -> None:
        """
        Request interruption of the current execution (local control).
        
        This is the primary method for programmatic interrupt control. It sets
        the internal interrupt flag and creates an InterruptSignal with details.
        The next interrupt checkpoint will detect this and raise ExecutionInterrupted.
        
        Args:
            reason: Why execution is being interrupted
            message: Optional human-readable explanation
            triggered_by: Optional identifier of who/what triggered the interrupt
            save_checkpoint: Whether to save state before terminating (default: True)
        
        Example:
            ```python
            # In user-facing API or interrupt handler
            if should_cancel:
                state.request_interrupt(
                    reason=InterruptReason.USER_CANCELLED,
                    message="User pressed Cancel button",
                    triggered_by="web_ui_user_123"
                )
            ```
        """
        self.is_interrupted = True
        self.interrupt_signal = InterruptSignal(
            reason=reason,
            message=message,
            triggered_by=triggered_by,
            save_checkpoint=save_checkpoint
        )
    
    def set_interrupt_checker(self, checker: Optional[InterruptChecker]) -> None:
        """
        Set external interrupt checker for distributed control.
        
        The checker function is called at each interrupt checkpoint to poll
        external systems (Redis, database, REST API, etc.) for interrupt signals.
        Supports both sync/async functions returning bool or InterruptSignal.
        
        Args:
            checker: Function to check for external interrupt signal, or None to clear
        
        Example:
            ```python
            # Redis-based distributed interrupt
            import redis.asyncio as redis
            
            redis_client = redis.Redis(host='localhost', port=6379)
            
            async def check_redis():
                value = await redis_client.get(f"interrupt:{state.state_id}")
                if value == b"cancel":
                    return InterruptSignal(
                        reason=InterruptReason.USER_CANCELLED,
                        message="Cancelled from admin dashboard",
                        triggered_by="admin_user"
                    )
                return False
            
            state.set_interrupt_checker(check_redis)
            ```
        """
        self.interrupt_checker = checker
    
    async def check_interrupt(self) -> None:
        """
        Check for interrupt signal and raise ExecutionInterrupted if detected.
        
        This is called at strategic checkpoints during execution:
        - Before each agent/tool call
        - Before each LLM call
        - Before/after parallel execution batches
        - During long-running operations
        
        Checking logic:
        1. Check local flag first (fast path)
        2. If external checker is set, call it and handle return value:
           - bool True → create default interrupt signal
           - InterruptSignal → use directly
           - bool False / None → no interrupt
        3. If interrupted, publish InterruptEvent and raise ExecutionInterrupted
        
        Raises:
            ExecutionInterrupted: If interrupt is detected from local flag or external checker
        
        Example:
            ```python
            # Manual checkpoint in long operation
            async def process_large_dataset(data, context, state):
                for batch in data.batches():
                    await state.check_interrupt()  # Check before each batch
                    results.extend(process_batch(batch))
            ```
        """
        from rh_agents.core.exceptions import ExecutionInterrupted
        from rh_agents.core.types import InterruptEvent
        
        # Fast path: check local flag first
        if self.is_interrupted:
            signal = self.interrupt_signal or InterruptSignal(
                reason=InterruptReason.USER_CANCELLED,
                message="Execution interrupted"
            )
            
            # Publish interrupt event to event bus (for monitoring/logging)
            try:
                interrupt_event = InterruptEvent(signal=signal, state_id=self.state_id)
                await self.event_bus.publish(interrupt_event)
            except Exception as e:
                # If publishing fails, log but don't block interrupt
                logging.warning(f"Failed to publish interrupt event: {e}")
            
            raise ExecutionInterrupted(reason=signal.reason, message=signal.message)
        
        # Check external interrupt checker if configured
        if self.interrupt_checker is not None:
            result = self.interrupt_checker()
            
            # Handle async checker
            if inspect.iscoroutine(result):
                result = await result
            
            # Handle different return types
            if result is True:
                # bool True → create default signal
                signal = InterruptSignal(
                    reason=InterruptReason.USER_CANCELLED,
                    message="External interrupt signal detected"
                )
            elif isinstance(result, InterruptSignal):
                # InterruptSignal → use directly
                signal = result
            elif result is False or result is None:
                # No interrupt
                return
            else:
                # Unexpected return type
                logging.warning(f"Interrupt checker returned unexpected type: {type(result)}")
                return
            
            # Store signal and set flag
            self.is_interrupted = True
            self.interrupt_signal = signal
            
            # Publish interrupt event
            try:
                interrupt_event = InterruptEvent(signal=signal, state_id=self.state_id)
                await self.event_bus.publish(interrupt_event)
            except Exception as e:
                logging.warning(f"Failed to publish interrupt event: {e}")
            
            raise ExecutionInterrupted(reason=signal.reason, message=signal.message)
    
    # ===== Generator Registry Management =====
    
    def register_generator(self, generator_task: asyncio.Task) -> None:
        """
        Register an active event generator task for cleanup on interrupt.
        
        This allows the interrupt system to track and cancel all active generators
        (e.g., EventBus.stream(), SSE endpoints) when execution is interrupted.
        
        Args:
            generator_task: The asyncio Task running the generator
        
        Example:
            ```python
            # In streaming endpoint
            stream_task = asyncio.create_task(state.event_bus.stream())
            state.register_generator(stream_task)
            
            try:
                async for event in stream_task:
                    yield event
            finally:
                state.unregister_generator(stream_task)
            ```
        """
        self.active_generators.add(generator_task)
    
    def unregister_generator(self, generator_task: asyncio.Task) -> None:
        """
        Remove a generator task from the active registry.
        
        Should be called when a generator completes normally or is cancelled.
        
        Args:
            generator_task: The asyncio Task to remove from tracking
        """
        self.active_generators.discard(generator_task)
    
    async def kill_generators(self) -> None:
        """
        Cancel all active event generators immediately.
        
        This is called automatically when an interrupt is triggered to ensure
        all streaming endpoints (SSE, WebSocket, etc.) are cleanly terminated.
        
        The method:
        1. Cancels all registered generator tasks
        2. Waits for them to complete (with cancellation)
        3. Clears the registry
        
        Example:
            ```python
            # Automatically called by check_interrupt(), but can be called manually:
            state.request_interrupt(reason=InterruptReason.USER_CANCELLED)
            await state.kill_generators()  # Clean up all streams
            ```
        """
        if not self.active_generators:
            return
        
        # Cancel all active generators
        for gen_task in list(self.active_generators):
            if not gen_task.done():
                gen_task.cancel()
        
        # Wait for all to complete (ignoring exceptions from cancellation)
        if self.active_generators:
            await asyncio.gather(*self.active_generators, return_exceptions=True)
        
        # Clear the registry
        self.active_generators.clear()
