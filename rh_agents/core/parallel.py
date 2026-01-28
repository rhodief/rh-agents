"""
Parallel Execution Support for RH-Agents Framework

This module provides data models, enums, and the core execution manager
for parallel event execution with concurrency control.
"""

from __future__ import annotations
import asyncio
import uuid
from enum import Enum
from time import time
from typing import Optional, Any, Awaitable, AsyncGenerator, Union, TYPE_CHECKING
from collections.abc import Callable
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from rh_agents.core.execution import ExecutionState
    from rh_agents.core.events import ExecutionResult


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class GroupStatus(str, Enum):
    """Status of a parallel execution group."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorStrategy(str, Enum):
    """Strategy for handling errors in parallel execution."""
    
    FAIL_SLOW = "fail_slow"  # Collect all results, report errors at end
    FAIL_FAST = "fail_fast"  # Cancel remaining tasks on first error


class ParallelDisplayMode(str, Enum):
    """Display mode for parallel event visualization."""
    
    REALTIME = "realtime"  # Interleaved real-time output
    PROGRESS = "progress"  # Grouped output with progress bar


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


class ParallelEventGroup(BaseModel):
    """
    Metadata for a group of events executing in parallel.
    
    This model tracks the lifecycle and statistics of a parallel execution group,
    including completion status, timing, and error information.
    """
    
    group_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this parallel group"
    )
    
    name: Optional[str] = Field(
        default=None,
        description="Optional human-readable name for the group"
    )
    
    total: int = Field(
        default=0,
        description="Total number of events in the group"
    )
    
    completed: int = Field(
        default=0,
        description="Number of events that completed successfully"
    )
    
    failed: int = Field(
        default=0,
        description="Number of events that failed"
    )
    
    status: GroupStatus = Field(
        default=GroupStatus.PENDING,
        description="Current status of the group"
    )
    
    start_time: Optional[float] = Field(
        default=None,
        description="Timestamp when group execution started"
    )
    
    end_time: Optional[float] = Field(
        default=None,
        description="Timestamp when group execution completed"
    )
    
    error_strategy: ErrorStrategy = Field(
        default=ErrorStrategy.FAIL_SLOW,
        description="Strategy for handling errors in this group"
    )
    
    max_workers: int = Field(
        default=5,
        description="Maximum number of concurrent workers"
    )
    
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds for the entire group (None = no timeout)"
    )
    
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages from failed events"
    )
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total execution time if both start and end times are set."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100.0
    
    def increment_completed(self):
        """Increment the completed counter."""
        self.completed += 1
    
    def increment_failed(self):
        """Increment the failed counter."""
        self.failed += 1
    
    def add_error(self, error_message: str):
        """Add an error message to the group."""
        self.errors.append(error_message)
        self.increment_failed()
    
    def mark_started(self):
        """Mark the group as started."""
        self.status = GroupStatus.RUNNING
        self.start_time = time()
    
    def mark_completed(self):
        """Mark the group as completed."""
        self.status = GroupStatus.COMPLETED if self.failed == 0 else GroupStatus.FAILED
        self.end_time = time()
    
    def mark_cancelled(self):
        """Mark the group as cancelled."""
        self.status = GroupStatus.CANCELLED
        self.end_time = time()


class ParallelGroupTracker(BaseModel):
    """
    Tracks state for parallel group visualization in EventPrinter.
    
    This model is used by ParallelEventPrinter to maintain state
    about ongoing parallel execution for progress display.
    """
    
    group_id: str = Field(
        description="ID of the parallel group being tracked"
    )
    
    name: Optional[str] = Field(
        default=None,
        description="Name of the parallel group"
    )
    
    total: int = Field(
        default=0,
        description="Total events in group"
    )
    
    completed: int = Field(
        default=0,
        description="Events completed so far"
    )
    
    failed: int = Field(
        default=0,
        description="Events failed so far"
    )
    
    started: int = Field(
        default=0,
        description="Events started so far"
    )
    
    start_time: Optional[float] = Field(
        default=None,
        description="When the group started"
    )
    
    last_update_time: Optional[float] = Field(
        default=None,
        description="Last time this tracker was updated"
    )
    
    progress_line_number: Optional[int] = Field(
        default=None,
        description="Terminal line number where progress bar is displayed"
    )
    
    @property
    def is_complete(self) -> bool:
        """Check if all events have completed (success or failure)."""
        return (self.completed + self.failed) >= self.total
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return ((self.completed + self.failed) / self.total) * 100.0
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time since group started."""
        if self.start_time is None:
            return None
        return time() - self.start_time
    
    def update_from_event(self, event_status: str):
        """
        Update tracker based on event status.
        
        Args:
            event_status: Status from ExecutionStatus enum
        """
        self.last_update_time = time()
        
        # Map event status to tracker updates
        if event_status.lower() == "started":
            self.started += 1
        elif event_status.lower() == "completed":
            self.completed += 1
        elif event_status.lower() == "failed":
            self.failed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Execution Manager
# ═══════════════════════════════════════════════════════════════════════════════


class ParallelExecutionManager:
    """
    Manages parallel execution of events with concurrency control.
    
    This class orchestrates parallel execution of independent events using
    asyncio semaphores for concurrency limiting. It supports multiple result
    collection strategies and configurable error handling.
    
    Usage:
        ```python
        async with execution_state.parallel(max_workers=5) as parallel:
            parallel.add(event1(...))
            parallel.add(event2(...))
            results = await parallel.gather()
        ```
    
    Attributes:
        execution_state: Reference to ExecutionState for context
        semaphore: Asyncio semaphore for concurrency control
        error_strategy: How to handle errors (FAIL_SLOW or FAIL_FAST)
        timeout: Optional timeout for the entire group
        group: Metadata about the parallel group
        tasks: List of asyncio tasks being executed
        results: Collected results after execution
    """
    
    def __init__(
        self,
        execution_state: "ExecutionState",
        max_workers: int = 5,
        error_strategy: ErrorStrategy = ErrorStrategy.FAIL_SLOW,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        """
        Initialize ParallelExecutionManager.
        
        Args:
            execution_state: ExecutionState instance for event publishing
            max_workers: Maximum number of concurrent workers (default: 5)
            error_strategy: How to handle errors (default: FAIL_SLOW)
            timeout: Optional timeout in seconds for the entire group
            name: Optional human-readable name for the group
            max_retries: Number of retry attempts for failed tasks (default: 0)
            retry_delay: Base delay between retries in seconds (default: 1.0)
            circuit_breaker_threshold: Failures before circuit opens (default: 5)
            circuit_breaker_timeout: Seconds before circuit half-opens (default: 60.0)
        """
        self.execution_state = execution_state
        self.semaphore = asyncio.Semaphore(max_workers)
        self.error_strategy = error_strategy
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize group metadata
        self.group = ParallelEventGroup(
            name=name,
            max_workers=max_workers,
            error_strategy=error_strategy,
            timeout=timeout
        )
        
        # Task tracking
        self.coroutines: list[tuple[Union[Awaitable, Callable[[], Awaitable]], Optional[str]]] = []
        self.tasks: list[asyncio.Task] = []
        self.results: list["ExecutionResult"] = []
        self._gathered = False
        
        # Circuit breaker state
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._circuit_failures = 0
        self._circuit_opened_at: Optional[float] = None
        self._circuit_lock = asyncio.Lock()
        
        # Save the base execution stack to isolate parallel tasks
        # Each parallel task will reset to this base stack before executing
        self._base_execution_stack: list[str] = []
    
    async def __aenter__(self):
        """Enter async context manager - mark group as started."""
        self.group.mark_started()
        # Capture the current execution stack as the base for all parallel tasks
        self._base_execution_stack = self.execution_state.execution_stack.copy()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context manager - auto-gather if not done.
        
        This ensures all tasks complete even if user doesn't call gather()
        explicitly, providing automatic resource cleanup.
        """
        if self.coroutines and not self._gathered:
            await self.gather()
        
        # Mark group as completed/cancelled
        if exc_type is not None:
            self.group.mark_cancelled()
        elif not self._gathered:
            self.group.mark_completed()
    
    def add(self, coro: Union[Awaitable, Callable[[], Awaitable]], name: Optional[str] = None):
        """
        Add a coroutine or coroutine-callable to the parallel execution group.
        
        Args:
            coro: Coroutine to execute, or a callable that returns a coroutine.
                  For retry support, pass a callable (function) that returns a coroutine.
            name: Optional name for this task (used in dict mode)
        
        Raises:
            RuntimeError: If called after gather() has been called
        
        Example:
            ```python
            async with state.parallel() as p:
                # Direct coroutine (no retry support)
                p.add(process_doc(doc1))
                
                # Callable for retry support
                p.add(lambda: process_doc(doc2), name="doc2")
            ```
        """
        if self._gathered:
            raise RuntimeError(
                "Cannot add tasks after gather() has been called. "
                "Create a new parallel execution context."
            )
        
        self.coroutines.append((coro, name))
        self.group.total += 1
    
    async def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker allows execution.
        
        Returns:
            True if execution allowed, False if circuit is open
        """
        async with self._circuit_lock:
            # Check if circuit is open
            if self._circuit_opened_at is not None:
                elapsed = time() - self._circuit_opened_at
                
                # Check if timeout has passed (half-open state)
                if elapsed >= self.circuit_breaker_timeout:
                    # Reset circuit to closed state
                    self._circuit_opened_at = None
                    self._circuit_failures = 0
                    return True
                
                # Circuit still open
                return False
            
            return True
    
    async def _record_circuit_failure(self):
        """Record a failure for circuit breaker tracking."""
        async with self._circuit_lock:
            self._circuit_failures += 1
            
            # Open circuit if threshold exceeded
            if self._circuit_failures >= self.circuit_breaker_threshold:
                self._circuit_opened_at = time()
    
    async def _record_circuit_success(self):
        """Record a success for circuit breaker tracking."""
        async with self._circuit_lock:
            # On success, reduce failure count
            if self._circuit_failures > 0:
                self._circuit_failures = max(0, self._circuit_failures - 1)
    
    async def _execute_with_semaphore(
        self,
        coro_or_callable: Union[Awaitable, Callable[[], Awaitable]],
        index: int,
        name: Optional[str] = None
    ) -> "ExecutionResult":
        """
        Execute a single coroutine with semaphore-based concurrency control.
        
        This wrapper ensures only max_workers coroutines execute concurrently.
        It also handles error wrapping, retry logic, and circuit breaker.
        
        Args:
            coro_or_callable: Coroutine to execute, or callable that returns a coroutine
            index: Index of this task in the parallel group
            name: Optional name for this task
        
        Returns:
            ExecutionResult with the result or error
        """
        from rh_agents.core.events import ExecutionResult
        import inspect
        
        async with self.semaphore:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                error_msg = f"Circuit breaker open - task {index} ({name or 'unnamed'}) rejected"
                self.group.add_error(error_msg)
                
                if self.error_strategy == ErrorStrategy.FAIL_FAST:
                    raise RuntimeError(error_msg)
                
                return ExecutionResult(
                    result=None,
                    ok=False,
                    erro_message=error_msg,
                    execution_time=None
                )
            
            # Reset execution stack to the base state before this parallel block
            # This isolates each parallel task's context from other concurrent tasks
            self.execution_state.execution_stack = self._base_execution_stack.copy()
            
            try:
                # Check if it's a callable or a coroutine
                is_callable = callable(coro_or_callable) and not inspect.iscoroutine(coro_or_callable)
                
                # Retry logic
                last_exception = None
                for attempt in range(self.max_retries + 1):
                    try:
                        # Get the coroutine for this attempt
                        if is_callable:
                            # Type narrowing: if is_callable, it's a Callable
                            assert callable(coro_or_callable)
                            coro = coro_or_callable()  # type: ignore[operator]
                        else:
                            # For direct coroutines, we can only execute once
                            if attempt > 0:
                                raise RuntimeError(
                                    "Cannot retry a direct coroutine. Pass a callable (lambda/function) "
                                    "that returns a coroutine to enable retries."
                                )
                            # Type narrowing: if not callable, it's an Awaitable
                            coro = coro_or_callable  # type: ignore[assignment]
                        
                        result = await coro  # type: ignore[misc]
                        
                        # Success - record for circuit breaker
                        await self._record_circuit_success()
                        
                        # Wrap result in ExecutionResult if not already
                        if isinstance(result, ExecutionResult):
                            return result
                        else:
                            return ExecutionResult(
                                result=result,
                                ok=True,
                                execution_time=None
                            )
                            
                    except Exception as e:
                        last_exception = e
                        
                        # Record failure for circuit breaker
                        await self._record_circuit_failure()
                        
                        # Check if we should retry
                        if attempt < self.max_retries:
                            # Calculate exponential backoff delay
                            delay = self.retry_delay * (2 ** attempt)
                            await asyncio.sleep(delay)
                            continue
                        
                        # No more retries - handle error
                        error_msg = f"Task {index} ({name or 'unnamed'}): {str(e)}"
                        if self.max_retries > 0:
                            error_msg += f" (failed after {self.max_retries + 1} attempts)"
                        
                        self.group.add_error(error_msg)
                        
                        # For fail-fast, re-raise immediately
                        if self.error_strategy == ErrorStrategy.FAIL_FAST:
                            raise
                        
                        # For fail-slow, wrap error in ExecutionResult
                        return ExecutionResult(
                            result=None,
                            ok=False,
                            erro_message=str(e),
                            execution_time=None
                        )
                
                # Should never reach here, but handle it
                return ExecutionResult(
                    result=None,
                    ok=False,
                    erro_message=str(last_exception) if last_exception else "Unknown error",
                    execution_time=None
                )
            finally:
                # Restore execution stack to base state after task completes
                # This ensures the stack is clean for the next parallel task
                self.execution_state.execution_stack = self._base_execution_stack.copy()
    
    async def gather(self) -> list["ExecutionResult"]:
        """
        Execute all tasks in parallel and return results in order.
        
        Results are returned in the same order as tasks were added,
        regardless of completion order. This method respects the
        configured error_strategy and monitors for interrupt signals.
        
        Returns:
            List of ExecutionResult objects, one per task, in order.
            Failed tasks have ok=False and erro_message set.
        
        Raises:
            asyncio.TimeoutError: If group timeout exceeded
            Exception: If error_strategy is FAIL_FAST and any task fails
            ExecutionInterrupted: If execution is interrupted during parallel execution
        
        Example:
            ```python
            async with state.parallel(max_workers=5) as p:
                p.add(process_doc(doc1))
                p.add(process_doc(doc2))
                results = await p.gather()
            
            for result in results:
                if result.ok:
                    print(f"Success: {result.result}")
                else:
                    print(f"Failed: {result.erro_message}")
            ```
        """
        if self._gathered:
            return self.results
        
        self._gathered = True
        
        # INTERRUPT CHECK: Before starting parallel execution
        await self.execution_state.check_interrupt()
        
        # Create tasks with semaphore wrapper
        self.tasks = [
            asyncio.create_task(
                self._execute_with_semaphore(coro, idx, name)
            )
            for idx, (coro, name) in enumerate(self.coroutines)
        ]
        
        # Create interrupt monitor task
        async def interrupt_monitor():
            """Monitor for interrupt signals and cancel all tasks if triggered."""
            while not all(task.done() for task in self.tasks):
                try:
                    # Check for interrupt periodically
                    await self.execution_state.check_interrupt()
                    await asyncio.sleep(0.1)  # Check every 100ms
                except Exception:
                    # Interrupt detected - cancel all tasks
                    for task in self.tasks:
                        if not task.done():
                            task.cancel()
                    raise
        
        monitor_task = asyncio.create_task(interrupt_monitor())
        
        try:
            # Execute with optional timeout
            if self.timeout:
                self.results = await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=False),
                    timeout=self.timeout
                )
            else:
                self.results = await asyncio.gather(
                    *self.tasks,
                    return_exceptions=False
                )
            
            # Mark group as completed
            self.group.mark_completed()
            
            # INTERRUPT CHECK: After parallel execution completes
            await self.execution_state.check_interrupt()
            
            return self.results
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            self.group.mark_cancelled()
            self.group.add_error(f"Group timed out after {self.timeout}s")
            raise
            
        except Exception as e:
            # For FAIL_FAST, cancel remaining tasks
            if self.error_strategy == ErrorStrategy.FAIL_FAST:
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                
                self.group.mark_cancelled()
            
            raise
        
        finally:
            # Always cancel the monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    async def stream(self) -> AsyncGenerator["ExecutionResult", None]:
        """
        Execute tasks and yield results as they complete (streaming mode).
        
        Unlike gather(), this method yields results immediately as each task
        completes, allowing processing to begin before all tasks finish.
        Results are yielded in completion order, not submission order.
        
        Yields:
            ExecutionResult objects as tasks complete
        
        Raises:
            asyncio.TimeoutError: If group timeout exceeded
            Exception: If error_strategy is FAIL_FAST and any task fails
        
        Example:
            ```python
            async with state.parallel(max_workers=5) as p:
                for doc in documents:
                    p.add(process_doc(doc))
                
                async for result in p.stream():
                    if result.ok:
                        print(f"Processed: {result.result}")
                        # Can start processing immediately
                        await save_result(result.result)
            ```
        """
        if self._gathered:
            # If already gathered, yield from cached results
            for result in self.results:
                yield result
            return
        
        self._gathered = True
        
        # Create tasks with semaphore wrapper
        self.tasks = [
            asyncio.create_task(
                self._execute_with_semaphore(coro, idx, name)
            )
            for idx, (coro, name) in enumerate(self.coroutines)
        ]
        
        try:
            # Use as_completed to yield results as they finish
            if self.timeout:
                # Wrap with timeout
                done, pending = await asyncio.wait(
                    self.tasks,
                    timeout=self.timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Check if timed out
                if pending:
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                    
                    self.group.mark_cancelled()
                    self.group.add_error(f"Group timed out after {self.timeout}s")
                    raise asyncio.TimeoutError(f"Parallel group timed out after {self.timeout}s")
                
                # Yield completed results
                for task in done:
                    result = await task
                    self.results.append(result)
                    yield result
            else:
                # No timeout - use as_completed
                for coro in asyncio.as_completed(self.tasks):
                    result = await coro
                    self.results.append(result)
                    yield result
            
            # Mark group as completed
            self.group.mark_completed()
            
        except asyncio.TimeoutError:
            raise
            
        except Exception as e:
            # For FAIL_FAST, cancel remaining tasks
            if self.error_strategy == ErrorStrategy.FAIL_FAST:
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                
                self.group.mark_cancelled()
            
            raise
    
    async def gather_dict(self) -> dict[str, "ExecutionResult"]:
        """
        Execute tasks and return named results as a dictionary.
        
        Tasks must be added with names using add(coro, name="task_name").
        Unnamed tasks will be accessible by their numeric index as a string.
        
        Returns:
            Dictionary mapping task names to ExecutionResult objects
        
        Raises:
            asyncio.TimeoutError: If group timeout exceeded
            Exception: If error_strategy is FAIL_FAST and any task fails
        
        Example:
            ```python
            async with state.parallel(max_workers=3) as p:
                p.add(process_doc(doc1), name="doc1")
                p.add(process_doc(doc2), name="doc2")
                p.add(process_doc(doc3), name="doc3")
                
                results = await p.gather_dict()
                
                if results["doc1"].ok:
                    print(f"Doc 1: {results['doc1'].result}")
                if results["doc2"].ok:
                    print(f"Doc 2: {results['doc2'].result}")
            ```
        """
        # First gather all results
        results_list = await self.gather()
        
        # Build dictionary with names or indices
        results_dict = {}
        for idx, (result, (coro, name)) in enumerate(zip(results_list, self.coroutines)):
            # Use provided name or fall back to index
            key = name if name is not None else str(idx)
            results_dict[key] = result
        
        return results_dict
