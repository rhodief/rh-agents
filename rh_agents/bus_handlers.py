

# ═══════════════════════════════════════════════════════════════════════════════
# Beautiful Event Printer & SSE Streamer
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
from collections.abc import Callable
from typing import AsyncGenerator, Optional, Any, TYPE_CHECKING
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus

if TYPE_CHECKING:
    from rh_agents.core.parallel import ParallelGroupTracker


class EventPrinter:
    """Pretty printer for execution events with colors and formatting."""
    
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # Status colors and symbols
    STATUS_CONFIG = {
        ExecutionStatus.STARTED: ("▶", CYAN, "STARTED"),
        ExecutionStatus.COMPLETED: ("✔", GREEN, "COMPLETED"),
        ExecutionStatus.FAILED: ("✖", RED, "FAILED"),
        ExecutionStatus.AWAITING: ("⏳", YELLOW, "AWAITING"),
        ExecutionStatus.HUMAN_INTERVENTION: ("👤", MAGENTA, "HUMAN"),
        ExecutionStatus.RECOVERED: ("♻", YELLOW, "RECOVERED"),
    }
    
    # Event type icons
    EVENT_ICONS = {
        EventType.AGENT_CALL: "🤖",
        EventType.TOOL_CALL: "🔧",
        EventType.LLM_CALL: "🧠",
    }
    
    def __init__(self, show_timestamp: bool = True, show_address: bool = True):
        self.show_timestamp = show_timestamp
        self.show_address = show_address
        self.indent_cache: dict[str, int] = {}
        # Statistics tracking
        self.total_events = 0
        self.completed_events = 0
        self.failed_events = 0
        self.started_events = 0
        self.recovered_events = 0
        self.total_execution_time = 0.0
        self.events_by_type: dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_indent_level(self, address: str) -> int:
        """Calculate indentation based on address depth."""
        if not address:
            return 0
        return address.count("::")
    
    def _format_time(self, execution_time: float | None) -> str:
        """Format execution time nicely."""
        if execution_time is None:
            return ""
        if execution_time < 0.001:
            return f"{execution_time * 1_000_000:.0f}μs"
        elif execution_time < 1:
            return f"{execution_time * 1000:.1f}ms"
        else:
            return f"{execution_time:.2f}s"
    
    def _truncate(self, text: str, max_len: int = 50) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def print_event(self, event: ExecutionEvent):
        """Print a beautifully formatted event."""
        # Track statistics
        self.total_events += 1
        status = event.execution_status
        
        if status == ExecutionStatus.STARTED:
            self.started_events += 1
            self.cache_misses += 1
        elif status == ExecutionStatus.COMPLETED:
            self.completed_events += 1
            if event.execution_time:
                self.total_execution_time += event.execution_time
        elif status == ExecutionStatus.FAILED:
            self.failed_events += 1
        elif status == ExecutionStatus.RECOVERED:
            self.recovered_events += 1
            self.cache_hits += 1
        
        # Track by event type
        event_type = event.actor.event_type.value if hasattr(event.actor.event_type, 'value') else str(event.actor.event_type)
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1
        
        symbol, color, status_text = self.STATUS_CONFIG.get(
            status, ("?", self.WHITE, "UNKNOWN")
        )
        
        event_icon = self.EVENT_ICONS.get(event.actor.event_type, "📌")
        indent_level = self._get_indent_level(event.address)
        indent = "  │ " * indent_level
        
        # Build the output
        lines = []
        
        # Main event line
        actor_name = event.actor.name
        time_str = self._format_time(event.execution_time)
        time_display = f" {self.GRAY}({time_str}){self.RESET}" if time_str else ""
        
        main_line = (
            f"{self.GRAY}{indent}{self.RESET}"
            f"{color}{self.BOLD}{symbol}{self.RESET} "
            f"{event_icon} "
            f"{self.BOLD}{actor_name}{self.RESET} "
            f"{color}[{status_text}]{self.RESET}"
            f"{time_display}"
        )
        lines.append(main_line)
        
        # Address line (if enabled and has content)
        if self.show_address and event.address:
            address_line = (
                f"{self.GRAY}{indent}  ├─ 📍 {event.address}{self.RESET}"
            )
            lines.append(address_line)
        
        # Timestamp line (if enabled)
        if self.show_timestamp:
            timestamp = event.datetime[:19].replace("T", " ")  # Trim to readable format
            time_line = (
                f"{self.GRAY}{indent}  ├─ 🕐 {timestamp}{self.RESET}"
            )
            lines.append(time_line)
        
        # Detail information (if available)
        if event.detail:
            detail_preview = self._truncate(event.detail.replace('\n', ' '), 100)
            detail_icon = "📥" if status == ExecutionStatus.STARTED else "📤"
            
            # Special icon for cached results
            if event.from_cache and status == ExecutionStatus.RECOVERED:
                detail_icon = "💾"
                detail_preview = f"{self.YELLOW}[FROM CACHE]{self.RESET} {detail_preview}"
            
            detail_line = (
                f"{self.GRAY}{indent}  ├─ {detail_icon} {detail_preview}{self.RESET}"
            )
            lines.append(detail_line)
        
        # Cache message (if recovered from cache)
        if status == ExecutionStatus.RECOVERED and event.message:
            cache_msg = self._truncate(event.message, 80)
            cache_line = (
                f"{self.GRAY}{indent}  ├─ {self.RESET}"
                f"{self.YELLOW}✨ {cache_msg}{self.RESET}"
            )
            lines.append(cache_line)
        
        # Error message (if failed)
        if status == ExecutionStatus.FAILED and event.message:
            error_msg = self._truncate(event.message, 80)
            error_line = (
                f"{self.GRAY}{indent}  {self.RESET}"
                f"{self.RED}└─ ⚠️  {error_msg}{self.RESET}"
            )
            lines.append(error_line)
        else:
            # Closing line
            lines.append(f"{self.GRAY}{indent}  └{'─' * 40}{self.RESET}")
        
        # Print all lines
        print("\n".join(lines))
    
    def print_summary(self):
        """Print execution summary statistics."""
        print(f"\n{self.BOLD}{self.CYAN}{'═' * 70}{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}{'📊 EXECUTION SUMMARY':^70}{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}{'═' * 70}{self.RESET}\n")
        
        # Total events
        print(f"{self.BOLD}Total Events:{self.RESET} {self.WHITE}{self.total_events}{self.RESET}")
        print(f"  {self.CYAN}├─{self.RESET} Started: {self.CYAN}{self.started_events}{self.RESET}")
        print(f"  {self.GREEN}├─{self.RESET} Completed: {self.GREEN}{self.completed_events}{self.RESET}")
        print(f"  {self.YELLOW}├─{self.RESET} Recovered: {self.YELLOW}{self.recovered_events}{self.RESET}")
        print(f"  {self.RED}└─{self.RESET} Failed: {self.RED}{self.failed_events}{self.RESET}\n")
        
        # Success rate
        if self.started_events > 0:
            success_rate = (self.completed_events / self.started_events) * 100
            rate_color = self.GREEN if success_rate >= 80 else (self.YELLOW if success_rate >= 50 else self.RED)
            print(f"{self.BOLD}Success Rate:{self.RESET} {rate_color}{success_rate:.1f}%{self.RESET}\n")
        
        # Cache statistics
        if self.cache_hits > 0 or self.cache_misses > 0:
            total_cacheable = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cacheable) * 100 if total_cacheable > 0 else 0
            cache_color = self.GREEN if cache_hit_rate >= 50 else (self.YELLOW if cache_hit_rate >= 25 else self.CYAN)
            
            print(f"{self.BOLD}Cache Performance:{self.RESET}")
            print(f"  {self.GREEN}├─{self.RESET} Hits: {self.GREEN}{self.cache_hits}{self.RESET}")
            print(f"  {self.CYAN}├─{self.RESET} Misses: {self.CYAN}{self.cache_misses}{self.RESET}")
            print(f"  {cache_color}└─{self.RESET} Hit Rate: {cache_color}{cache_hit_rate:.1f}%{self.RESET}\n")
        
        # Total execution time
        time_str = self._format_time(self.total_execution_time)
        print(f"{self.BOLD}Total Execution Time:{self.RESET} {self.MAGENTA}{time_str}{self.RESET}\n")
        
        # Events by type
        if self.events_by_type:
            print(f"{self.BOLD}Events by Type:{self.RESET}")
            for event_type, count in sorted(self.events_by_type.items(), key=lambda x: x[1], reverse=True):
                # Find matching icon by checking if event_type matches any EventType value
                icon = "📌"
                for et, et_icon in self.EVENT_ICONS.items():
                    if (hasattr(et, 'value') and et.value == event_type) or str(et) == event_type:
                        icon = et_icon
                        break
                print(f"  {icon} {event_type}: {self.BLUE}{count}{self.RESET}")
        
        print(f"\n{self.BOLD}{self.CYAN}{'═' * 70}{self.RESET}\n")
    
    def __call__(self, event: ExecutionEvent):
        """Allow using the printer as a callback."""
        self.print_event(event)


def create_event_handler(printer: EventPrinter | None = None) -> Callable:
    """Factory to create an event handler with optional custom printer."""
    if printer is None:
        printer = EventPrinter()
    return printer


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Event Printer - Phase 6
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelEventPrinter(EventPrinter):
    """Enhanced event printer with parallel execution support."""
    
    def __init__(
        self,
        show_timestamp: bool = True,
        show_address: bool = True,
        parallel_mode: Optional[str] = "realtime"
    ):
        """
        Initialize parallel event printer.
        
        Args:
            show_timestamp: Show timestamp for each event
            show_address: Show event address
            parallel_mode: Display mode - "realtime" (interleaved) or "progress" (bars)
        """
        super().__init__(show_timestamp=show_timestamp, show_address=show_address)
        
        # Import here to avoid circular dependency
        from rh_agents.core.parallel import ParallelDisplayMode, ParallelGroupTracker
        
        self.parallel_mode = parallel_mode or "realtime"
        self.parallel_groups: dict[str, ParallelGroupTracker] = {}
        self.ParallelGroupTracker = ParallelGroupTracker
        self.ParallelDisplayMode = ParallelDisplayMode
    
    def print_event(self, event: ExecutionEvent):
        """Print event with parallel group awareness."""
        # Check if this is a parallel event
        if event.is_parallel and event.group_id:
            self._handle_parallel_event(event)
        else:
            # Regular event - use parent implementation
            super().print_event(event)
    
    def _handle_parallel_event(self, event: ExecutionEvent):
        """Handle parallel execution events."""
        group_id = event.group_id
        
        # Type narrowing: group_id should not be None here due to earlier check
        if group_id is None:
            return
        
        # Initialize group tracker if needed
        if group_id not in self.parallel_groups:
            self._initialize_group(event)
        
        tracker = self.parallel_groups[group_id]
        
        # Update tracker based on event status
        status = event.execution_status
        
        if status == ExecutionStatus.STARTED:
            tracker.started += 1
            tracker.last_update_time = asyncio.get_event_loop().time()
        elif status == ExecutionStatus.COMPLETED:
            tracker.completed += 1
            tracker.last_update_time = asyncio.get_event_loop().time()
        elif status == ExecutionStatus.FAILED:
            tracker.failed += 1
            tracker.last_update_time = asyncio.get_event_loop().time()
        
        # Display based on mode
        if self.parallel_mode == "realtime":
            self._print_realtime_event(event, tracker)
        elif self.parallel_mode == "progress":
            self._print_progress_mode(event, tracker)
        else:
            # Fallback to realtime
            self._print_realtime_event(event, tracker)
        
        # Check if group is complete
        if tracker.is_complete:
            self._print_group_summary(tracker)
    
    def _initialize_group(self, event: ExecutionEvent):
        """Initialize tracking for a new parallel group."""
        group_id = event.group_id
        
        # Type narrowing: group_id should not be None when this is called
        if group_id is None:
            return
        
        # Create tracker - we'll set total later when we know it
        tracker = self.ParallelGroupTracker(
            group_id=group_id,
            name=event.detail or f"Group {group_id}",
            total=0,  # Will be updated as events arrive
            started=0,
            completed=0,
            failed=0,
            start_time=asyncio.get_event_loop().time(),
            last_update_time=asyncio.get_event_loop().time()
        )
        
        self.parallel_groups[group_id] = tracker
        
        # Print group start header (different for each mode)
        if self.parallel_mode == "progress":
            # For progress mode, just print a simple header
            print(f"\n{self.BOLD}{self.CYAN}{'═' * 70}{self.RESET}")
            print(f"{self.CYAN}Starting: {self.BOLD}{tracker.name}{self.RESET}")
            print(f"{self.CYAN}{'═' * 70}{self.RESET}\n")
        else:
            # Realtime mode - original header
            print(f"\n{self.BOLD}{self.CYAN}{'═' * 70}{self.RESET}")
            print(f"{self.CYAN}▶ Parallel Group Started: {self.BOLD}{tracker.name}{self.RESET}")
            print(f"{self.CYAN}{'═' * 70}{self.RESET}\n")
    
    def _print_realtime_event(self, event: ExecutionEvent, tracker: "ParallelGroupTracker"):
        """Print event in real-time mode (interleaved output)."""
        status = event.execution_status
        symbol, color, status_text = self.STATUS_CONFIG.get(
            status, ("?", self.WHITE, "UNKNOWN")
        )
        
        # Event icon
        event_icon = self.EVENT_ICONS.get(event.actor.event_type, "📌")
        
        # Indent for parallel events
        indent = "  │ "
        
        # Format actor name with index
        actor_name = event.actor.name
        if event.parallel_index is not None:
            actor_name = f"{actor_name} #{event.parallel_index}"
        
        # Time formatting
        time_str = self._format_time(event.execution_time)
        time_display = f" {self.GRAY}({time_str}){self.RESET}" if time_str else ""
        
        # Build output
        main_line = (
            f"{self.GRAY}{indent}{self.RESET}"
            f"{color}{self.BOLD}{symbol}{self.RESET} "
            f"{event_icon} "
            f"{self.BOLD}{actor_name}{self.RESET} "
            f"{color}[{status_text}]{self.RESET}"
            f"{time_display}"
        )
        
        print(main_line)
        
        # Show detail if available (truncated)
        if event.detail and len(event.detail) > 0:
            detail_preview = self._truncate(event.detail.replace('\n', ' '), 80)
            detail_line = f"{self.GRAY}{indent}  ├─ 💬 {detail_preview}{self.RESET}"
            print(detail_line)
        
        # Show error message if failed
        if status == ExecutionStatus.FAILED and event.message:
            error_msg = self._truncate(event.message, 80)
            error_line = f"{self.GRAY}{indent}  └─ {self.RED}⚠️  {error_msg}{self.RESET}"
            print(error_line)
    
    def _print_progress_mode(self, event: ExecutionEvent, tracker: "ParallelGroupTracker"):
        """Print event in progress bar mode with visual progress bar."""
        import sys
        import shutil
        
        status = event.execution_status
        
        # On first event, update total and print initial progress bar
        if tracker.total == 0 and status == ExecutionStatus.STARTED:
            # We don't know total yet, just track started
            tracker.total = max(tracker.started, tracker.completed + tracker.failed)
        
        # Update total if needed (as more events come in)
        tracker.total = max(tracker.total, tracker.started, tracker.completed + tracker.failed)
        
        # Check if we're in a TTY
        is_tty = sys.stdout.isatty()
        
        if is_tty and tracker.progress_line_number is not None:
            # Move cursor up to progress line and redraw
            lines_to_move = 1
            sys.stdout.write(f"\033[{lines_to_move}A")  # Move up
            sys.stdout.write("\033[K")  # Clear line
            self._render_progress_bar(tracker)
            sys.stdout.flush()
        else:
            # First time or not TTY - just print
            self._render_progress_bar(tracker)
            if is_tty:
                tracker.progress_line_number = 0  # Mark that we've printed
        
        # Show individual event details below progress bar (only for failures or completed)
        if status == ExecutionStatus.FAILED:
            indent = "  "
            error_msg = self._truncate(event.message or "Unknown error", 60)
            actor_name = event.actor.name
            if event.parallel_index is not None:
                actor_name = f"{actor_name} #{event.parallel_index}"
            
            error_line = (
                f"{indent}{self.RED}✖{self.RESET} {actor_name}: {error_msg}"
            )
            print(error_line)
    
    def _render_progress_bar(self, tracker: "ParallelGroupTracker"):
        """Render a visual progress bar for a parallel group."""
        import shutil
        
        # Get terminal width
        try:
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80
        
        # Calculate progress
        total = max(tracker.total, 1)  # Avoid division by zero
        completed = tracker.completed + tracker.failed
        percentage = (completed / total) * 100
        
        # Calculate elapsed time
        if tracker.start_time is None:
            elapsed_str = "0s"
        else:
            elapsed = asyncio.get_event_loop().time() - tracker.start_time
            elapsed_str = self._format_time(elapsed)
        
        # Build progress bar components
        group_name = tracker.name or "Parallel Group"
        
        # Status indicators
        success_indicator = f"{self.GREEN}{tracker.completed}✓{self.RESET}" if tracker.completed > 0 else ""
        fail_indicator = f"{self.RED}{tracker.failed}✖{self.RESET}" if tracker.failed > 0 else ""
        indicators = " ".join(filter(None, [success_indicator, fail_indicator]))
        
        # Progress text
        progress_text = f"{completed}/{total}"
        
        # Header line
        header = f"{self.BOLD}{self.CYAN}▶ {group_name}{self.RESET}"
        status_info = f"[{progress_text}] {percentage:.0f}% {indicators} {self.GRAY}({elapsed_str}){self.RESET}"
        
        # Calculate bar width
        # Format: "▶ Name [10/20] 50% 5✓ 1✖ (1.2s) [=========>          ]"
        header_text_len = len(group_name) + 2  # "▶ " prefix
        status_text_len = len(progress_text) + len(str(int(percentage))) + 10  # rough estimate
        available_bar_width = terminal_width - header_text_len - status_text_len - 10
        available_bar_width = max(20, min(50, available_bar_width))  # Clamp between 20-50
        
        # Build the progress bar
        filled_width = int((completed / total) * available_bar_width)
        empty_width = available_bar_width - filled_width
        
        # Bar characters
        if percentage >= 100:
            bar = f"{self.GREEN}{'█' * available_bar_width}{self.RESET}"
        else:
            bar = f"{self.CYAN}{'█' * filled_width}{self.RESET}{self.GRAY}{'░' * empty_width}{self.RESET}"
        
        # Print the complete progress line
        print(f"{header} {status_info}")
        print(f"  [{bar}]")
    
    
    def _print_group_summary(self, tracker: "ParallelGroupTracker"):
        """Print summary when parallel group completes."""
        if tracker.start_time is None:
            elapsed_str = "0s"
        else:
            elapsed = asyncio.get_event_loop().time() - tracker.start_time
            elapsed_str = self._format_time(elapsed)
        
        total = tracker.completed + tracker.failed
        success_rate = (tracker.completed / total * 100) if total > 0 else 0
        
        # For progress mode, print final progress bar then summary
        if self.parallel_mode == "progress":
            print()  # Blank line after last progress update
            self._render_progress_bar(tracker)
            print()
        
        print(f"\n{self.GRAY}{'─' * 70}{self.RESET}")
        print(f"{self.BOLD}✓ Parallel Group Complete: {tracker.name}{self.RESET}")
        print(f"  {self.GREEN}├─{self.RESET} Completed: {self.GREEN}{tracker.completed}{self.RESET}")
        print(f"  {self.RED}├─{self.RESET} Failed: {self.RED}{tracker.failed}{self.RESET}")
        print(f"  {self.MAGENTA}├─{self.RESET} Success Rate: {self.GREEN}{success_rate:.1f}%{self.RESET}")
        print(f"  {self.MAGENTA}└─{self.RESET} Total Time: {self.MAGENTA}{elapsed_str}{self.RESET}")
        print(f"{self.GRAY}{'─' * 70}{self.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SSE Event Streamer (for FastAPI streaming)
# ═══════════════════════════════════════════════════════════════════════════════

class EventStreamer:
    """
    Wrapper for Server-Sent Events (SSE) streaming of execution events.
    
    Usage is just like EventPrinter - simply plug it into the event bus:
    
    ```python
    streamer = EventStreamer()
    bus = EventBus()
    bus.subscribe(streamer)
    
    # Then use streamer.stream() for FastAPI response:
    return StreamingResponse(streamer.stream(), media_type="text/event-stream")
    ```
    """
    
    def __init__(
        self, 
        include_cache_stats: bool = True,
        heartbeat_interval: float = 0.25
    ):
        """
        Initialize the SSE event streamer.
        
        Args:
            include_cache_stats: Whether to include cache statistics in completion event
            heartbeat_interval: Interval in seconds for keep-alive heartbeats
        """
        self.queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()
        self.include_cache_stats = include_cache_stats
        self.heartbeat_interval = heartbeat_interval
        self._execution_task: Optional[asyncio.Task] = None
        self._completed = False
    
    async def __call__(self, event: ExecutionEvent):
        """Handle incoming events - called by EventBus subscriber."""
        await self.queue.put(event)
    
    def set_execution_task(self, task: asyncio.Task):
        """Set the execution task to monitor for completion."""
        self._execution_task = task
    
    async def stream(
        self, 
        execution_task: Optional[asyncio.Task] = None,
        cache_backend: Optional[Any] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate SSE-formatted event stream for FastAPI.
        
        Args:
            execution_task: Optional task to monitor for completion
            cache_backend: Optional cache backend to get stats from
            
        Yields:
            SSE-formatted event strings (data: {...})
        """
        if execution_task:
            self._execution_task = execution_task
        
        # Start stream with connection message
        yield ": stream-start\n\n"
        
        try:
            while True:
                # Check if execution completed and queue is empty
                if self._execution_task and self._execution_task.done() and self.queue.empty():
                    break
                
                try:
                    # Wait for next event with timeout for heartbeat
                    event = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.heartbeat_interval
                    )
                    
                    # Send event as SSE data
                    yield f"data: {event.model_dump_json()}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": keep-alive\n\n"
            
            # Check for execution errors
            if self._execution_task:
                await self._execution_task
            
            # Send completion event
            final_event = {
                "event_type": "complete",
                "message": "Execution completed successfully"
            }
            
            # Add cache statistics if available
            if self.include_cache_stats and cache_backend:
                if hasattr(cache_backend, 'get_stats'):
                    final_event["cache_stats"] = cache_backend.get_stats()
            
            yield f"data: {json.dumps(final_event)}\n\n"
            
        except Exception as e:
            # Send error event
            error_event = {
                "event_type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        
        finally:
            self._completed = True
