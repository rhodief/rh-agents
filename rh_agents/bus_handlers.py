

# ═══════════════════════════════════════════════════════════════════════════════
# Beautiful Event Printer
# ═══════════════════════════════════════════════════════════════════════════════

from collections.abc import Callable
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus


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
        self.total_execution_time = 0.0
        self.events_by_type: dict[str, int] = {}
    
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
        elif status == ExecutionStatus.COMPLETED:
            self.completed_events += 1
            if event.execution_time:
                self.total_execution_time += event.execution_time
        elif status == ExecutionStatus.FAILED:
            self.failed_events += 1
        
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
            detail_line = (
                f"{self.GRAY}{indent}  ├─ {detail_icon} {detail_preview}{self.RESET}"
            )
            lines.append(detail_line)
        
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
        print(f"  {self.RED}└─{self.RESET} Failed: {self.RED}{self.failed_events}{self.RESET}\n")
        
        # Success rate
        if self.started_events > 0:
            success_rate = (self.completed_events / self.started_events) * 100
            rate_color = self.GREEN if success_rate >= 80 else (self.YELLOW if success_rate >= 50 else self.RED)
            print(f"{self.BOLD}Success Rate:{self.RESET} {rate_color}{success_rate:.1f}%{self.RESET}\n")
        
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
