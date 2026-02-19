from enum import Enum
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class EventType(str, Enum):
    AGENT_CALL = 'agent_call'
    TOOL_CALL = 'tool_call'
    LLM_CALL = 'llm_call'


class ExecutionStatus(str, Enum):
    STARTED = 'started'
    COMPLETED = 'completed'
    FAILED = 'failed'
    AWAITING = 'awaiting'
    HUMAN_INTERVENTION = 'human_intervention'
    RECOVERED = 'recovered'
    INTERRUPTED = 'interrupted'  # NEW: Execution was interrupted
    CANCELLING = 'cancelling'    # NEW: Transitional state during cancellation
    RETRYING = 'retrying'        # NEW: Event is being retried after failure


class InterruptReason(str, Enum):
    """Reason for execution interruption."""
    USER_CANCELLED = "user_cancelled"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    ERROR_THRESHOLD = "error_threshold"
    PRIORITY_OVERRIDE = "priority_override"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CUSTOM = "custom"


class ErrorStrategy(str, Enum):
    """Error handling strategy for builder-generated agents."""
    RAISE = "raise"                  # Raise exception immediately (default)
    RETURN_NONE = "return_none"      # Return ExecutionResult with ok=False
    LOG_AND_CONTINUE = "log"         # Log error and return partial results
    SILENT = "silent"                # Suppress errors, return None


class AggregationStrategy(str, Enum):
    """Strategy for aggregating multiple tool execution results."""
    DICT = "dict"                    # Dictionary keyed by tool name (default)
    LIST = "list"                    # List of results in execution order
    CONCATENATE = "concatenate"      # Concatenate string results with separator
    FIRST = "first"                  # Return only the first successful result


class InterruptSignal(BaseModel):
    """Signal model for execution interruption."""
    reason: InterruptReason = Field(description="Why execution was interrupted")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    triggered_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    triggered_by: Optional[str] = Field(default=None, description="Who/what triggered interrupt")
    save_checkpoint: bool = Field(default=True, description="Save state before terminating")


class InterruptEvent(BaseModel):
    """Special event published when execution is interrupted."""
    signal: InterruptSignal
    state_id: str = Field(description="ID of the ExecutionState that was interrupted")
    
    model_config = {"arbitrary_types_allowed": True}


class LogSeverity(str, Enum):
    """Severity levels for log events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEvent(BaseModel):
    """
    Log event that can be published to the event bus.
    
    This allows structured logging within execution flows that integrates
    with the event stream and printers.
    
    Example:
        ```python
        # Within an agent or tool execution
        await state.log(
            severity=LogSeverity.INFO,
            message="Processing document",
            metadata={"doc_id": "123", "page": 5}
        )
        ```
    """
    severity: LogSeverity = Field(description="Log severity level")
    message: str = Field(description="Log message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    address: str = Field(default="", description="Event address in execution tree")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    model_config = {"arbitrary_types_allowed": True}
