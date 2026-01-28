from enum import Enum
from datetime import datetime
from typing import Optional
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


class InterruptReason(str, Enum):
    """Reason for execution interruption."""
    USER_CANCELLED = "user_cancelled"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    ERROR_THRESHOLD = "error_threshold"
    PRIORITY_OVERRIDE = "priority_override"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CUSTOM = "custom"


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
