"""Custom exceptions for RH-Agents framework."""
from typing import Optional
from rh_agents.core.types import InterruptReason


class ExecutionInterrupted(Exception):
    """Exception raised when execution is interrupted by user or system."""
    
    def __init__(self, reason: InterruptReason, message: Optional[str] = None):
        self.reason = reason
        self.message = message or f"Execution interrupted: {reason.value}"
        super().__init__(self.message)
