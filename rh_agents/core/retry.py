"""
Retry mechanism for execution events.

Provides configurable retry behavior with:
- Multiple backoff strategies (constant, linear, exponential, fibonacci)
- Smart error filtering (whitelist/blacklist with sensible defaults)
- Configuration at multiple levels (event, actor-type, global, built-in defaults)
"""
from __future__ import annotations
import asyncio
import random
from enum import Enum
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from rh_agents.core.types import EventType


class BackoffStrategy(str, Enum):
    """Backoff strategies for retry delays."""
    CONSTANT = "constant"      # Fixed delay between retries
    LINEAR = "linear"          # Linearly increasing delay (delay * attempt)
    EXPONENTIAL = "exponential"  # Exponentially increasing delay (delay * multiplier^attempt)
    FIBONACCI = "fibonacci"    # Fibonacci sequence delays (1, 1, 2, 3, 5, 8, ...)


# Default transient errors that should be retried
DEFAULT_RETRYABLE_EXCEPTIONS: list[type[Exception]] = [
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    # Note: HTTP-specific errors (429, 503, etc.) can be added by users via retry_on_exceptions
]

# Default permanent errors that should NOT be retried
DEFAULT_NON_RETRYABLE_EXCEPTIONS: list[type[Exception]] = [
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    # Note: Pydantic ValidationError should be added if used in the project
    # Note: Auth/permission errors can be added by users via exclude_exceptions
]


class RetryConfig(BaseModel):
    """
    Configuration for retry behavior.
    
    Supports multiple backoff strategies, error filtering with smart defaults,
    and timeout controls.
    
    Examples:
        >>> # Basic exponential backoff
        >>> config = RetryConfig(max_attempts=3, initial_delay=1.0)
        
        >>> # Custom error filtering
        >>> config = RetryConfig(
        ...     max_attempts=5,
        ...     retry_on_exceptions=[TimeoutError, ConnectionError]
        ... )
        
        >>> # Disable retry
        >>> config = RetryConfig(enabled=False)
    """
    
    # Core retry settings
    max_attempts: int = Field(
        default=3, 
        ge=1, 
        description="Maximum retry attempts (including initial attempt)"
    )
    backoff_strategy: BackoffStrategy = Field(
        default=BackoffStrategy.EXPONENTIAL,
        description="Strategy for calculating delay between retries"
    )
    initial_delay: float = Field(
        default=1.0, 
        ge=0, 
        description="Initial delay in seconds before first retry"
    )
    max_delay: float = Field(
        default=60.0, 
        ge=0, 
        description="Maximum delay between retries in seconds"
    )
    backoff_multiplier: float = Field(
        default=2.0, 
        ge=1.0, 
        description="Multiplier for exponential/linear backoff"
    )
    jitter: bool = Field(
        default=True, 
        description="Add random jitter to prevent thundering herd"
    )
    
    # Error filtering (override defaults)
    retry_on_exceptions: Optional[list[type[Exception]]] = Field(
        default=None, 
        description="Whitelist of exceptions to retry (overrides defaults if set)"
    )
    exclude_exceptions: Optional[list[type[Exception]]] = Field(
        default=None, 
        description="Blacklist of exceptions to never retry (merged with defaults)"
    )
    
    # Timeout settings
    retry_timeout: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Total timeout for all retry attempts combined (seconds)"
    )
    
    # Control
    enabled: bool = Field(
        default=True, 
        description="Enable/disable retry for this config"
    )
    
    model_config = {"arbitrary_types_allowed": True}
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Logic:
        1. Check if exception is in exclude list (defaults + user-specified)
        2. If whitelist is set, exception must be in it
        3. Otherwise, check if exception is in default retryable list
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the exception should trigger a retry, False otherwise
            
        Examples:
            >>> config = RetryConfig()
            >>> config.should_retry(TimeoutError())  # True (in defaults)
            >>> config.should_retry(ValueError())    # False (non-retryable)
            
            >>> config = RetryConfig(retry_on_exceptions=[TimeoutError])
            >>> config.should_retry(TimeoutError())     # True (in whitelist)
            >>> config.should_retry(ConnectionError())  # False (not in whitelist)
        """
        # Build effective exclude list (user's + defaults)
        exclude_list = DEFAULT_NON_RETRYABLE_EXCEPTIONS.copy()
        if self.exclude_exceptions:
            exclude_list.extend(self.exclude_exceptions)
        
        # Check exclusions first
        if any(isinstance(exception, exc_type) for exc_type in exclude_list):
            return False
        
        # If whitelist is set, exception must be in it
        if self.retry_on_exceptions is not None:
            return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)
        
        # Otherwise, check default retryable list
        return any(isinstance(exception, exc_type) for exc_type in DEFAULT_RETRYABLE_EXCEPTIONS)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry attempt.
        
        Args:
            attempt: Retry attempt number (1-based, so 1 = first retry)
        
        Returns:
            Delay in seconds (with jitter if enabled)
            
        Examples:
            >>> config = RetryConfig(backoff_strategy=BackoffStrategy.CONSTANT, initial_delay=2.0)
            >>> config.calculate_delay(1)  # ~2.0 seconds
            
            >>> config = RetryConfig(backoff_strategy=BackoffStrategy.EXPONENTIAL, initial_delay=1.0)
            >>> config.calculate_delay(1)  # ~1.0 seconds
            >>> config.calculate_delay(2)  # ~2.0 seconds
            >>> config.calculate_delay(3)  # ~4.0 seconds
        """
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.initial_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Generate fibonacci number for attempt
            fib = self._fibonacci(attempt)
            delay = self.initial_delay * fib
        else:
            # Fallback to constant
            delay = self.initial_delay
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter if enabled (±25% jitter)
        if self.jitter:
            jitter_factor = 0.75 + (random.random() * 0.5)
            delay = delay * jitter_factor
        
        return delay
    
    @staticmethod
    def _fibonacci(n: int) -> int:
        """
        Calculate nth fibonacci number (1-based).
        
        Args:
            n: Position in fibonacci sequence (1-based)
            
        Returns:
            Fibonacci number at position n
            
        Examples:
            >>> RetryConfig._fibonacci(1)  # 1
            >>> RetryConfig._fibonacci(2)  # 1
            >>> RetryConfig._fibonacci(3)  # 2
            >>> RetryConfig._fibonacci(4)  # 3
            >>> RetryConfig._fibonacci(5)  # 5
        """
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a


def get_default_retry_config_by_actor_type() -> dict["EventType", RetryConfig]:
    """
    Get default retry configurations by actor type.
    
    Returns a dict with sensible defaults:
    - TOOL_CALL: Retry enabled (tools often hit external services)
    - LLM_CALL: Retry enabled (LLM APIs can be flaky)
    - AGENT_CALL: Retry disabled (agents handle their own logic)
    
    Returns:
        Dictionary mapping EventType to RetryConfig
    """
    from rh_agents.core.types import EventType
    
    return {
        EventType.TOOL_CALL: RetryConfig(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=30.0,
            enabled=True
        ),
        EventType.LLM_CALL: RetryConfig(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=30.0,
            enabled=True
        ),
        EventType.AGENT_CALL: RetryConfig(
            enabled=False  # Agents handle their own logic
        )
    }
