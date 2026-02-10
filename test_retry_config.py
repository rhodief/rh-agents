"""
Unit tests for RetryConfig model.

Tests:
- Config validation and defaults
- Backoff calculation with different strategies
- Error filtering (whitelist/blacklist logic)
- Smart defaults behavior
"""
import pytest
import asyncio
from rh_agents.core.retry import (
    RetryConfig,
    BackoffStrategy,
    DEFAULT_RETRYABLE_EXCEPTIONS,
    DEFAULT_NON_RETRYABLE_EXCEPTIONS
)


class TestRetryConfigDefaults:
    """Test default configuration values."""
    
    def test_default_values(self):
        """Test that defaults are sensible."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.enabled is True
        assert config.retry_on_exceptions is None
        assert config.exclude_exceptions is None
        assert config.retry_timeout is None
    
    def test_enabled_flag(self):
        """Test enabled/disabled flag."""
        enabled = RetryConfig(enabled=True)
        disabled = RetryConfig(enabled=False)
        
        assert enabled.enabled is True
        assert disabled.enabled is False
    
    def test_validation_min_max_attempts(self):
        """Test that max_attempts must be >= 1."""
        with pytest.raises(Exception):  # Pydantic validation error
            RetryConfig(max_attempts=0)
        
        with pytest.raises(Exception):
            RetryConfig(max_attempts=-1)
        
        # Should work
        config = RetryConfig(max_attempts=1)
        assert config.max_attempts == 1


class TestBackoffCalculation:
    """Test backoff delay calculation with different strategies."""
    
    def test_constant_backoff(self):
        """Constant backoff always returns same delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.CONSTANT,
            initial_delay=2.0,
            jitter=False
        )
        
        # All attempts should have same delay
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 2.0
        assert config.calculate_delay(10) == 2.0
    
    def test_linear_backoff(self):
        """Linear backoff increases linearly with attempt."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.LINEAR,
            initial_delay=2.0,
            jitter=False
        )
        
        # delay = initial_delay * attempt
        assert config.calculate_delay(1) == 2.0   # 2 * 1
        assert config.calculate_delay(2) == 4.0   # 2 * 2
        assert config.calculate_delay(3) == 6.0   # 2 * 3
        assert config.calculate_delay(5) == 10.0  # 2 * 5
    
    def test_exponential_backoff(self):
        """Exponential backoff increases exponentially."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        
        # delay = initial_delay * (multiplier ^ (attempt - 1))
        assert config.calculate_delay(1) == 1.0   # 1 * 2^0 = 1
        assert config.calculate_delay(2) == 2.0   # 1 * 2^1 = 2
        assert config.calculate_delay(3) == 4.0   # 1 * 2^2 = 4
        assert config.calculate_delay(4) == 8.0   # 1 * 2^3 = 8
        assert config.calculate_delay(5) == 16.0  # 1 * 2^4 = 16
    
    def test_fibonacci_backoff(self):
        """Fibonacci backoff follows fibonacci sequence."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.FIBONACCI,
            initial_delay=1.0,
            jitter=False
        )
        
        # Fibonacci: 1, 1, 2, 3, 5, 8, 13, ...
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 1.0
        assert config.calculate_delay(3) == 2.0
        assert config.calculate_delay(4) == 3.0
        assert config.calculate_delay(5) == 5.0
        assert config.calculate_delay(6) == 8.0
        assert config.calculate_delay(7) == 13.0
    
    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        
        # Exponential would be: 1, 2, 4, 8, 16, 32...
        # But capped at max_delay=10
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 8.0
        assert config.calculate_delay(5) == 10.0  # Capped
        assert config.calculate_delay(6) == 10.0  # Capped
        assert config.calculate_delay(10) == 10.0  # Capped
    
    def test_jitter(self):
        """Jitter adds randomness to delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.CONSTANT,
            initial_delay=10.0,
            jitter=True
        )
        
        # With jitter, delays should vary but be in range [7.5, 12.5] (±25%)
        delays = [config.calculate_delay(1) for _ in range(100)]
        
        # All should be different (probabilistically)
        assert len(set(delays)) > 50  # At least 50 different values
        
        # All should be in expected range
        for delay in delays:
            assert 7.5 <= delay <= 12.5
    
    def test_fibonacci_helper(self):
        """Test fibonacci number calculation."""
        assert RetryConfig._fibonacci(1) == 1
        assert RetryConfig._fibonacci(2) == 1
        assert RetryConfig._fibonacci(3) == 2
        assert RetryConfig._fibonacci(4) == 3
        assert RetryConfig._fibonacci(5) == 5
        assert RetryConfig._fibonacci(6) == 8
        assert RetryConfig._fibonacci(7) == 13
        assert RetryConfig._fibonacci(8) == 21


class TestErrorFiltering:
    """Test error filtering with whitelist/blacklist."""
    
    def test_default_retryable_exceptions(self):
        """Default behavior retries transient errors."""
        config = RetryConfig()
        
        # Should retry these (in DEFAULT_RETRYABLE_EXCEPTIONS)
        assert config.should_retry(TimeoutError()) is True
        assert config.should_retry(asyncio.TimeoutError()) is True
        assert config.should_retry(ConnectionError()) is True
        assert config.should_retry(ConnectionResetError()) is True
        assert config.should_retry(ConnectionAbortedError()) is True
        assert config.should_retry(ConnectionRefusedError()) is True
    
    def test_default_retries_all_exceptions(self):
        """New intuitive behavior: retry ALL exceptions by default."""
        config = RetryConfig()
        
        # ALL exceptions should be retried by default
        assert config.should_retry(ValueError()) is True
        assert config.should_retry(TypeError()) is True
        assert config.should_retry(KeyError()) is True
        assert config.should_retry(AttributeError()) is True
        assert config.should_retry(AssertionError()) is True
        assert config.should_retry(Exception()) is True
    
    def test_whitelist_overrides_defaults(self):
        """When whitelist is set, only those exceptions are retried."""
        config = RetryConfig(
            retry_on_exceptions=[TimeoutError]
        )
        
        # Only TimeoutError should be retried
        assert config.should_retry(TimeoutError()) is True
        
        # Other normally retryable errors should NOT be retried
        assert config.should_retry(ConnectionError()) is False
        assert config.should_retry(RuntimeError()) is False
    
    def test_exclude_exceptions_opt_out(self):
        """exclude_exceptions lets you opt-out specific errors from retry."""
        config = RetryConfig(
            exclude_exceptions=[RuntimeError, ValueError]
        )
        
        # Excluded exceptions should NOT be retried
        assert config.should_retry(RuntimeError()) is False
        assert config.should_retry(ValueError()) is False
        
        # Other exceptions should be retried (default behavior)
        assert config.should_retry(TimeoutError()) is True
        assert config.should_retry(ConnectionError()) is True
        assert config.should_retry(TypeError()) is True
    
    def test_whitelist_and_blacklist_together(self):
        """Blacklist takes precedence over whitelist."""
        config = RetryConfig(
            retry_on_exceptions=[TimeoutError, ConnectionError],
            exclude_exceptions=[ConnectionError]
        )
        
        # TimeoutError is in whitelist and not in blacklist
        assert config.should_retry(TimeoutError()) is True
        
        # ConnectionError is in whitelist BUT also in blacklist - blacklist wins
        assert config.should_retry(ConnectionError()) is False
        
        # RuntimeError is not in whitelist
        assert config.should_retry(RuntimeError()) is False
    
    def test_none_whitelist_retries_all(self):
        """When whitelist is None, retry ALL exceptions (new default)."""
        config = RetryConfig(retry_on_exceptions=None)
        
        # ALL exceptions should be retried when retry_on_exceptions is None
        assert config.should_retry(TimeoutError()) is True
        assert config.should_retry(ValueError()) is True
        assert config.should_retry(RuntimeError()) is True
        assert config.should_retry(Exception()) is True


class TestRetryTimeout:
    """Test retry timeout configuration."""
    
    def test_no_retry_timeout(self):
        """No timeout by default."""
        config = RetryConfig()
        assert config.retry_timeout is None
    
    def test_retry_timeout_value(self):
        """Can set retry timeout."""
        config = RetryConfig(retry_timeout=60.0)
        assert config.retry_timeout == 60.0
    
    def test_retry_timeout_validation(self):
        """Retry timeout must be >= 0."""
        with pytest.raises(Exception):  # Pydantic validation error
            RetryConfig(retry_timeout=-1.0)
        
        # Should work
        config = RetryConfig(retry_timeout=0.0)
        assert config.retry_timeout == 0.0


class TestCustomConfigurations:
    """Test custom configuration scenarios."""
    
    def test_aggressive_retry(self):
        """Test aggressive retry config."""
        config = RetryConfig(
            max_attempts=10,
            initial_delay=0.1,
            backoff_multiplier=1.5,
            max_delay=5.0
        )
        
        assert config.max_attempts == 10
        assert config.initial_delay == 0.1
        assert config.backoff_multiplier == 1.5
        assert config.max_delay == 5.0
    
    def test_conservative_retry(self):
        """Test conservative retry config."""
        config = RetryConfig(
            max_attempts=2,
            initial_delay=5.0,
            backoff_strategy=BackoffStrategy.CONSTANT
        )
        
        assert config.max_attempts == 2
        assert config.initial_delay == 5.0
        assert config.backoff_strategy == BackoffStrategy.CONSTANT
    
    def test_disabled_retry(self):
        """Test completely disabled retry."""
        config = RetryConfig(enabled=False)
        
        assert config.enabled is False
        # Other fields still have defaults
        assert config.max_attempts == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
