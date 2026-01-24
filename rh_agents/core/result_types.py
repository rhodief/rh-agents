from typing import Union, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field

@runtime_checkable
class ActorOutput(Protocol):
    """
    Protocol for actor output types.
    
    All actor result types should implement this protocol
    to enable generic code while maintaining domain-specific fields.
    """
    success: bool
    error: str | None

class LLM_Tool_Call(BaseModel):
    """Model representing a tool call made by an LLM"""
    tool_name: str
    arguments: str

class LLM_Result(BaseModel):
    """
    Result type for LLM actor executions.
    
    Implements ActorOutput protocol.
    """
    content: str
    tools: list[LLM_Tool_Call] = Field(default_factory=list)
    tokens_used: Union[int, None] = None
    model_name: Union[str, None] = None
    error_message: Union[str, None] = None
    
    @property
    def is_content(self) -> bool:
        return bool(self.content and not self.tools)

    @property
    def is_tool_call(self) -> bool:
        return len(self.tools) > 0
    
    @property
    def succeeded(self) -> bool:
        """Alias for protocol compatibility."""
        return self.error_message is None
    
    @property
    def success(self) -> bool:
        """Protocol implementation."""
        return self.error_message is None
    
    @property
    def error(self) -> str | None:
        """Protocol implementation."""
        return self.error_message

class Tool_Result(BaseModel):
    """
    Result type for Tool actor executions.
    
    Implements ActorOutput protocol.
    """
    output: Any
    tool_name: str
    success: bool = True
    error: str | None = None
    
    
    