from typing import Union, Any, Protocol, runtime_checkable, Dict
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


class ToolExecutionResult(BaseModel):
    """
    Result from executing multiple tools in parallel.
    
    Used by ToolExecutorAgent to aggregate results from multiple tool calls.
    Tools are keyed by their name, and execution order is preserved for debugging.
    """
    results: Dict[str, Any] = Field(
        description="Tool results keyed by tool name"
    )
    execution_order: list[str] = Field(
        default_factory=list,
        description="Order in which tools were executed"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors that occurred during tool execution, keyed by tool name"
    )
    
    def get(self, tool_name: str) -> Any:
        """Get result for a specific tool by name."""
        return self.results.get(tool_name)
    
    def first(self) -> Any:
        """Get the first successful result (by execution order)."""
        if self.execution_order:
            first_tool = self.execution_order[0]
            return self.results.get(first_tool)
        return None
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during execution."""
        return len(self.errors) > 0
    
    def all_failed(self) -> bool:
        """Check if all tool executions failed."""
        return len(self.errors) > 0 and len(self.results) == 0
    
    def to_list(self) -> list[Any]:
        """
        Convert results to list in execution order.
        
        Returns:
            List of results in the order tools were executed
        """
        return [self.results.get(tool_name) for tool_name in self.execution_order if tool_name in self.results]
    
    def to_concatenated(self, separator: str = "\n\n") -> str:
        """
        Concatenate all string results with a separator.
        
        Args:
            separator: String to join results with (default: double newline)
            
        Returns:
            Concatenated string of all results
        """
        str_results = []
        for tool_name in self.execution_order:
            if tool_name in self.results:
                result = self.results[tool_name]
                # Convert to string if not already
                str_results.append(str(result) if not isinstance(result, str) else result)
        return separator.join(str_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return results as dictionary (already in this format).
        
        Returns:
            Dictionary of results keyed by tool name
        """
        return self.results