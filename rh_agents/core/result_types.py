from typing import Union, Any
from pydantic import BaseModel, Field

class LLM_Result(BaseModel):
    """Result type for LLM actor executions"""
    content: str
    tools: list[str] = Field(default_factory=list)
    tokens_used: Union[int, None] = None
    model_name: Union[str, None] = None

class Tool_Result(BaseModel):
    """Result type for Tool actor executions"""
    output: Any
    tool_name: str
    success: bool = True

class Agent_Result(BaseModel):
    """Result type for Agent actor executions"""
    response: Any
    agent_name: str
    sub_tasks_completed: list[str] = Field(default_factory=list)
    
    
    