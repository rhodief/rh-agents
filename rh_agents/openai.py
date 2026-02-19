import os
from typing import Any, List, Dict, Union, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, Field

from rh_agents.core.actors import LLM, ToolSet
from rh_agents.core.result_types import LLM_Result, LLM_Tool_Call
from rh_agents.models import Message, AuthorType


class ToolCall(BaseModel):
    """Model representing a tool call with OpenAI's tool call ID"""
    id: str
    tool_name: str
    arguments: str


class OpenAIFunction(BaseModel):
    """Pydantic model for OpenAI function definition"""
    name: str
    description: str
    parameters: Dict[str, Any]


class OpenAITool(BaseModel):
    """Pydantic model for OpenAI tool definition"""
    type: str = "function"
    function: OpenAIFunction


class OpenAIRequest(BaseModel):
    """Input model for OpenAI LLM calls"""
    prompt: str
    model: str = "gpt-3.5-turbo"
    max_completion_tokens: int = 4000
    temperature: float = 1
    system_message: str = "You are a helpful assistant."
    history: Optional[List[Message]] = None
    tools: Optional[ToolSet] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


async def openai_handler(request: OpenAIRequest, **kwargs) -> LLM_Result:
    """
    Handler function for OpenAI LLM calls with function calling support.
    
    Args:
        request: OpenAIRequest containing the prompt and parameters
    
    Returns:
        LLM_Result with the response and metadata
    """
    # Initialize OpenAI client with API key from environment
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare messages with proper typing
    messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]] = [
        {"role": "system", "content": request.system_message}
    ]
    
    # Add conversation history if provided
    if request.history:
        for msg in request.history:
            if msg.author == AuthorType.USER:
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append({"role": "assistant", "content": msg.content})
    
    # Add the main prompt
    messages.append({"role": "user", "content": request.prompt})
    
    try:
        # Prepare API call parameters
        api_params = {
            "model": request.model,
            "messages": messages,
            "max_completion_tokens": request.max_completion_tokens,
            "temperature": request.temperature
        }
        
        # Convert ToolSet tools to OpenAI function calling format
        if request.tools and request.tools.tools:
            tools_list = []
            for tool in request.tools.tools:
                # Convert Tool to OpenAI function format
                function_def = tool_to_openai_function(tool)
                tools_list.append({
                    "type": "function",
                    "function": function_def
                })
            api_params["tools"] = tools_list
            
            if request.tool_choice:
                api_params["tool_choice"] = request.tool_choice
        
        # Make the API call
        response = client.chat.completions.create(**api_params)
        
        # Extract response data
        message = response.choices[0].message
        content = message.content or "No content generated"
        tokens_used = response.usage.total_tokens if response.usage else None
        
        # Handle tool calls if present
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(LLM_Tool_Call(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                ))
        
        return LLM_Result(
            content=content,
            tools=tool_calls,
            tokens_used=tokens_used,
            model_name=request.model
        )
        
    except Exception as e:
        # Handle errors gracefully
        return LLM_Result(
            content=f"Error calling OpenAI API: {str(e)}",
            tokens_used=0,
            model_name=request.model
        )



# Alternative handler for simple text prompts
class SimplePromptRequest(BaseModel):
    """Simple input model for basic text prompts"""
    text: str
    model: str = "gpt-3.5-turbo"


async def simple_openai_handler(request: SimplePromptRequest, extra_context: str = "", tools_from_llm: Optional[ToolSet] = None) -> LLM_Result:
    """
    Simplified handler for basic text prompts.
    
    Args:
        request: SimplePromptRequest with just text and model
        extra_context: Additional context string (optional)
        tools_from_llm: ToolSet from the LLM instance to convert to function calls
    
    Returns:
        LLM_Result with the response and metadata
    """
    # Convert simple request to full OpenAI request
    full_request = OpenAIRequest(
        prompt=request.text,
        model=request.model
    )
    
    return await openai_handler(full_request)



# Helper functions for creating function definitions
def create_function_parameter_schema(properties: Dict[str, Dict[str, Any]], required: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Helper function to create OpenAI function parameter schema.
    
    Args:
        properties: Dict of parameter names to their schema definitions
        required: List of required parameter names
    
    Returns:
        Parameter schema dict for OpenAI function
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or []
    }


def create_openai_function(name: str, description: str, properties: Dict[str, Dict[str, Any]], required: Optional[List[str]] = None) -> OpenAIFunction:
    """
    Helper function to create an OpenAI function definition.
    
    Args:
        name: Function name
        description: Function description
        properties: Parameter properties schema
        required: Required parameter names
    
    Returns:
        OpenAIFunction instance
    """
    parameters = create_function_parameter_schema(properties, required)
    return OpenAIFunction(
        name=name,
        description=description,
        parameters=parameters
    )


def create_openai_tool(name: str, description: str, properties: Dict[str, Dict[str, Any]], required: Optional[List[str]] = None) -> OpenAITool:
    """
    Helper function to create an OpenAI tool definition.
    
    Args:
        name: Tool name  
        description: Tool description
        properties: Parameter properties schema
        required: Required parameter names
    
    Returns:
        OpenAITool instance
    """
    function = create_openai_function(name, description, properties, required)
    return OpenAITool(function=function)


def tool_to_openai_function(tool) -> Dict[str, Any]:
    """
    Convert a Tool from ToolSet to OpenAI function format.
    
    Args:
        tool: Tool instance from ToolSet
        
    Returns:
        Dict representing OpenAI function definition
    """
    from rh_agents.core.actors import Tool
    
    if not isinstance(tool, Tool) or tool.input_model is None:
        raise ValueError(f"Expected Tool instance, got {type(tool)}")
    
    # Use Pydantic's built-in model_json_schema to generate the parameters schema
    parameters = tool.input_model.model_json_schema()    
    
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters
    }
