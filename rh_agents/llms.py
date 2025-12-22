from rh_agents.core.actors import LLM, ToolSet
from rh_agents.core.result_types import LLM_Result
from rh_agents.core.types import EventType
from rh_agents.models import Message
from rh_agents.openai import OpenAIRequest, openai_handler, SimplePromptRequest, simple_openai_handler


class OpenAILLM(LLM[OpenAIRequest]):
    """OpenAI LLM Actor with function calling support"""
    
    def __init__(
        self,
        name: str = "OpenAI-LLM",
        description: str = "OpenAI GPT model with function calling capabilities"        
    ):
        async def handler_wrapper(request: OpenAIRequest, extra_context: str, execution_state) -> LLM_Result:
            return await openai_handler(request)
        
        super().__init__(
            name=name,
            description=description,
            input_model=OpenAIRequest,
            output_model=LLM_Result,
            handler=handler_wrapper,
            event_type=EventType.LLM_CALL
            
        )
        
        
