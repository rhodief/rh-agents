from typing import Callable, Union
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.types import EventType
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import ExecutionState
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message


class DoctrineStep(BaseModel):    
    index: int
    description: str
    feasible: bool
    required_steps: list[int] = Field(default_factory=list)
    
class Doctrine(BaseModel):
    goal: str
    constraints: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)
    steps: list[DoctrineStep] = Field(default_factory=list)
    

class DoctrineReceverAgent(Agent):
    def __init__(self,
                 llm: LLM,
                 tools: Union[list[Tool], None] = None
                 ) -> None:
        INTENT_PARSER_PROMPT = '''
        Você é um analisador de intenções. 
        Use a ferramenta disponível para estruturar o pedido do usuário.
        Considere que cada passo será passado para um subagente, utilize linguagem clara e objetiva.
    '''
    
        async def handler(input_data: Message, execution_state: ExecutionState) -> Union[Doctrine, Message]:
            llm_event = ExecutionEvent[llm.output_model](
                actor=llm
            )            
            # Execute LLM to parse the user input into a Doctrine
            execution_result = await llm_event(input_data, {}, execution_state)
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            result = execution_result.result
            if result.content:
                return Message(content=result.content, author=AuthorType.ASSISTANT)
            return Doctrine(**result.model_dump())
            
        
        super().__init__(
            name="DoctrineReceverAgent",
            description=INTENT_PARSER_PROMPT,
            input_model=Message,
            output_model=Doctrine,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools=tools) if tools is not None else ToolSet()
        )
