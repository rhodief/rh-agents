import asyncio
from db import DOC_LIST, DOCS
from rh_agents.agents import DoctrineReceverAgent, DoctrineTool, OmniAgent, OpenAILLM, ReviewerAgent, StepExecutorAgent
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.actors import Tool
from rh_agents.core.result_types import Tool_Result
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import EventBus, ExecutionState
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message



class ListPecasArgs(BaseModel):
    processo: int = Field(..., description="Número do processo judicial")
    tipo_peca: str = Field(..., description="Tipo da peça judicial, ex: DEC_ADM, ARESP")

class GetTextoPecaArgs(BaseModel):
    id_peca: str = Field(..., description="ID da peça")


 
class ListPecasTool(Tool):
    def __init__(self) -> None:
        LISTA_PECAS_TOOL_PROMPT = '''
        Obtém uma lista de peças (nome e id) baseado no tipo_peca que pode ser: DEC_ADM para decisão de admissibilidade e ARESP para agravo em recurso especial
        '''
        
        async def handler(args: ListPecasArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOC_LIST.get(args.tipo_peca, [])
            return Tool_Result(output=result, tool_name="lista_pecas_por_tipo")
        
        super().__init__(
            name="lista_pecas_por_tipo",
            description=LISTA_PECAS_TOOL_PROMPT,
            input_model=ListPecasArgs,
            handler=handler
        )
    
class GetTextoPecaTool(Tool):
    def __init__(self) -> None:
        GET_TEXTO_PECA_TOOL_PROMPT = '''
        Obtém o texto completo e alguns metadados de uma peça judicial baseado no id da peça
        Utilize essa ferramenta quando precisar acessar o inteiro teor do texto para alguma análise ou sumarização.
        '''
        
        async def handler(args: GetTextoPecaArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOCS.get(args.id_peca, "Peça não encontrada.")
            return Tool_Result(output=result, tool_name="get_texto_peca")
        
        super().__init__(
            name="get_texto_peca",
            description=GET_TEXTO_PECA_TOOL_PROMPT,
            input_model=GetTextoPecaArgs,
            handler=handler
        )
if __name__ == "__main__":
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    msg = 'Faça um relatório para a Análise da Admissibilidade Cotejada de modo a extrair os óbices jurídicos da decisão de Admissibilidade e verificar o respectivo rebatimento no agravo de Recurso Especial correspondente'
    message = Message(content=msg, author=AuthorType.USER)
    
    # Create beautiful event printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    
    bus = EventBus()
    bus.subscribe(printer)  # Use the beautiful printer
    agent_execution_state = ExecutionState(event_bus=bus)
    
    async def streamer():
        async for event in bus.event_stream():
            pass  # Events already printed by subscriber
    
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
        
    
    async def main():
        print(f"\n{'═' * 60}")
        print(f"{'🚀 EXECUTION STARTED':^60}")
        print(f"{'═' * 60}\n")
        
        await asyncio.gather(
            ExecutionEvent[Message](actor=omni_agent)(message, "", agent_execution_state),
            streamer()        
        )
        
        print(f"\n{'═' * 60}")
        print(f"{'✅ EXECUTION FINISHED':^60}")
        print(f"{'═' * 60}\n")
        
        # Print summary statistics
        printer.print_summary()
    
    asyncio.run(main())
        