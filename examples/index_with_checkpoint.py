"""
Example: Execute pipeline with state checkpoint

This example demonstrates:
1. Running a multi-agent pipeline (DoctrineReceiver -> StepExecutor -> Reviewer)
2. Saving execution state as a checkpoint after step execution
3. The saved state can be restored to resume execution later

The checkpoint includes:
- Complete execution history (all events)
- Storage data (intermediate results)
- Artifacts (large objects like LLM responses)
- Execution stack and context

Next: See index_resume.py to restore this checkpoint and continue execution
"""
import asyncio
from db import DOC_LIST, DOCS
from rh_agents.agents import DoctrineReceverAgent, DoctrineTool, OmniAgent, OpenAILLM, ReviewerAgent, StepExecutorAgent
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.actors import Tool
from rh_agents.core.result_types import Tool_Result
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import EventBus, ExecutionState
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.state_recovery import StateStatus, StateMetadata
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
    # Initialize LLM and tools
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    # Create agents
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # User message
    msg = 'Faça um relatório com o resumo combinado dos óbices jurídicos da decisão de Admissibilidade e do respectivo Agravo de Recurso Especial mostrando em uma tabela correspondência ou não entre os óbices e seus rebatimentosque constam nos dois documentos. Utilize as ferramentas disponíveis para buscar as peças necessárias.'
    message = Message(content=msg, author=AuthorType.USER)
    
    # Create event bus with printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    
    # STATE RECOVERY: Initialize backends for state persistence
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    # Create execution state with state recovery enabled
    agent_execution_state = ExecutionState(
        event_bus=bus,
        state_backend=state_backend,
        artifact_backend=artifact_backend
    )
    
    # Create OmniAgent
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    async def main():
        print(f"\n{'═' * 80}")
        print(f"{'🚀 EXECUTION WITH CHECKPOINT':^80}")
        print(f"{'═' * 80}\n")
        print(f"📝 This run will execute the complete pipeline and save a checkpoint")
        print(f"💾 State will be saved to: .state_store/")
        print(f"🔄 Next: Run index_resume.py to restore and continue\n")
        
        # Execute the pipeline (EventPrinter subscriber handles all event printing)
        result = await ExecutionEvent[Message](actor=omni_agent)(message, "", agent_execution_state)
        
        print(f"\n{'═' * 80}")
        print(f"{'💾 SAVING CHECKPOINT':^80}")
        print(f"{'═' * 80}\n")
        
        # Save checkpoint after complete execution
        saved = agent_execution_state.save_checkpoint(
            status=StateStatus.COMPLETED,
            metadata=StateMetadata(
                tags=["omni-agent", "complete", "example"],
                description="Complete pipeline execution - ready for resume",
                pipeline_name="doctrine_analysis",
                custom={
                    "user_message": msg[:100],
                    "agents_executed": ["doctrine_receiver", "step_executor", "reviewer"],
                    "total_events": len(agent_execution_state.history.get_event_list())
                }
            )
        )
        
        if saved:
            print(f"✅ Checkpoint saved successfully!")
            print(f"📋 State ID: {agent_execution_state.state_id}")
            print(f"📊 Total events: {len(agent_execution_state.history.get_event_list())}")
            print(f"💾 Storage keys: {len(agent_execution_state.storage.data)}")
            print(f"🎯 Artifacts: {len(agent_execution_state.storage.artifacts)}")
            print(f"\n📝 To resume this execution, run:")
            print(f"   python examples/index_resume.py")
            print(f"\n💡 The resume will skip already-completed steps and execute only new work.")
            
            # Save state_id to file for easy resumption
            with open(".state_store/latest_state_id.txt", "w") as f:
                f.write(agent_execution_state.state_id)
            print(f"📄 State ID saved to: .state_store/latest_state_id.txt")
        else:
            print(f"❌ Failed to save checkpoint")
        
        print(f"\n{'═' * 80}")
        print(f"{'✅ EXECUTION FINISHED':^80}")
        print(f"{'═' * 80}\n")
        
        # Print summary statistics
        printer.print_summary()
    
    asyncio.run(main())
