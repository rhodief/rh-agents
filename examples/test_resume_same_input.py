"""
Test: Resume with SAME input to verify re-execution
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
from rh_agents.core.state_recovery import ReplayMode
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message
from pathlib import Path


class ListPecasArgs(BaseModel):
    processo: int = Field(..., description="Número do processo judicial")
    tipo_peca: str = Field(..., description="Tipo da peça judicial, ex: DEC_ADM, ARESP")

class GetTextoPecaArgs(BaseModel):
    id_peca: str = Field(..., description="ID da peça")

class ListPecasTool(Tool):
    def __init__(self) -> None:
        async def handler(args: ListPecasArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOC_LIST.get(args.tipo_peca, [])
            return Tool_Result(output=result, tool_name="lista_pecas_por_tipo")
        super().__init__(name="lista_pecas_por_tipo", description="...", input_model=ListPecasArgs, handler=handler)
    
class GetTextoPecaTool(Tool):
    def __init__(self) -> None:
        async def handler(args: GetTextoPecaArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOCS.get(args.id_peca, "Peça não encontrada.")
            return Tool_Result(output=result, tool_name="get_texto_peca")
        super().__init__(name="get_texto_peca", description="...", input_model=GetTextoPecaArgs, handler=handler)


if __name__ == "__main__":
    state_id_file = Path(".state_store/latest_state_id.txt")
    if not state_id_file.exists():
        print("❌ Run index_with_checkpoint.py first")
        exit(1)
    
    with open(state_id_file, "r") as f:
        saved_state_id = f.read().strip()
    
    print(f"\n{'═' * 80}")
    print(f"{'🧪 TEST: Resume with SAME input (no changes)':^80}")
    print(f"{'═' * 80}\n")
    print(f"📋 State: {saved_state_id}")
    print(f"🎯 Resume from: OmniAgent::ReviewerAgent::final_review::agent_call")
    print(f"📝 Input: UNCHANGED (same as original checkpoint)")
    print(f"❓ Question: Will ReviewerAgent re-execute or skip?\n")
    
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    
    restored_state = ExecutionState.load_from_state_id(
        state_id=saved_state_id,
        state_backend=state_backend,
        artifact_backend=artifact_backend,
        event_bus=bus,
        replay_mode=ReplayMode.NORMAL,
        resume_from_address="OmniAgent::ReviewerAgent::final_review::agent_call"
    )
    
    print(f"✅ State loaded: {len(restored_state.history.get_event_list())} events\n")
    
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # SAME message as original (no modification!)
    msg = 'Faça um relatório com o resumo combinado dos óbices jurídicos da decisão de Admissibilidade e do respectivo Agravo de Recurso Especial mostrando em uma tabela correspondência ou não entre os óbices e seus rebatimentosque constam nos dois documentos. Utilize as ferramentas disponíveis para buscar as peças necessárias.'
    message = Message(content=msg, author=AuthorType.USER)
    
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    async def main():
        print(f"{'═' * 80}")
        print(f"{'▶️  EXECUTING':^80}")
        print(f"{'═' * 80}\n")
        
        result = await ExecutionEvent[Message](actor=omni_agent)(message, "", restored_state)
        
        print(f"\n{'═' * 80}")
        print(f"{'✅ FINISHED':^80}")
        print(f"{'═' * 80}\n")
        
        printer.print_summary()
        
        print(f"\n{'═' * 80}")
        print(f"{'📊 RESULT':^80}")
        print(f"{'═' * 80}\n")
        
        # Check the summary statistics  
        all_events = restored_state.history.get_event_list()
        completed_count = sum(1 for e in all_events if (isinstance(e, dict) and e.get('status') == 'completed') or 
                              (hasattr(e, 'status') and str(e.status).endswith('COMPLETED')))
        
        if printer.total_execution_time > 0:
            print(f"✅ EVENTS RE-EXECUTED!")
            print(f"⏱️  Execution time: {printer.total_execution_time:.2f}s")
            print(f"📊 Total events: {printer.total_events}")
            print(f"📊 Events in history: {len(all_events)}")
            print(f"\n💡 ANSWER: YES! resume_from_address FORCES re-execution")
            print(f"   even when input is UNCHANGED.")
            print(f"\n🔑 Why? The _resume_point_reached flag makes should_skip_event()")
            print(f"   return False, forcing fresh execution from that point forward.")
        else:
            print(f"❌ NO EVENTS EXECUTED (all skipped)")
            print(f"\n💡 ANSWER: NO. resume_from_address does NOT force re-execution")
            print(f"   when input is unchanged.")
        
        print(f"\n{'═' * 80}\n")
    
    asyncio.run(main())
