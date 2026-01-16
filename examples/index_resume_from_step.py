"""
Example: Resume from a specific step (Selective Replay)

This example demonstrates how to use resume_from_address to:
1. Skip all completed steps before a specific address
2. Resume execution from a chosen point in the pipeline
3. Useful for debugging specific steps or re-running failed portions

Usage:
1. Run index_with_checkpoint.py first to create a checkpoint
2. Check the event addresses in the output
3. Modify the RESUME_FROM_ADDRESS below to start from that point
4. Run this script to resume from that specific step

Example addresses you might use:
- "OmniAgent::DoctrineReceverAgent::agent_call" - Start from doctrine receiver
- "OmniAgent::StepExecutorAgent::step_5-8::agent_call" - Start from step 5
- "OmniAgent::ReviewerAgent::final_review::agent_call" - Start from final review
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


# ============================================================================
# CONFIGURATION: Set the address you want to resume from
# ============================================================================
RESUME_FROM_ADDRESS = "OmniAgent::StepExecutorAgent::step_5-8::agent_call"
# Set to None to replay from the beginning

# IMPORTANT: resume_from_address skips TO that address, but if the event is 
# already completed, it will STILL be skipped by replay logic (instant return).
# To force re-execution, you need to either:
#   1. Change the input (so addresses don't match)
#   2. Use VALIDATION mode instead of NORMAL
#   3. Pick an address that wasn't completed yet
# ============================================================================


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
    # Check if checkpoint exists
    state_id_file = Path(".state_store/latest_state_id.txt")
    if not state_id_file.exists():
        print(f"\n{'═' * 80}")
        print(f"{'❌ ERROR: No checkpoint found':^80}")
        print(f"{'═' * 80}\n")
        print(f"Please run index_with_checkpoint.py first to create a checkpoint.")
        print(f"Then run this script to resume from a specific address.\n")
        exit(1)
    
    # Load state_id from file
    with open(state_id_file, "r") as f:
        saved_state_id = f.read().strip()
    
    print(f"\n{'═' * 80}")
    print(f"{'🎯 SELECTIVE RESUME - FROM SPECIFIC ADDRESS':^80}")
    print(f"{'═' * 80}\n")
    print(f"📋 Loading state: {saved_state_id}")
    
    if RESUME_FROM_ADDRESS:
        print(f"🎯 Resume from address: {RESUME_FROM_ADDRESS}")
        print(f"⚡ All events before this point will be SKIPPED")
        print(f"🔄 Execution will continue from this address forward")
    else:
        print(f"🎯 Resume mode: Full replay (no skip point set)")
    
    print()
    
    # Initialize backends
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    # Create event bus with printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    
    # LOAD SAVED STATE WITH RESUME ADDRESS
    restored_state = ExecutionState.load_from_state_id(
        state_id=saved_state_id,
        state_backend=state_backend,
        artifact_backend=artifact_backend,
        event_bus=bus,
        replay_mode=ReplayMode.NORMAL,
        resume_from_address=RESUME_FROM_ADDRESS  # ← This skips to the specified address
    )
    
    if not restored_state:
        print(f"❌ Failed to load state: {saved_state_id}")
        exit(1)
    
    print(f"✅ State loaded successfully!")
    print(f"📊 Total events in history: {len(restored_state.history.get_event_list())}")
    
    if RESUME_FROM_ADDRESS:
        # Find position of resume address
        events = restored_state.history.get_event_list()
        for idx, event in enumerate(events):
            addr = event.get('address') if isinstance(event, dict) else getattr(event, 'address', None)
            if addr == RESUME_FROM_ADDRESS:
                print(f"📍 Resume address found at position {idx + 1}/{len(events)}")
                print(f"⏭️  Skipping {idx} events before this point")
                break
    
    print()
    
    # Initialize agents (same as before)
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # Same message as original
    msg = 'Faça um relatório com o resumo combinado dos óbices jurídicos da decisão de Admissibilidade e do respectivo Agravo de Recurso Especial mostrando em uma tabela correspondência ou não entre os óbices e seus rebatimentosque constam nos dois documentos. Utilize as ferramentas disponíveis para buscar as peças necessárias.'
    message = Message(content=msg, author=AuthorType.USER)
    
    # Create OmniAgent
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    async def main():
        print(f"{'═' * 80}")
        print(f"{'▶️  SELECTIVE REPLAY STARTED':^80}")
        print(f"{'═' * 80}\n")
        
        if RESUME_FROM_ADDRESS:
            print(f"📝 Events BEFORE '{RESUME_FROM_ADDRESS}' will be skipped instantly")
            print(f"🎬 Execution begins FROM this address\n")
        else:
            print(f"📝 Full replay from the beginning\n")
        
        # Execute the pipeline (will skip to resume address if set)
        result = await ExecutionEvent[Message](actor=omni_agent)(message, "", restored_state)
        
        print(f"\n{'═' * 80}")
        print(f"{'✅ SELECTIVE REPLAY FINISHED':^80}")
        print(f"{'═' * 80}\n")
        
        # Print summary
        printer.print_summary()
        
        # Show detailed statistics
        all_events = restored_state.history.get_event_list()
        
        print(f"\n{'═' * 80}")
        print(f"{'📊 SELECTIVE REPLAY STATISTICS':^80}")
        print(f"{'═' * 80}\n")
        print(f"📌 Total events in history: {len(all_events)}")
        
        if RESUME_FROM_ADDRESS and restored_state._resume_point_reached:
            print(f"✅ Resume point was reached")
            print(f"⚡ Events before resume point: SKIPPED (instant)")
            print(f"🔄 Events from resume point: RE-EXECUTED")
        
        print(f"💾 Storage keys: {len(restored_state.storage.data)}")
        print(f"🎯 Artifacts: {len(restored_state.storage.artifacts)}")
        
        print(f"\n{'═' * 80}")
        print(f"💡 TIP: To resume from a different step:")
        print(f"   1. Look at the event addresses in the output above")
        print(f"   2. Copy an address like 'OmniAgent::StepExecutorAgent::step_X-Y::agent_call'")
        print(f"   3. Set RESUME_FROM_ADDRESS at the top of this script")
        print(f"   4. Run again to skip to that specific step")
        print(f"{'═' * 80}\n")
    
    asyncio.run(main())
