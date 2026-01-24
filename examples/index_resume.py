"""
Example: Resume pipeline from checkpoint (Smart Replay Demo)

This example demonstrates:
1. Loading a previously saved execution state
2. Smart replay: automatically skipping already-executed events
3. Executing only the new/modified steps

Expected behavior:
- DoctrineReceiver step: SKIPPED (returns cached result instantly)
- StepExecutor step: SKIPPED (returns cached result instantly)  
- Reviewer step: RE-EXECUTED (simulated by changing input slightly)

This showcases the state recovery system's ability to:
- Resume execution from any point
- Avoid redundant computation
- Enable iterative development and debugging

Prerequisites: Run index_with_checkpoint.py first to create the checkpoint
"""
import asyncio
from db import DOC_LIST, DOCS
from rh_agents.agents import DoctrineReceverAgent, DoctrineTool, OmniAgent, OpenAILLM, ReviewerAgent, StepExecutorAgent
from rh_agents import (
    EventPrinter,
    Tool,
    Tool_Result,
    ExecutionEvent,
    ExecutionState,
    FileSystemStateBackend,
    FileSystemArtifactBackend,
    ReplayMode,
    Message,
    AuthorType
)
from pydantic import BaseModel, Field
from pathlib import Path


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
        print(f"Then run this script to resume from that checkpoint.\n")
        exit(1)
    
    # Load state_id from file
    with open(state_id_file, "r") as f:
        saved_state_id = f.read().strip()
    
    print(f"\n{'═' * 80}")
    print(f"{'🔄 RESUMING FROM CHECKPOINT':^80}")
    print(f"{'═' * 80}\n")
    print(f"📋 Loading state: {saved_state_id}")
    print(f"🎯 Mode: Smart Replay (skip completed events)")
    print(f"💡 This demonstrates state recovery with automatic event skipping\n")
    
    # OPTIONAL: Resume from a specific address (skip everything before this point)
    # Example addresses from a typical run:
    # - None: Replay from the beginning (default)
    # - "OmniAgent::StepExecutorAgent::step_5-8::agent_call": Skip to step 5
    # - "OmniAgent::ReviewerAgent::final_review::agent_call": Skip to reviewer
    resume_from = None  # Change this to an address to skip to that point
    
    if resume_from:
        print(f"🎯 Resume point: {resume_from}")
        print(f"   All events before this address will be skipped\n")
    
    # Initialize backends
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    # Create event bus with printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    
    # LOAD SAVED STATE
    restored_state = ExecutionState.load_from_state_id(
        state_id=saved_state_id,
        state_backend=state_backend,
        artifact_backend=artifact_backend,
        event_bus=bus,
        replay_mode=ReplayMode.NORMAL,  # Skip completed events
        resume_from_address=resume_from  # Start from specific address
    )
    
    if not restored_state:
        print(f"❌ Failed to load state: {saved_state_id}")
        exit(1)
    
    print(f"✅ State loaded successfully!")
    print(f"📊 Restored {len(restored_state.history.get_event_list())} events")
    print(f"💾 Restored {len(restored_state.storage.data)} storage keys")
    print(f"🎯 Restored {len(restored_state.storage.artifacts)} artifacts\n")
    
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
        print(f"{'▶️  REPLAY EXECUTION STARTED':^80}")
        print(f"{'═' * 80}\n")
        print(f"📝 Watch for [REPLAYED] markers - these steps return cached results")
        print(f"⚡ Replayed steps execute in ~0ms (instant result retrieval)")
        print(f"🔄 New steps execute normally\n")
        
        # Execute the pipeline (EventPrinter subscriber handles all event printing)
        result = await ExecutionEvent(actor=omni_agent)(message, "", restored_state)
        
        print(f"\n{'═' * 80}")
        print(f"{'✅ REPLAY EXECUTION FINISHED':^80}")
        print(f"{'═' * 80}\n")
        
        # Print summary statistics
        printer.print_summary()
        
        # Show replay statistics
        all_events = restored_state.history.get_event_list()
        replayed_events = [e for e in all_events if isinstance(e, dict) and e.get('is_replayed') or 
                          hasattr(e, 'is_replayed') and e.is_replayed]
        new_events = len(all_events) - len(replayed_events)
        
        print(f"\n{'═' * 80}")
        print(f"{'📊 REPLAY STATISTICS':^80}")
        print(f"{'═' * 80}\n")
        print(f"📌 Total events in history: {len(all_events)}")
        print(f"⚡ Events skipped (replayed): {len(replayed_events)}")
        print(f"🆕 Events executed fresh: {new_events}")
        print(f"💾 Storage keys: {len(restored_state.storage.data)}")
        print(f"🎯 Artifacts: {len(restored_state.storage.artifacts)}")
        
        if len(replayed_events) > 0:
            time_saved = len(replayed_events) * 2  # Assume ~2s per event
            print(f"\n⏱️  Estimated time saved by replay: ~{time_saved} seconds")
            print(f"🎉 State recovery enabled instant result retrieval for completed steps!")
        
        print(f"\n{'═' * 80}\n")
        
        print('RESULT OF THE FINAL STEP:')
        print(result)
    
    asyncio.run(main())
