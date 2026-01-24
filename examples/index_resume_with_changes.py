"""
Example: Resume from checkpoint with MODIFIED input

This example demonstrates:
1. Loading a checkpoint
2. Modifying the input slightly
3. Re-executing from a specific point with the new input
4. Smart replay: completed steps are skipped, new work executes

Use case: You ran a pipeline, want to adjust the prompt, and re-run 
just the final steps without repeating expensive early steps.

How it works:
- Events BEFORE resume_from_address: SKIPPED (addresses still match)
- Events FROM resume_from_address: RE-EXECUTED (addresses don't match due to input change)
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


# ============================================================================
# CONFIGURATION
# ============================================================================
# Resume from ReviewerAgent to re-run just the final review with modified input
# Early steps (document retrieval, analysis) will be skipped - saves ~2 minutes!
RESUME_FROM_ADDRESS = "OmniAgent::ReviewerAgent::final_review::agent_call"

# MODIFIED MESSAGE - This change will cause events from resume point to re-execute
MODIFIED_MSG = '''Faça um relatório DETALHADO e EXPANDIDO com o resumo combinado dos óbices 
jurídicos da decisão de Admissibilidade e do respectivo Agravo de Recurso Especial 
mostrando em uma tabela correspondência ou não entre os óbices e seus rebatimentos
que constam nos dois documentos. Inclua análise crítica e recomendações.
Utilize as ferramentas disponíveis para buscar as peças necessárias.'''
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
        exit(1)
    
    # Load state_id from file
    with open(state_id_file, "r") as f:
        saved_state_id = f.read().strip()
    
    print(f"\n{'═' * 80}")
    print(f"{'🔄 RESUME WITH MODIFIED INPUT':^80}")
    print(f"{'═' * 80}\n")
    print(f"📋 Loading state: {saved_state_id}")
    print(f"🎯 Resume from: {RESUME_FROM_ADDRESS}")
    print(f"📝 Input has been MODIFIED - events from resume point will RE-EXECUTE")
    print(f"⚡ Events before resume point will still be SKIPPED (save time!)\n")
    
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
        resume_from_address=RESUME_FROM_ADDRESS
    )
    
    if not restored_state:
        print(f"❌ Failed to load state: {saved_state_id}")
        exit(1)
    
    print(f"✅ State loaded successfully!")
    print(f"📊 Total events in history: {len(restored_state.history.get_event_list())}")
    
    # Find position of resume address
    events = restored_state.history.get_event_list()
    for idx, event in enumerate(events):
        addr = event.get('address') if isinstance(event, dict) else getattr(event, 'address', None)
        if addr == RESUME_FROM_ADDRESS:
            print(f"📍 Resume address at position {idx + 1}/{len(events)}")
            print(f"⏭️  Will skip {idx} events before this point")
            break
    
    print()
    
    # Initialize agents (same as before)
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # MODIFIED message - This will cause re-execution from resume point!
    message = Message(content=MODIFIED_MSG, author=AuthorType.USER)
    
    print(f"{'═' * 80}")
    print(f"📝 MODIFIED INPUT:")
    print(f"{'═' * 80}")
    print(f"{MODIFIED_MSG[:150]}...")
    print(f"{'═' * 80}\n")
    
    # Create OmniAgent
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    async def main():
        print(f"{'═' * 80}")
        print(f"{'▶️  REPLAY WITH MODIFIED INPUT STARTED':^80}")
        print(f"{'═' * 80}\n")
        print(f"📝 Early steps (before resume point): SKIPPED instantly")
        print(f"🎬 From '{RESUME_FROM_ADDRESS}': RE-EXECUTED with new input\n")
        
        # Execute the pipeline
        result = await ExecutionEvent[Message](actor=omni_agent)(message, "", restored_state)
        
        print(f"\n{'═' * 80}")
        print(f"{'✅ EXECUTION FINISHED':^80}")
        print(f"{'═' * 80}\n")
        
        # Print summary
        printer.print_summary()
        
        # Show statistics
        all_events = restored_state.history.get_event_list()
        
        print(f"\n{'═' * 80}")
        print(f"{'📊 STATISTICS':^80}")
        print(f"{'═' * 80}\n")
        print(f"📌 Total events now: {len(all_events)}")
        print(f"💾 Storage keys: {len(restored_state.storage.data)}")
        print(f"🎯 Artifacts: {len(restored_state.storage.artifacts)}")
        
        if restored_state._resume_point_reached:
            print(f"\n✅ Resume point was reached")
            print(f"⚡ Benefit: Skipped expensive early steps, only re-ran what changed")
        
        print(f"\n{'═' * 80}")
        print(f"💡 KEY INSIGHT:")
        print(f"By modifying the input and using resume_from_address, you can:")
        print(f"  • Skip expensive early computation (doctrine analysis, document retrieval)")
        print(f"  • Re-execute only the steps affected by your changes")
        print(f"  • Iterate quickly on later stages of your pipeline")
        print(f"{'═' * 80}\n")
    
    asyncio.run(main())
