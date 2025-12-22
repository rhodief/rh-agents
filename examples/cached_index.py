"""
Example demonstrating execution caching and recovery.
Shows how to use FileCacheBackend to cache expensive LLM calls.
"""
import asyncio
from pathlib import Path
from db import DOC_LIST, DOCS
from rh_agents.agents import DoctrineReceverAgent, DoctrineTool, OmniAgent, OpenAILLM, ReviewerAgent, StepExecutorAgent
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.actors import Tool
from rh_agents.core.result_types import Tool_Result
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import EventBus, ExecutionState
from rh_agents.cache_backends import FileCacheBackend, InMemoryCacheBackend
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
            handler=handler,
            # Tools not cached by default (may have side effects)
            cacheable=False
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
            handler=handler,
            # This tool could be cached since it reads static data
            cacheable=True,
            cache_ttl=3600  # Cache for 1 hour
        )


async def run_with_cache(use_file_cache: bool = True):
    """Run the agent system with caching enabled."""
    
    # Choose cache backend
    if use_file_cache:
        cache_dir = Path(".cache/executions")
        cache_backend = FileCacheBackend(cache_dir)
        print(f"📁 Using file cache at: {cache_dir.absolute()}")
    else:
        cache_backend = InMemoryCacheBackend()
        print("💾 Using in-memory cache")
    
    # Create LLM with caching enabled (default)
    llm = OpenAILLM()
    print(f"🤖 LLM cacheable: {llm.cacheable}, TTL: {llm.cache_ttl}s")
    
    # Setup tools
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    # Create agents
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # Create message
    msg = 'Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade.'
    message = Message(content=msg, author=AuthorType.USER)
    
    # Setup event bus with printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus = EventBus()
    bus.subscribe(printer)
    
    # Create execution state WITH cache backend
    agent_execution_state = ExecutionState(
        event_bus=bus,
        cache_backend=cache_backend
    )
    
    async def streamer():
        async for event in bus.event_stream():
            pass  # Events already printed by subscriber
    
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
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
    
    # Print cache statistics
    stats = cache_backend.get_stats()
    print(f"\n{'═' * 60}")
    print(f"{'📊 CACHE STATISTICS':^60}")
    print(f"{'═' * 60}")
    print(f"Backend: {stats['backend']}")
    print(f"Cached entries: {stats['size']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    if 'total_bytes' in stats:
        print(f"Total size: {stats['total_bytes']:,} bytes")
    print(f"{'═' * 60}\n")
    
    # Print summary statistics
    printer.print_summary()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          EXECUTION CACHING DEMONSTRATION                     ║
╚══════════════════════════════════════════════════════════════╝

This example demonstrates the caching system:
- LLM calls are cached by default (TTL: 1 hour)
- Tool calls are NOT cached by default (may have side effects)
- Cached results show [CACHED] in the detail
- Run twice to see the cache in action!

""")
    
    print("Run 1: Initial execution (will cache LLM calls)")
    print("=" * 60)
    asyncio.run(run_with_cache(use_file_cache=True))
    
    print("\n\n")
    print("Run 2: Re-execution (will use cached results)")
    print("=" * 60)
    asyncio.run(run_with_cache(use_file_cache=True))
