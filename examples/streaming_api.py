"""
FastAPI example for agent execution with streaming events.

This example demonstrates how to use EventStreamer for SSE (Server-Sent Events)
streaming with minimal boilerplate. EventStreamer works just like EventPrinter -
simply plug it into the event bus!

Key simplification:
- Before: 60+ lines of queue management, async generator logic, error handling
- After: 4 lines - create streamer, subscribe to bus, start task, return stream

See STREAMING_SIMPLE.md for detailed comparison and usage guide.
"""
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from db import DOC_LIST, DOCS

load_dotenv(Path(__file__).parent.parent / ".env")

from rh_agents.agents import (
    DoctrineReceverAgent, 
    DoctrineTool, 
    OmniAgent, 
    OpenAILLM, 
    ReviewerAgent, 
    StepExecutorAgent
)
from rh_agents import Tool, Tool_Result, ExecutionEvent, ExecutionState, Message, AuthorType
from rh_agents import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.bus_handlers import EventStreamer


# === Tool Definitions ===

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
        
        super().__init__(
            name="lista_pecas_por_tipo",
            description="Obtém uma lista de peças (nome e id) baseado no tipo_peca",
            input_model=ListPecasArgs,
            handler=handler,
            cacheable=False
        )


class GetTextoPecaTool(Tool):
    def __init__(self) -> None:
        async def handler(args: GetTextoPecaArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOCS.get(args.id_peca, "Peça não encontrada.")
            return Tool_Result(output=result, tool_name="get_texto_peca")
        
        super().__init__(
            name="get_texto_peca",
            description="Obtém o texto completo de uma peça judicial",
            input_model=GetTextoPecaArgs,
            handler=handler,
            cacheable=True,
            cache_ttl=3600
        )


# === FastAPI App ===

app = FastAPI(title="RH Agents API")

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


@app.get("/")
async def root():
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return {"status": "ok"}


@app.post("/api/stream")
async def stream_execution(request: QueryRequest):
    """
    Stream agent execution events using Server-Sent Events (SSE).
    """
    # Setup LLM and tools
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    
    # Create agents
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    
    # Setup state backend for persistence (if caching enabled)
    state_backend = FileSystemStateBackend(".state_store") if request.use_cache else None
    
    # Create event bus with SSE streamer (just like EventPrinter!)
    bus = EventBus()
    streamer = EventStreamer(include_cache_stats=True)
    bus.subscribe(streamer)
    
    # Create execution state
    agent_execution_state = ExecutionState(event_bus=bus, state_backend=state_backend)
    
    # Create OmniAgent
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    # Create message
    message = Message(content=request.query, author=AuthorType.USER)
    
    # Start execution task
    execution_task = asyncio.create_task(
        ExecutionEvent(actor=omni_agent)(message, "", agent_execution_state)
    )
    
    # Return streaming response - streamer handles all the complexity!
    return StreamingResponse(
        streamer.stream(execution_task=execution_task, cache_backend=cache_backend),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = 8002
    if len(sys.argv) > 2 and sys.argv[1] == '--port':
        try:
            port = int(sys.argv[2])
        except ValueError:
            pass
    
    print(f"\nStarting FastAPI server on http://localhost:{port}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
