"""
FastAPI example for agent execution with streaming events.
Minimal setup - customize as needed.
"""
import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator
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
from rh_agents.core.actors import Tool
from rh_agents.core.result_types import Tool_Result
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import EventBus, ExecutionState
from rh_agents.cache_backends import FileCacheBackend
from rh_agents.models import AuthorType, Message


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
    
    # Setup cache backend
    cache_backend = FileCacheBackend(cache_dir=".cache/executions") if request.use_cache else None
    
    # Create event bus and queue for real-time streaming
    bus = EventBus()
    event_queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()

    async def queue_handler(event: ExecutionEvent):
        await event_queue.put(event)

    bus.subscribe(queue_handler)
    
    # Create execution state
    agent_execution_state = ExecutionState(event_bus=bus, cache_backend=cache_backend)
    
    # Create OmniAgent
    omni_agent = OmniAgent(
        receiver_agent=doctrine_receiver_agent,
        step_executor_agent=step_executor_agent,
        reviewer_agent=reviewer_agent
    )
    
    # Create message
    message = Message(content=request.query, author=AuthorType.USER)
    
    async def run_execution(
        omni_agent: OmniAgent,
        message: Message,
        state: ExecutionState,
    ):
        await ExecutionEvent[Message](actor=omni_agent)(
            message, "", state
        )
    
    async def event_generator() -> AsyncGenerator[str, None]:
        # Yield immediately to start the HTTP streaming response
        yield ": stream-start\n\n"

        execution_task = asyncio.create_task(
            run_execution(
                omni_agent=omni_agent,
                message=message,
                state=agent_execution_state,
            )
        )

        try:
            while True:
                # Exit condition
                if execution_task.done() and event_queue.empty():
                    break

                try:
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=0.25,
                    )
                    yield f"data: {event.model_dump_json()}\n\n"

                except asyncio.TimeoutError:
                    # heartbeat (forces flush)
                    yield ": keep-alive\n\n"

            # Propagate execution errors if any
            await execution_task

            final_event = {
                "event_type": "complete",
                "message": "Execution completed successfully",
            }

            if cache_backend:
                final_event["cache_stats"] = cache_backend.get_stats()
            yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as e:
            error_event = {
                "event_type": "error",
                "message": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
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
