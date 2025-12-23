# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Web Browser │  │  Swagger UI  │  │ HTTP Client  │    │
│  │  (index.html)│  │   (/docs)    │  │  (curl/code) │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │                                │
│                    HTTP POST /api/stream                    │
│                  { query, use_cache }                       │
└────────────────────────────┼───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          StreamingResponse (SSE)                     │  │
│  │  - Yields Server-Sent Events in real-time           │  │
│  │  - Format: data: {json}\n\n                         │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          stream_agent_execution()                    │  │
│  │  - Sets up agents, tools, event bus                 │  │
│  │  - Captures events from execution                   │  │
│  │  - Streams events as SSE                            │  │
│  └──────────────────┬───────────────────────────────────┘  │
└────────────────────┼────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENT SYSTEM                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 OmniAgent                            │  │
│  │  Orchestrates the entire execution flow             │  │
│  └──────────┬─────────────┬─────────────┬───────────────┘  │
│             │             │             │                   │
│             ▼             ▼             ▼                   │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐        │
│  │  Doctrine    │ │   Step     │ │  Reviewer    │        │
│  │  Receiver    │ │  Executor  │ │   Agent      │        │
│  │   Agent      │ │   Agent    │ │              │        │
│  └──────┬───────┘ └─────┬──────┘ └──────┬───────┘        │
│         │               │               │                  │
│         └───────────────┼───────────────┘                  │
│                         │                                  │
│                         ▼                                  │
│           ┌─────────────────────────┐                     │
│           │      Event Bus          │                     │
│           │  - agent_start          │                     │
│           │  - agent_complete       │                     │
│           │  - llm_start            │                     │
│           │  - llm_complete         │                     │
│           │  - tool_start           │                     │
│           │  - tool_complete        │                     │
│           └───────┬─────────────────┘                     │
└───────────────────┼────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  EXECUTION LAYER                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   OpenAI     │  │    Tools     │  │    Cache     │    │
│  │     LLM      │  │  - Doctrine  │  │   Backend    │    │
│  │              │  │  - ListPecas │  │ (File/Memory)│    │
│  │  cacheable=T │  │  - GetTexto  │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

## Event Flow

```
1. Client Request
   ↓
2. FastAPI receives POST /api/stream
   ↓
3. Create EventBus, CacheBackend, Agents
   ↓
4. Start SSE streaming response
   ↓
5. Execute OmniAgent
   ├→ DoctrineReceiverAgent
   │  ├→ LLM Call (may be cached)
   │  └→ Tool Calls
   ├→ StepExecutorAgent
   │  ├→ LLM Call (may be cached)
   │  └→ Tool Calls
   └→ ReviewerAgent
      └→ LLM Call (may be cached)
   ↓
6. Each action emits events to EventBus
   ↓
7. Events are captured and formatted as SSE
   ↓
8. Events streamed to client in real-time
   ↓
9. Final result + cache stats sent
   ↓
10. Stream closes
```

## Caching Flow

```
LLM Call Request
   ↓
Check Cache Backend
   ↓
   ├─ HIT? ──→ Return cached result (fast!)
   │           Event: cached=true
   │
   └─ MISS ──→ Call OpenAI API (slow)
               Store in cache
               Event: cached=false
```

## SSE Message Format

```
data: {
  "event_type": "llm_complete",
  "timestamp": "14:23:45",
  "address": "OmniAgent/DoctrineReceiverAgent",
  "detail": "Generated response with 250 tokens",
  "actor_name": "OpenAILLM",
  "cached": true
}

data: {
  "event_type": "complete",
  "result": "Final execution result...",
  "message": "Agent execution completed successfully",
  "cache_stats": {
    "backend": "FileCacheBackend",
    "size": 5,
    "hits": 3,
    "misses": 2,
    "hit_rate": 0.6
  }
}
```

## Key Components

### Frontend Layer
- **Web UI**: Beautiful interface with real-time event display
- **Swagger UI**: Interactive API documentation
- **HTTP Clients**: curl, Python requests, fetch API, etc.

### API Layer (FastAPI)
- **Endpoint**: POST /api/stream
- **Response**: StreamingResponse with SSE
- **Format**: Server-Sent Events (text/event-stream)

### Agent Layer
- **OmniAgent**: Main orchestrator
- **DoctrineReceiverAgent**: Initial processing with doctrine tool
- **StepExecutorAgent**: Executes planned steps
- **ReviewerAgent**: Reviews and validates results

### Execution Layer
- **EventBus**: Captures and distributes events
- **CacheBackend**: Stores and retrieves cached results
- **OpenAI LLM**: Makes API calls (cacheable)
- **Tools**: Execute domain-specific actions

## Comparison: Terminal vs HTTP Streaming

### cached_index.py (Terminal)
```
EventPrinter ──→ stdout (terminal)
- Run locally only
- No remote access
- Manual inspection
- No API interface
```

### streaming_api.py (HTTP)
```
EventBus ──→ SSE Stream ──→ HTTP Response ──→ Client
- Access from anywhere
- Web interface
- Swagger documentation
- Programmatic access
- Real-time updates
```

## Benefits

✅ **Real-time visibility**: See execution as it happens
✅ **Remote access**: Test from any device
✅ **API documentation**: Auto-generated Swagger UI
✅ **Cache monitoring**: See hit rates and performance
✅ **Error handling**: Graceful error reporting
✅ **Client flexibility**: Use any HTTP client
✅ **Production ready**: Standard HTTP/SSE protocol
