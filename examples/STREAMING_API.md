# RH Agents Streaming API

A FastAPI application that streams agent execution events in real-time using Server-Sent Events (SSE).

## 🎯 New in This Version: EventStreamer

This example now uses the new `EventStreamer` class that dramatically simplifies SSE streaming!

**Instead of 60+ lines of boilerplate**, you now need just **4 lines**:

```python
streamer = EventStreamer(include_cache_stats=True)
bus.subscribe(streamer)
execution_task = asyncio.create_task(...)
return StreamingResponse(streamer.stream(execution_task, cache_backend))
```

👉 See [STREAMING_SIMPLE.md](./STREAMING_SIMPLE.md) for a detailed before/after comparison.

## Features

- 🚀 Real-time event streaming over HTTP
- 📊 Live cache statistics
- 🎨 Beautiful web interface for testing
- 📚 Full Swagger/OpenAPI documentation
- 💾 Support for file-based and in-memory caching

## Quick Start

### 1. Install Dependencies

```bash
# Install FastAPI and uvicorn if not already installed
pip install fastapi uvicorn[standard]

# Or use the requirements.txt
pip install -r ../requirements.txt
```

### 2. Set Up Environment

Make sure you have a `.env` file in the parent directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Server

```bash
# From the examples directory
python streaming_api.py

# Or use uvicorn directly
uvicorn streaming_api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Enter your query in the text area
3. Toggle caching if needed
4. Click "🚀 Start Execution"
5. Watch the events stream in real-time!

## Using the Swagger UI

1. Open http://localhost:8000/docs
2. Navigate to `POST /api/stream`
3. Click "Try it out"
4. Enter your request body:
   ```json
   {
     "query": "Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade.",
     "use_cache": true
   }
   ```
5. Click "Execute"
6. Watch the events stream in the response body

## API Endpoints

### POST /api/stream

Stream agent execution events in real-time.

**Request Body:**
```json
{
  "query": "Your query here",
  "use_cache": true
}
```

**Response:**
Server-Sent Events stream with the following event types:
- `start`: Execution started
- `agent_start`: Agent started processing
- `agent_complete`: Agent completed processing
- `llm_start`: LLM call started
- `llm_complete`: LLM call completed
- `tool_start`: Tool execution started
- `tool_complete`: Tool execution completed
- `complete`: Final result with execution summary
- `error`: Error occurred during execution

**Example Event:**
```
data: {"event_type": "llm_start", "timestamp": "12:34:56", "address": "OmniAgent", "detail": "Calling LLM...", "actor_name": "OpenAILLM", "cached": false}

data: {"event_type": "complete", "result": "Final result here", "message": "Agent execution completed successfully", "cache_stats": {"backend": "FileCacheBackend", "size": 5, "hits": 3, "misses": 2, "hit_rate": 0.6}}
```

### GET /

Serves the web interface for testing the streaming API.

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "rh-agents-streaming-api"
}
```

## Event Stream Format

Each event in the stream follows the Server-Sent Events (SSE) format:

```
data: <JSON object>\n\n
```

The JSON object contains:
- `event_type`: Type of event (start, agent_start, llm_complete, etc.)
- `timestamp`: When the event occurred
- `address`: Which component generated the event
- `detail`: Detailed description of what happened
- `actor_name`: Name of the actor that executed
- `cached`: Whether the result was cached (boolean)

The final event includes:
- `result`: The final execution result
- `cache_stats`: Statistics about cache usage (if caching enabled)

## Testing with curl

```bash
curl -N -X POST "http://localhost:8000/api/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade.",
    "use_cache": true
  }'
```

## Testing with JavaScript

```javascript
const response = await fetch('/api/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Your query here',
    use_cache: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log('Event:', data);
    }
  }
}
```

## Architecture

The streaming API works by:

1. Accepting a POST request with a query
2. Creating an event bus to capture agent execution events
3. Executing the agent system asynchronously
4. Streaming events as they occur using Server-Sent Events
5. Sending the final result and cache statistics at the end

The system uses the same agent setup as `cached_index.py` but replaces the `EventPrinter` with a streaming handler that formats events for HTTP transmission.

## Caching

- **LLM calls**: Cached by default (TTL: 1 hour)
- **Tool calls**: Most are NOT cached by default (may have side effects)
- **get_texto_peca tool**: Cached (TTL: 1 hour) since it reads static data
- **Cache backend**: File-based cache stored in `.cache/executions/`

## Troubleshooting

### Events not streaming in Swagger UI

Some browsers or proxies may buffer SSE streams. Try:
- Using the web interface at http://localhost:8000 instead
- Testing with curl or a dedicated HTTP client
- Disabling any proxies or buffering middleware

### Import errors

Make sure you're running from the `examples` directory and have installed all dependencies:

```bash
cd examples
pip install -r ../requirements.txt
python streaming_api.py
```

### Port already in use

If port 8000 is already in use, change the port:

```bash
uvicorn streaming_api:app --port 8001
```

## Development

To run in development mode with auto-reload:

```bash
uvicorn streaming_api:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

For production, consider:
- Using a production ASGI server (uvicorn with workers)
- Adding authentication/authorization
- Rate limiting
- CORS configuration for frontend domains
- Nginx or similar reverse proxy for buffering control

```bash
uvicorn streaming_api:app --host 0.0.0.0 --port 8000 --workers 4
```
