# FastAPI Streaming Example - Quick Start Guide

## What Was Created

I've created a complete FastAPI application that streams agent execution events over HTTP in real-time using Server-Sent Events (SSE). This is based on the `cached_index.py` example but replaces terminal printing with HTTP streaming.

## Files Created

1. **`streaming_api.py`** - Main FastAPI application with SSE streaming
2. **`static/index.html`** - Beautiful web UI for testing the streaming
3. **`test_streaming_client.py`** - Python client to test the API from command line
4. **`STREAMING_API.md`** - Comprehensive documentation
5. **`run_streaming_api.sh`** - Helper script to run the server

## Quick Start

### 1. The server is already running! 🎉

```
✅ Server Status: Running on http://localhost:8001
```

### 2. Access the Application

Choose one of these options:

#### Option A: Beautiful Web Interface (Recommended)
Open in your browser: **http://localhost:8001**

- Enter your query in the text box
- Click "Start Execution"
- Watch events stream in real-time with beautiful formatting!

#### Option B: Swagger UI (For API Testing)
Open in your browser: **http://localhost:8001/docs**

1. Find `POST /api/stream` endpoint
2. Click "Try it out"
3. Enter request body:
   ```json
   {
     "query": "Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade.",
     "use_cache": true
   }
   ```
4. Click "Execute"
5. Watch the events stream!

#### Option C: Python Test Client
```bash
cd /app/examples
python test_streaming_client.py --port 8001
```

#### Option D: curl Command
```bash
curl -N -X POST "http://localhost:8001/api/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade.",
    "use_cache": true
  }'
```

## How It Works

### Architecture

```
User Request → FastAPI → Agent System → Event Bus → SSE Stream → Frontend
                            ↓
                    [Cache Backend]
```

1. **Client sends POST request** to `/api/stream` with a query
2. **FastAPI creates event bus** to capture execution events
3. **Agent system executes** (OmniAgent → DoctrineReceiver → StepExecutor → Reviewer)
4. **Events are captured** in real-time (LLM calls, tool executions, agent steps)
5. **Events are streamed** as Server-Sent Events (SSE) to the client
6. **Final result** is sent with cache statistics

### Event Types

The stream sends these event types:

- `start` - Execution started
- `agent_start` - Agent began processing
- `agent_complete` - Agent finished processing
- `llm_start` - LLM call started
- `llm_complete` - LLM call completed (may show [CACHED])
- `tool_start` - Tool execution started
- `tool_complete` - Tool execution completed
- `complete` - Final result with cache stats
- `error` - Error occurred

### Server-Sent Events (SSE) Format

Each event follows this format:
```
data: {"event_type": "llm_complete", "timestamp": "12:34:56", "detail": "Response received", "cached": false}

```

### Differences from `cached_index.py`

| Feature | cached_index.py | streaming_api.py |
|---------|----------------|------------------|
| Output | Terminal (EventPrinter) | HTTP Stream (SSE) |
| Interface | Command line | Web UI + API |
| Testing | Run script twice | Real-time in browser |
| Documentation | None | Swagger/OpenAPI |
| Client | Local only | Any HTTP client |

## Key Features

✅ **Real-time streaming** - See events as they happen
✅ **Cache visualization** - See which calls are cached
✅ **Beautiful UI** - Modern web interface
✅ **Swagger docs** - Interactive API documentation  
✅ **Statistics** - Cache hit rate, execution time
✅ **Error handling** - Graceful error reporting
✅ **Cross-platform** - Works with any HTTP client

## Testing the Caching

Run the same query twice to see caching in action:

1. First execution: All events will be fresh (no caching)
2. Second execution: LLM calls will show `[CACHED]` badge
3. Check the final statistics to see cache hit rate

## Example Event Flow

```
1. [START] Starting agent execution
2. [AGENT_START] OmniAgent started
3. [AGENT_START] DoctrineReceverAgent started
4. [LLM_START] Calling OpenAI...
5. [LLM_COMPLETE] Response received (250 tokens)
6. [TOOL_START] lista_pecas_por_tipo
7. [TOOL_COMPLETE] Found 3 documents
8. [AGENT_COMPLETE] DoctrineReceverAgent finished
9. [AGENT_START] StepExecutorAgent started
10. [LLM_START] Calling OpenAI... [CACHED] ⚡
11. [LLM_COMPLETE] Response received (180 tokens) [CACHED] ⚡
...
N. [COMPLETE] Execution completed
   📊 Cache Stats: 5 entries, 3 hits, 2 misses (60% hit rate)
```

## Troubleshooting

### Server not responding?
```bash
# Check if server is running
curl http://localhost:8001/health

# Restart server
cd /app/examples
python streaming_api.py --port 8001
```

### Can't see events in Swagger UI?
- Some browsers buffer SSE streams
- Use the web interface at http://localhost:8001 instead
- Or use the Python test client

### Import errors?
```bash
# Install dependencies
pip install fastapi 'uvicorn[standard]'

# Or use requirements.txt
pip install -r /app/requirements.txt
```

## Next Steps

1. **Try the web interface** - Most user-friendly option
2. **Customize the query** - Test with different questions
3. **Examine cache behavior** - Run same query multiple times
4. **Check the code** - See how events are captured and streamed
5. **Integrate with your frontend** - Use the SSE endpoint in your app

## API Integration Example

### JavaScript/TypeScript
```javascript
const response = await fetch('http://localhost:8001/api/stream', {
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
  // Process SSE events...
}
```

### Python
```python
import requests
import json

response = requests.post(
    'http://localhost:8001/api/stream',
    json={'query': 'Your query here', 'use_cache': True},
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        event = json.loads(line[6:])
        print(f"Event: {event['event_type']}")
```

## Production Considerations

Before deploying to production:

- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Configure CORS for your domain
- [ ] Use multiple workers: `uvicorn streaming_api:app --workers 4`
- [ ] Set up reverse proxy (nginx) with proper SSE configuration
- [ ] Add monitoring and logging
- [ ] Implement request timeouts
- [ ] Add error tracking (Sentry, etc.)

## Resources

- 📁 Main file: [streaming_api.py](streaming_api.py)
- 🌐 Web UI: [static/index.html](static/index.html)
- 📚 Full docs: [STREAMING_API.md](STREAMING_API.md)
- 🧪 Test client: [test_streaming_client.py](test_streaming_client.py)

## Support

For issues or questions:
1. Check the logs in terminal where server is running
2. Try the health check endpoint: http://localhost:8001/health
3. Review the full documentation in STREAMING_API.md
4. Test with the Python client for debugging

---

**Enjoy streaming! 🚀**
