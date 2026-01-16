# Simple SSE Streaming with EventStreamer

The `EventStreamer` class wraps all the Server-Sent Events (SSE) complexity to work just like the `EventPrinter`.

## Comparison: Before vs After

### Before (Verbose - 60+ lines of boilerplate):

```python
# Create event queue
event_queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()

# Create queue handler
async def queue_handler(event: ExecutionEvent):
    await event_queue.put(event)

# Subscribe to bus
bus = EventBus()
bus.subscribe(queue_handler)

# Create complex event generator
async def event_generator() -> AsyncGenerator[str, None]:
    yield ": stream-start\n\n"
    
    execution_task = asyncio.create_task(
        run_execution(omni_agent, message, state)
    )
    
    try:
        while True:
            if execution_task.done() and event_queue.empty():
                break
            
            try:
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=0.25
                )
                yield f"data: {event.model_dump_json()}\n\n"
            
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
        
        await execution_task
        
        final_event = {
            "event_type": "complete",
            "message": "Execution completed successfully"
        }
        if cache_backend:
            final_event["cache_stats"] = cache_backend.get_stats()
        yield f"data: {json.dumps(final_event)}\n\n"
    
    except Exception as e:
        error_event = {
            "event_type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"

return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### After (Simple - 8 lines):

```python
# Create event bus with streamer (just like EventPrinter!)
bus = EventBus()
streamer = EventStreamer(include_cache_stats=True)
bus.subscribe(streamer)

# Start execution
execution_task = asyncio.create_task(
    ExecutionEvent[Message](actor=omni_agent)(message, "", state)
)

# Return streaming response - that's it!
return StreamingResponse(
    streamer.stream(execution_task=execution_task, cache_backend=cache_backend),
    media_type="text/event-stream"
)
```

## Usage Pattern

The `EventStreamer` follows the same pattern as `EventPrinter`:

### Terminal Printing:
```python
printer = EventPrinter(show_timestamp=True, show_address=True)
bus = EventBus()
bus.subscribe(printer)
```

### SSE Streaming:
```python
streamer = EventStreamer(include_cache_stats=True)
bus = EventBus()
bus.subscribe(streamer)
```

## Features

- **Simple API**: Just plug it into the event bus like any other subscriber
- **Automatic heartbeats**: Keeps connection alive with configurable interval
- **Error handling**: Automatically catches and sends error events
- **Cache stats**: Optional inclusion of cache statistics in completion event
- **SSE formatting**: Handles all Server-Sent Events protocol details internally

## Constructor Options

```python
EventStreamer(
    include_cache_stats=True,    # Include cache stats in completion event
    heartbeat_interval=0.25       # Heartbeat interval in seconds
)
```

## Full Example

See [streaming_api.py](./streaming_api.py) for a complete FastAPI implementation.
