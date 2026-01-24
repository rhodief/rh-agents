#!/usr/bin/env python3
"""
Streaming Results Example

This example demonstrates how to process results as they become available
using the streaming mode, which is useful for:
- Starting downstream processing as soon as possible
- Reducing perceived latency
- Progressive UI updates
"""

import asyncio
from rh_agents import ExecutionState


async def fetch_api_data(api_id: int, delay: float) -> dict:
    """
    Simulate fetching data from an API endpoint.
    Different endpoints have different response times.
    """
    await asyncio.sleep(delay)
    
    return {
        "api_id": api_id,
        "data": f"Data from API {api_id}",
        "response_time": delay
    }


async def main():
    print("=" * 70)
    print("STREAMING RESULTS EXAMPLE")
    print("=" * 70)
    print()
    
    state = ExecutionState()
    
    # Different APIs with varying response times
    api_calls = [
        (1, 0.5),   # Fast API
        (2, 1.5),   # Slow API
        (3, 0.3),   # Very fast API
        (4, 1.0),   # Medium API
        (5, 0.7),   # Fast API
    ]
    
    print(f"Fetching data from {len(api_calls)} APIs...")
    print("Results will be displayed as they complete (not in order)")
    print()
    
    start_time = asyncio.get_event_loop().time()
    
    async with state.parallel(max_workers=3, name="API Calls") as p:
        # Add all API calls
        for api_id, delay in api_calls:
            p.add(lambda api_id=api_id, delay=delay: fetch_api_data(api_id, delay))
        
        # Process results as they complete (streaming mode)
        result_count = 0
        async for result in p.stream():
            result_count += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if result.ok:
                data = result.result
                print(f"[{elapsed:.2f}s] ✓ Result {result_count}/{len(api_calls)}")
                print(f"         API {data['api_id']}: {data['data']}")
                print(f"         Response time: {data['response_time']}s")
            else:
                print(f"[{elapsed:.2f}s] ✗ Result {result_count}/{len(api_calls)}")
                print(f"         Error: {result.erro_message}")
            print()
    
    total_time = asyncio.get_event_loop().time() - start_time
    
    print("=" * 70)
    print(f"All {len(api_calls)} API calls completed in {total_time:.2f}s")
    print("=" * 70)
    print()
    print("Benefits of streaming:")
    print("  • First result available immediately when ready")
    print("  • Can start processing while other tasks complete")
    print("  • Better for real-time/progressive UIs")
    print("  • Lower memory footprint for large result sets")


if __name__ == "__main__":
    asyncio.run(main())
