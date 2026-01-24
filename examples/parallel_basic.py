#!/usr/bin/env python3
"""
Basic Parallel Execution Example

This example demonstrates the simplest use case for parallel execution:
running multiple independent tasks concurrently with controlled concurrency.
"""

import asyncio
from rh_agents import ExecutionState


async def process_document(doc_id: int) -> str:
    """
    Simulate processing a document.
    
    In a real application, this might involve:
    - Fetching document from database
    - Running ML model inference
    - Extracting entities or metadata
    - Storing results
    """
    # Simulate IO-bound work
    await asyncio.sleep(0.5)
    
    return f"Document {doc_id} processed successfully"


async def main():
    print("=" * 70)
    print("BASIC PARALLEL EXECUTION EXAMPLE")
    print("=" * 70)
    print()
    
    # Create execution state
    state = ExecutionState()
    
    # List of documents to process
    document_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"Processing {len(document_ids)} documents...")
    print(f"Max concurrent workers: 3")
    print()
    
    # Sequential processing (for comparison)
    print("Sequential processing:")
    start_time = asyncio.get_event_loop().time()
    
    sequential_results = []
    for doc_id in document_ids[:3]:  # Just 3 for comparison
        result = await process_document(doc_id)
        sequential_results.append(result)
    
    sequential_time = asyncio.get_event_loop().time() - start_time
    print(f"  Time: {sequential_time:.2f}s for 3 documents")
    print()
    
    # Parallel processing
    print("Parallel processing:")
    start_time = asyncio.get_event_loop().time()
    
    async with state.parallel(max_workers=3, name="Document Processing") as p:
        # Add all documents to parallel execution
        for doc_id in document_ids:
            p.add(lambda doc_id=doc_id: process_document(doc_id))
        
        # Wait for all to complete
        results = await p.gather()
    
    parallel_time = asyncio.get_event_loop().time() - start_time
    print(f"  Time: {parallel_time:.2f}s for {len(document_ids)} documents")
    print()
    
    # Results summary
    print("Results:")
    successful = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print()
    
    # Show first few results
    print("Sample results:")
    for i, result in enumerate(results[:3]):
        if result.ok:
            print(f"  {i+1}. {result.result}")
        else:
            print(f"  {i+1}. ERROR: {result.erro_message}")
    
    print()
    print("=" * 70)
    print(f"Speedup: {sequential_time * (len(document_ids)/3) / parallel_time:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
