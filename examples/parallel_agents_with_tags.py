#!/usr/bin/env python3
"""
Parallel Agent Execution with Custom Tags and Event Printer

This example demonstrates:
1. Running multiple agents in parallel
2. Using ExecutionEvent tags to customize the address
3. Visualizing parallel execution with ParallelEventPrinter
4. Using both realtime and progress display modes
"""

import asyncio
from typing import Optional
from pydantic import BaseModel, Field

from rh_agents import ExecutionState, ExecutionEvent, EventType, BaseActor
from rh_agents.bus_handlers import ParallelEventPrinter


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


class AnalysisRequest(BaseModel):
    """Input for sentiment analysis agent."""
    text: str = Field(..., description="Text to analyze")
    model: str = Field(default="standard", description="Analysis model to use")


class SentimentResult(BaseModel):
    """Result of sentiment analysis."""
    sentiment: str = Field(..., description="Detected sentiment (positive, negative, neutral)")
    confidence: float = Field(..., description="Confidence score 0-1")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")


class TranslationRequest(BaseModel):
    """Input for translation agent."""
    text: str = Field(..., description="Text to translate")
    target_lang: str = Field(..., description="Target language code")


class TranslationResult(BaseModel):
    """Result of translation."""
    translated_text: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Detected source language")


class SummaryRequest(BaseModel):
    """Input for summarization agent."""
    text: str = Field(..., description="Text to summarize")
    max_words: int = Field(default=50, description="Maximum words in summary")


class SummaryResult(BaseModel):
    """Result of summarization."""
    summary: str = Field(..., description="Generated summary")
    compression_ratio: float = Field(..., description="Ratio of summary to original")


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Handlers (Simulated AI Processing)
# ═══════════════════════════════════════════════════════════════════════════════


async def sentiment_analysis_handler(
    request: AnalysisRequest,
    context: str,
    execution_state: ExecutionState
) -> SentimentResult:
    """Simulate sentiment analysis processing."""
    # Simulate AI model inference
    await asyncio.sleep(0.8)
    
    # Simple sentiment detection logic (simulated)
    text_lower = request.text.lower()
    
    if any(word in text_lower for word in ["great", "excellent", "amazing", "love", "wonderful"]):
        sentiment = "positive"
        confidence = 0.92
    elif any(word in text_lower for word in ["bad", "terrible", "awful", "hate", "poor"]):
        sentiment = "negative"
        confidence = 0.88
    else:
        sentiment = "neutral"
        confidence = 0.75
    
    # Extract simple keywords (simulated)
    keywords = [word for word in text_lower.split() if len(word) > 4][:3]
    
    return SentimentResult(
        sentiment=sentiment,
        confidence=confidence,
        keywords=keywords
    )


async def translation_handler(
    request: TranslationRequest,
    context: str,
    execution_state: ExecutionState
) -> TranslationResult:
    """Simulate translation processing."""
    # Simulate translation API call
    await asyncio.sleep(1.2)
    
    # Simulated translation (just adding prefix for demo)
    translations = {
        "es": f"[ES] {request.text}",
        "fr": f"[FR] {request.text}",
        "de": f"[DE] {request.text}",
        "pt": f"[PT] {request.text}",
    }
    
    translated = translations.get(request.target_lang, f"[{request.target_lang.upper()}] {request.text}")
    
    return TranslationResult(
        translated_text=translated,
        source_lang="en"
    )


async def summarization_handler(
    request: SummaryRequest,
    context: str,
    execution_state: ExecutionState
) -> SummaryResult:
    """Simulate text summarization."""
    # Simulate summarization model
    await asyncio.sleep(0.6)
    
    words = request.text.split()
    original_length = len(words)
    
    # Simple summarization (take first N words)
    summary_words = words[:request.max_words]
    summary = " ".join(summary_words)
    
    if len(words) > request.max_words:
        summary += "..."
    
    compression = len(summary_words) / max(original_length, 1)
    
    return SummaryResult(
        summary=summary,
        compression_ratio=compression
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Factories
# ═══════════════════════════════════════════════════════════════════════════════


def create_sentiment_agent() -> BaseActor:
    """Create a sentiment analysis agent."""
    return BaseActor(
        name="SentimentAnalyzer",
        description="Analyzes text sentiment and extracts keywords",
        input_model=AnalysisRequest,
        output_model=SentimentResult,
        handler=sentiment_analysis_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=True
    )


def create_translation_agent() -> BaseActor:
    """Create a translation agent."""
    return BaseActor(
        name="TranslationAgent",
        description="Translates text to target language",
        input_model=TranslationRequest,
        output_model=TranslationResult,
        handler=translation_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=True
    )


def create_summarization_agent() -> BaseActor:
    """Create a summarization agent."""
    return BaseActor(
        name="SummarizationAgent",
        description="Generates concise summaries of text",
        input_model=SummaryRequest,
        output_model=SummaryResult,
        handler=summarization_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Example Scenarios
# ═══════════════════════════════════════════════════════════════════════════════


async def example_1_parallel_with_tags_realtime():
    """
    Example 1: Parallel execution with custom tags in REALTIME mode.
    
    Tags are used to create custom addresses in the execution tree,
    allowing you to organize and track different parallel tasks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Parallel Agents with Tags (REALTIME Mode)")
    print("=" * 70)
    print()
    print("Processing 5 customer reviews in parallel...")
    print("Each agent uses a custom tag to identify the review being processed.")
    print()
    
    # Create execution state with realtime printer
    printer = ParallelEventPrinter(parallel_mode="realtime")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    # Create sentiment analysis agent
    sentiment_agent = create_sentiment_agent()
    
    # Sample customer reviews
    reviews = [
        "This product is amazing! I absolutely love it.",
        "Terrible quality, very disappointed with my purchase.",
        "It's okay, nothing special but does the job.",
        "Excellent customer service and great product!",
        "Poor packaging, item arrived damaged."
    ]
    
    # Execute sentiment analysis in parallel with custom tags
    async with state.parallel(max_workers=3, name="Review Analysis") as p:
        for idx, review_text in enumerate(reviews):
            # Create ExecutionEvent with custom tag
            event = ExecutionEvent(
                actor=sentiment_agent,
                tag=f"review_{idx+1}"  # Custom tag to identify each review
            )
            
            # Add to parallel execution
            request = AnalysisRequest(text=review_text, model="advanced")
            p.add(event(request, "", state))
    
        # Gather all results
        results = await p.gather()
    
    # Display results summary
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    for idx, (review, result) in enumerate(zip(reviews, results)):
        if result.ok:
            sentiment_data = result.result
            print(f"\nReview {idx+1}: {sentiment_data.sentiment.upper()} ({sentiment_data.confidence:.0%})")
            print(f"  Text: {review[:50]}...")
            print(f"  Keywords: {', '.join(sentiment_data.keywords)}")
        else:
            print(f"\nReview {idx+1}: ERROR")
            print(f"  {result.erro_message}")


async def example_2_parallel_mixed_agents_progress():
    """
    Example 2: Different agent types in parallel with PROGRESS mode.
    
    Demonstrates running different types of agents concurrently,
    each with custom tags to track their specific tasks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Mixed Agent Types (PROGRESS Mode)")
    print("=" * 70)
    print()
    print("Processing documents with different agent types in parallel:")
    print("  • Sentiment Analysis")
    print("  • Translation")
    print("  • Summarization")
    print()
    
    # Create execution state with progress printer
    printer = ParallelEventPrinter(parallel_mode="progress")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    # Create different agents
    sentiment_agent = create_sentiment_agent()
    translation_agent = create_translation_agent()
    summary_agent = create_summarization_agent()
    
    # Sample text for processing
    sample_text = (
        "Artificial intelligence is transforming the way we work and live. "
        "Machine learning models can now understand and generate human language "
        "with remarkable accuracy. This technology opens up amazing possibilities "
        "for automation and innovation across industries."
    )
    
    # Execute different agent types in parallel
    async with state.parallel(max_workers=5, name="Document Processing") as p:
        # Sentiment analysis tasks
        for i in range(3):
            event = ExecutionEvent(
                actor=sentiment_agent,
                tag=f"sentiment_doc_{i+1}"
            )
            request = AnalysisRequest(text=sample_text)
            p.add(event(request, "", state))
        
        # Translation tasks
        for lang in ["es", "fr", "de"]:
            event = ExecutionEvent(
                actor=translation_agent,
                tag=f"translate_{lang}"
            )
            request = TranslationRequest(text=sample_text, target_lang=lang)
            p.add(event(request, "", state))
        
        # Summarization tasks
        for max_words in [20, 30, 40]:
            event = ExecutionEvent(
                actor=summary_agent,
                tag=f"summary_{max_words}w"
            )
            request = SummaryRequest(text=sample_text, max_words=max_words)
            p.add(event(request, "", state))
        
        # Stream results as they complete
        completed = 0
        async for result in p.stream():
            completed += 1
            # Results are displayed by the printer
    
    print("\n" + "-" * 70)
    print(f"✓ All {completed} tasks completed successfully!")
    print("-" * 70)


async def example_3_custom_address_hierarchy():
    """
    Example 3: Using tags to create hierarchical addresses.
    
    Tags can be used to create a logical hierarchy in the execution tree,
    making it easier to understand the relationship between tasks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hierarchical Task Organization with Tags")
    print("=" * 70)
    print()
    print("Processing customer feedback by category and priority...")
    print()
    
    # Create execution state with realtime printer
    printer = ParallelEventPrinter(parallel_mode="realtime")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    # Create agents
    sentiment_agent = create_sentiment_agent()
    
    # Feedback organized by category
    feedback_data = {
        "product": [
            "Excellent build quality!",
            "Design could be better",
        ],
        "service": [
            "Amazing customer support",
            "Response time was slow",
        ],
        "delivery": [
            "Package arrived on time",
            "Box was slightly damaged",
        ]
    }
    
    # Process feedback with hierarchical tags
    async with state.parallel(max_workers=4, name="Feedback Processing") as p:
        for category, feedback_list in feedback_data.items():
            for idx, feedback in enumerate(feedback_list):
                # Create hierarchical tag: category::priority
                priority = "high" if idx == 0 else "normal"
                tag = f"{category}::{priority}"
                
                event = ExecutionEvent(
                    actor=sentiment_agent,
                    tag=tag  # Hierarchical tag for organized addresses
                )
                
                request = AnalysisRequest(text=feedback)
                p.add(event(request, "", state))
        
        results = await p.gather()
    
    # Display organized results
    print("\n" + "-" * 70)
    print("RESULTS BY CATEGORY")
    print("-" * 70)
    idx = 0
    for category, feedback_list in feedback_data.items():
        print(f"\n{category.upper()}:")
        for feedback in feedback_list:
            if results[idx].ok:
                sentiment = results[idx].result
                print(f"  • {sentiment.sentiment} ({sentiment.confidence:.0%}): {feedback[:40]}...")
            idx += 1


async def example_4_streaming_with_tags():
    """
    Example 4: Streaming results with tagged events.
    
    Shows how to process results as they become available while
    using tags to identify which task completed.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Streaming Results with Custom Tags")
    print("=" * 70)
    print()
    print("Translating content to multiple languages...")
    print("Results stream in as each translation completes.")
    print()
    
    # Create execution state with progress printer
    printer = ParallelEventPrinter(parallel_mode="progress")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    # Create translation agent
    translation_agent = create_translation_agent()
    
    original_text = "Welcome to our amazing product! We hope you enjoy using it."
    target_languages = [
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("pt", "Portuguese"),
    ]
    
    start_time = asyncio.get_event_loop().time()
    
    # Execute translations in parallel
    async with state.parallel(max_workers=2, name="Multi-Language Translation") as p:
        for lang_code, lang_name in target_languages:
            event = ExecutionEvent(
                actor=translation_agent,
                tag=f"lang_{lang_code}"  # Tag by language
            )
            
            request = TranslationRequest(text=original_text, target_lang=lang_code)
            p.add(event(request, "", state))
        
        # Stream results
        print("\nStreaming translations as they complete:\n")
        async for result in p.stream():
            elapsed = asyncio.get_event_loop().time() - start_time
            if result.ok:
                translation = result.result
                print(f"[{elapsed:.2f}s] ✓ {translation.source_lang} → {translation.translated_text[:30]}...")
    
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"\n✓ All translations completed in {total_time:.2f}s")


async def main():
    """Run all examples."""
    print("=" * 70)
    print("PARALLEL AGENT EXECUTION WITH TAGS AND EVENT PRINTER")
    print("=" * 70)
    print()
    print("This example demonstrates:")
    print("  1. Running agents in parallel with controlled concurrency")
    print("  2. Using ExecutionEvent tags to customize addresses")
    print("  3. Visualizing execution with ParallelEventPrinter")
    print("  4. Both realtime and progress display modes")
    print()
    input("Press Enter to start...")
    
    # Run all examples
    await example_1_parallel_with_tags_realtime()
    
    print("\n\n")
    input("Press Enter for next example...")
    await example_2_parallel_mixed_agents_progress()
    
    print("\n\n")
    input("Press Enter for next example...")
    await example_3_custom_address_hierarchy()
    
    print("\n\n")
    input("Press Enter for final example...")
    await example_4_streaming_with_tags()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: Key Concepts")
    print("=" * 70)
    print()
    print("1. PARALLEL EXECUTION:")
    print("   async with state.parallel(max_workers=N) as p:")
    print("       p.add(event(input, context, state))")
    print("       results = await p.gather()  # or p.stream()")
    print()
    print("2. CUSTOM TAGS:")
    print("   event = ExecutionEvent(actor=agent, tag='custom_identifier')")
    print("   • Creates organized addresses: 'agent_name::tag'")
    print("   • Useful for tracking specific tasks")
    print("   • Enables hierarchical organization")
    print()
    print("3. EVENT PRINTER MODES:")
    print("   • realtime: Shows events as they happen (detailed)")
    print("   • progress: Shows aggregated progress bars (clean)")
    print()
    print("4. STREAMING RESULTS:")
    print("   async for result in p.stream():")
    print("       # Process results as they complete")
    print()


if __name__ == "__main__":
    asyncio.run(main())
