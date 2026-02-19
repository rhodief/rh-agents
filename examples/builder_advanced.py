"""
Advanced examples demonstrating full builder capabilities.

Showcases:
- Dynamic prompt generation with context
- Error handling strategies with retry
- Result aggregation patterns
- Caching and artifacts
- Complex multi-agent workflows
- All configuration options
"""

import asyncio
from pydantic import BaseModel, Field
from typing import Optional

from rh_agents.builders import (
    StructuredAgent,
    CompletionAgent,
    ToolExecutorAgent,
    DirectToolAgent
)
from rh_agents.core.actors import Tool, LLM
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.types import ErrorStrategy, AggregationStrategy
from rh_agents.core.result_types import ToolExecutionResult
from rh_agents.models import Message, AuthorType
from rh_agents.llms import OpenAILLM
from rh_agents import EventPrinter


# ============================================================================
# Example 1: Dynamic Prompt Building with Context
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request for data analysis."""
    data_source: str
    analysis_type: str
    filters: Optional[dict] = None


class AnalysisResult(BaseModel):
    """Structured analysis output."""
    summary: str = Field(description="Brief summary of findings")
    key_insights: list[str] = Field(description="List of key insights")
    confidence: float = Field(description="Confidence score 0-1")
    recommendations: list[str] = Field(description="Actionable recommendations")


async def example_dynamic_prompts():
    """Demonstrate dynamic prompt generation based on context and state."""
    print("\n" + "="*80)
    print("Example 1: Dynamic Prompt Building with ExecutionState")
    print("="*80)
    
    llm = OpenAILLM()
    
    # Dynamic prompt builder with access to execution state
    async def build_analysis_prompt(input_data: AnalysisRequest, context: str, state: ExecutionState):
        """Build prompt dynamically based on input and prior results."""
        
        # Base prompt
        prompt = f"You are an expert data analyst specializing in {input_data.analysis_type} analysis."
        
        # Add data source specific instructions
        if input_data.data_source == "financial":
            prompt += "\n\nFocus on financial metrics, trends, and risk factors."
        elif input_data.data_source == "user_behavior":
            prompt += "\n\nFocus on user patterns, engagement metrics, and behavioral insights."
        
        # Include prior analysis if available
        # Note: get_steps_result expects list[int] indices, not string keys
        # In real usage, you'd track step indices during execution
        prior_results = state.get_steps_result([0, 1]) if len(state.history.get_event_list()) > 1 else None
        if prior_results:
            prompt += f"\n\nPrior Processing Results:\n{prior_results}"
        
        # Add filters context
        if input_data.filters:
            filter_str = ", ".join(f"{k}={v}" for k, v in input_data.filters.items())
            prompt += f"\n\nApply filters: {filter_str}"
        
        # Add external context
        if context:
            prompt += f"\n\nAdditional Context:\n{context}"
        
        return prompt
    
    # Create agent with dynamic prompt
    analyzer = await StructuredAgent.from_model(
        name="DynamicAnalyzer",
        llm=llm,
        input_model=AnalysisRequest,
        output_model=AnalysisResult,
        system_prompt="Placeholder - will be replaced by builder"
    )
    analyzer = (
        analyzer
        .with_system_prompt_builder(build_analysis_prompt)
        .with_temperature(0.3)  # Low for consistent analysis
        .with_max_tokens(2000)
        .as_cacheable(ttl=600)  # Cache for 10 minutes
    )
    
    print("\n✓ Agent created with dynamic prompt builder")
    print("  - Prompt adapts based on data_source")
    print("  - Incorporates prior execution results")
    print("  - Applies filters dynamically")
    print("  - Includes external context")


# ============================================================================
# Example 2: Advanced Error Handling with Retry
# ============================================================================

class APIRequest(BaseModel):
    """External API request."""
    endpoint: str
    params: dict


class APIResponse(BaseModel):
    """API response data."""
    status: str
    data: dict
    timestamp: str


# Mock tool that may fail
class UnreliableAPITool(Tool):
    """Simulates an unreliable external API."""
    def __init__(self):
        # Use closure to track call count
        call_count = [0]  # Mutable list for closure
        
        async def mock_api_call(input_data: APIRequest, context, state):
            call_count[0] += 1
            # Fail first 2 times, succeed on 3rd
            if call_count[0] < 3:
                raise Exception(f"API error: Connection timeout (attempt {call_count[0]})")
            
            return APIResponse(
                status="success",
                data={"result": "data from API"},
                timestamp="2026-02-19T12:00:00Z"
            )
        
        super().__init__(
            name="UnreliableAPI",
            description="Call external API that may fail",
            input_model=APIRequest,
            output_model=APIResponse,
            handler=mock_api_call
        )


async def example_error_handling_retry():
    """Demonstrate advanced error handling with retry configurations."""
    print("\n" + "="*80)
    print("Example 2: Error Handling & Retry Strategies")
    print("="*80)
    
    tool = UnreliableAPITool()
    
    # Strategy 1: Aggressive retry with backoff
    resilient_agent = await DirectToolAgent.from_tool(
        name="ResilientAPI",
        tool=tool
    )
    resilient_agent = (
        resilient_agent
        .with_error_strategy(ErrorStrategy.LOG_AND_CONTINUE)
        .with_retry(
            max_attempts=5,
            initial_delay=0.5,
            backoff_factor=2.0,  # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
            max_delay=10.0
        )
    )
    
    print("\n✓ Created resilient agent with exponential backoff")
    print("  - max_attempts: 5")
    print("  - Delays: 0.5s → 1s → 2s → 4s → 8s")
    print("  - Error strategy: LOG_AND_CONTINUE")
    
    # Strategy 2: Fail-fast for critical operations
    critical_agent = await DirectToolAgent.from_tool(
        name="CriticalAPI",
        tool=tool
    )
    critical_agent = (
        critical_agent
        .with_error_strategy(ErrorStrategy.RAISE)
        .with_retry(max_attempts=1, initial_delay=0.1)  # No retries
    )
    
    print("\n✓ Created fail-fast agent for critical operations")
    print("  - max_attempts: 1 (no retries)")
    print("  - Error strategy: RAISE (fail immediately)")
    
    # Strategy 3: Silent background operation
    background_agent = await DirectToolAgent.from_tool(
        name="BackgroundSync",
        tool=tool
    )
    background_agent = (
        background_agent
        .with_error_strategy(ErrorStrategy.SILENT)
        .with_retry(max_attempts=3, initial_delay=2.0)
    )
    
    print("\n✓ Created silent background agent")
    print("  - max_attempts: 3")
    print("  - Error strategy: SILENT (suppress errors)")


# ============================================================================
# Example 3: Result Aggregation Patterns
# ============================================================================

class SearchQuery(BaseModel):
    """Multi-source search query."""
    query: str
    max_results: int = 10


class SearchResult(BaseModel):
    """Individual search result."""
    source: str
    title: str
    snippet: str
    relevance: float


# Mock search tools
class WebSearchTool(Tool):
    def __init__(self, source_name: str):
        _source = source_name  # Store in closure
        
        async def search(query: SearchQuery, context, state):
            # Mock search results
            return SearchResult(
                source=_source,
                title=f"Result from {_source}",
                snippet=f"Found information about '{query.query}' from {_source}",
                relevance=0.85
            )
        
        super().__init__(
            name=f"Search{source_name}",
            description=f"Search {source_name} for information",
            input_model=SearchQuery,
            output_model=SearchResult,
            handler=search
        )


async def example_aggregation_patterns():
    """Demonstrate different result aggregation strategies."""
    print("\n" + "="*80)
    print("Example 3: Result Aggregation Patterns")
    print("="*80)
    
    llm = OpenAILLM()
    
    # Create multiple search tools
    tools: list[Tool] = [
        WebSearchTool("Google"),
        WebSearchTool("Bing"),
        WebSearchTool("DuckDuckGo"),
        WebSearchTool("ArXiv")
    ]
    
    # Pattern 1: DICT for detailed access
    print("\n--- Pattern 1: DICT (Default) ---")
    dict_executor = await ToolExecutorAgent.from_tools(
        name="MultiSearchDict",
        llm=llm,
        input_model=SearchQuery,
        output_model=SearchResult,
        system_prompt="Search all sources and return detailed results",
        tools=tools
    )
    
    print("✓ DICT aggregation")
    print("  Usage: result['SearchGoogle'], result.execution_order")
    print("  Best for: Individual tool result access, debugging")
    
    # Pattern 2: LIST for iteration
    print("\n--- Pattern 2: LIST ---")
    list_executor = await ToolExecutorAgent.from_tools(
        name="MultiSearchList",
        llm=llm,
        input_model=SearchQuery,
        output_model=SearchResult,
        system_prompt="Search all sources",
        tools=tools
    )
    list_executor = list_executor.with_aggregation(AggregationStrategy.LIST)
    
    print("✓ LIST aggregation")
    print("  Usage: for result in results: process(result)")
    print("  Best for: Sequential processing, filtering, mapping")
    
    # Pattern 3: CONCATENATE for reports
    print("\n--- Pattern 3: CONCATENATE ---")
    report_executor = await ToolExecutorAgent.from_tools(
        name="SearchReport",
        llm=llm,
        input_model=SearchQuery,
        output_model=SearchResult,
        system_prompt="Generate comprehensive search report",
        tools=tools
    )
    report_executor = (
        report_executor
        .with_aggregation(
            AggregationStrategy.CONCATENATE,
            separator="\n\n" + "="*60 + "\n\n"
        )
        .with_temperature(0.2)
    )
    
    print("✓ CONCATENATE aggregation with custom separator")
    print("  Usage: report_text = result")
    print("  Best for: Report generation, summaries, concatenated output")
    
    # Pattern 4: FIRST for fastest response
    print("\n--- Pattern 4: FIRST ---")
    fast_executor = await ToolExecutorAgent.from_tools(
        name="FastSearch",
        llm=llm,
        input_model=SearchQuery,
        output_model=SearchResult,
        system_prompt="Get fastest available result",
        tools=tools
    )
    fast_executor = fast_executor.with_aggregation(AggregationStrategy.FIRST)
    
    print("✓ FIRST aggregation")
    print("  Usage: fastest_result = result")
    print("  Best for: Quick responses, fallback chains, racing sources")


# ============================================================================
# Example 4: Caching & Artifacts
# ============================================================================

class DocumentRequest(BaseModel):
    """Request to process a document."""
    doc_id: str
    processing_type: str


class ProcessedDocument(BaseModel):
    """Processed document with metadata."""
    doc_id: str
    summary: str
    word_count: int
    key_topics: list[str]


async def example_caching_artifacts():
    """Demonstrate caching and artifact storage patterns."""
    print("\n" + "="*80)
    print("Example 4: Caching & Artifact Storage")
    print("="*80)
    
    llm = OpenAILLM()
    
    # Pattern 1: Short-lived cache for user sessions
    print("\n--- Pattern 1: Session Cache (5 minutes) ---")
    session_processor = await StructuredAgent.from_model(
        name="SessionProcessor",
        llm=llm,
        input_model=DocumentRequest,
        output_model=ProcessedDocument,
        system_prompt="Process document efficiently"
    )
    session_processor = (
        session_processor
        .with_temperature(0.5)
        .as_cacheable(ttl=300)  # 5 minutes
    )
    
    print("✓ Created with 5-minute cache")
    print("  - Same input within 5min returns cached result")
    print("  - No API calls for cached hits")
    print("  - Automatic expiration")
    
    # Pattern 2: Long-lived cache for static content
    print("\n--- Pattern 2: Static Content Cache (24 hours) ---")
    static_processor = await StructuredAgent.from_model(
        name="StaticProcessor",
        llm=llm,
        input_model=DocumentRequest,
        output_model=ProcessedDocument,
        system_prompt="Process static reference documents"
    )
    static_processor = (
        static_processor
        .with_temperature(0.1)  # Very deterministic
        .as_cacheable(ttl=86400)  # 24 hours
    )
    
    print("✓ Created with 24-hour cache")
    print("  - Perfect for reference docs that don't change")
    print("  - Significant cost savings")
    
    # Pattern 3: Indefinite cache
    print("\n--- Pattern 3: Indefinite Cache ---")
    reference_processor = await StructuredAgent.from_model(
        name="ReferenceProcessor",
        llm=llm,
        input_model=DocumentRequest,
        output_model=ProcessedDocument,
        system_prompt="Process permanent reference documents"
    )
    reference_processor = reference_processor.as_cacheable(ttl=None)  # Never expires
    
    print("✓ Created with indefinite cache")
    print("  - Cache persists until manual clear")
    print("  - For truly static content")
    
    # Pattern 4: Artifacts for large outputs
    print("\n--- Pattern 4: Artifact Storage ---")
    report_generator = await CompletionAgent.from_prompt(
        name="ReportGenerator",
        llm=llm,
        input_model=DocumentRequest,
        output_model=Message,
        system_prompt="Generate comprehensive report"
    )
    report_generator = (
        report_generator
        .with_max_tokens(8000)  # Large output
        .as_artifact()  # Store separately
        .as_cacheable(ttl=3600)
    )
    
    print("✓ Created with artifact storage")
    print("  - Large outputs stored separately from traces")
    print("  - Cleaner execution logs")
    print("  - Retrieve via artifact ID")


# ============================================================================
# Example 5: Complex Multi-Agent Workflow
# ============================================================================

class WorkflowInput(BaseModel):
    """Input for complex workflow."""
    task: str
    priority: str = "normal"


class ValidationInput(BaseModel):
    """Validation check input."""
    data: dict


class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: list[str]


class EnrichmentResult(BaseModel):
    """Enriched data result."""
    original_data: dict
    enrichments: dict
    confidence: float


# Mock tools
class ValidatorTool(Tool):
    def __init__(self):
        async def validate(input_data: ValidationInput, context, state):
            # Mock validation
            return ValidationResult(
                is_valid=True,
                errors=[]
            )
        
        super().__init__(
            name="Validator",
            description="Validate data structure",
            input_model=ValidationInput,
            output_model=ValidationResult,
            handler=validate
        )


class EnrichmentTool(Tool):
    def __init__(self):
        async def enrich(input_data: ValidationInput, context, state):
            # Mock enrichment
            return EnrichmentResult(
                original_data=input_data.data,
                enrichments={"metadata": "added", "score": 0.95},
                confidence=0.95
            )
        
        super().__init__(
            name="Enricher",
            description="Enrich data with additional information",
            input_model=ValidationInput,
            output_model=EnrichmentResult,
            handler=enrich
        )


async def example_complex_workflow():
    """Demonstrate complex multi-stage workflow with all features."""
    print("\n" + "="*80)
    print("Example 5: Complex Multi-Agent Workflow")
    print("="*80)
    
    llm = OpenAILLM()
    
    # Stage 1: Parse and structure input
    print("\n--- Stage 1: Input Parser ---")
    parser = await StructuredAgent.from_model(
        name="WorkflowParser",
        llm=llm,
        input_model=Message,
        output_model=WorkflowInput,
        system_prompt="Parse user input into structured workflow"
    )
    parser = (
        parser
        .with_temperature(0.3)
        .with_max_tokens(500)
        .as_cacheable(ttl=60)
    )
    print("✓ Parser: Message → WorkflowInput (cached)")
    
    # Stage 2: Validate and enrich data
    print("\n--- Stage 2: Validation & Enrichment ---")
    processor = await ToolExecutorAgent.from_tools(
        name="DataProcessor",
        llm=llm,
        input_model=WorkflowInput,
        output_model=ValidationResult,
        system_prompt="Validate and enrich workflow data",
        tools=[ValidatorTool(), EnrichmentTool()]
    )
    processor = (
        processor
        .with_aggregation(AggregationStrategy.LIST)
        .with_error_strategy(ErrorStrategy.RETURN_NONE)
        .with_retry(max_attempts=3, initial_delay=1.0)
    )
    print("✓ Processor: WorkflowInput → [ValidationResult, EnrichmentResult]")
    print("  - Runs tools in parallel")
    print("  - Returns list for easy iteration")
    print("  - Retries on failure")
    
    # Stage 3: Generate final report
    print("\n--- Stage 3: Report Generator ---")
    
    async def build_report_prompt(input_data, context, state: ExecutionState):
        # Access prior stage results
        # Note: In real usage, track step indices during execution
        parsed = state.get_steps_result([0]) if len(state.history.get_event_list()) > 0 else None
        validated = state.get_steps_result([1]) if len(state.history.get_event_list()) > 1 else None
        
        return f"""
Generate a comprehensive workflow execution report.

Original Input: {context}
Parsed Structure: {parsed}
Processing Results: {validated}

Create a detailed summary with:
1. Input analysis
2. Validation status
3. Enrichment results  
4. Recommendations
"""
    
    report_gen = await CompletionAgent.from_prompt(
        name="ReportGenerator",
        llm=llm,
        input_model=WorkflowInput,
        output_model=Message,
        system_prompt="Placeholder"
    )
    report_gen = (
        report_gen
        .with_system_prompt_builder(build_report_prompt)
        .with_temperature(0.7)
        .with_max_tokens(3000)
        .as_artifact()
        .as_cacheable(ttl=1800)
    )
    print("✓ Reporter: WorkflowInput → Message (artifact, cached)")
    print("  - Dynamic prompt with prior results")
    print("  - Large output as artifact")
    print("  - 30-minute cache")
    
    # Execution flow summary
    print("\n--- Complete Workflow ---")
    print("1. Parser:    User message → Structured input (StructuredAgent)")
    print("2. Processor: Validate + enrich in parallel (ToolExecutorAgent)")
    print("3. Reporter:  Generate final report with context (CompletionAgent)")
    print("\nFeatures:")
    print("  ✓ 3-stage pipeline with type safety")
    print("  ✓ Caching at each stage")
    print("  ✓ Error handling with retry")
    print("  ✓ Result aggregation (LIST)")
    print("  ✓ Dynamic prompts with state access")
    print("  ✓ Artifact storage for large output")


# ============================================================================
# Example 6: Production Configuration Pattern
# ============================================================================

async def example_production_config():
    """Demonstrate production-ready configuration pattern."""
    print("\n" + "="*80)
    print("Example 6: Production Configuration Pattern")
    print("="*80)
    
    llm = OpenAILLM()
    
    # Production-ready agent with all best practices
    production_agent = await StructuredAgent.from_model(
        name="ProductionAgent",
        llm=llm,
        input_model=WorkflowInput,
        output_model=ProcessedDocument,
        system_prompt="Production-grade document processing"
    )
    production_agent = (
        production_agent
        # LLM Configuration
        .with_model('gpt-4o')           # Best quality model
        .with_temperature(0.3)          # Low for consistency
        .with_max_tokens(2000)          # Reasonable limit
        
        # Error Handling
        .with_error_strategy(ErrorStrategy.RETURN_NONE)  # Graceful failures
        .with_retry(
            max_attempts=3,
            initial_delay=1.0,
            backoff_factor=1.5
        )
        
        # Optimization
        .as_cacheable(ttl=300)          # 5-minute cache
        .as_artifact()                  # Separate storage
    )
    
    print("\n✓ Production-ready agent configured with:")
    print("\nLLM Settings:")
    print("  - Model: gpt-4o (best quality)")
    print("  - Temperature: 0.3 (consistent)")
    print("  - Max tokens: 2000 (controlled cost)")
    
    print("\nReliability:")
    print("  - Error strategy: RETURN_NONE (graceful)")
    print("  - Retry: 3 attempts with 1.5x backoff")
    print("  - Automatic failure recovery")
    
    print("\nPerformance:")
    print("  - Cache: 5-minute TTL")
    print("  - Artifact storage for large outputs")
    print("  - Reduced API calls")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all advanced examples."""
    print("\n" + "="*80)
    print("BUILDER PATTERN - ADVANCED EXAMPLES")
    print("="*80)
    
    print("\nThese examples demonstrate:")
    print("  1. Dynamic prompt building with ExecutionState")
    print("  2. Advanced error handling with retry strategies")
    print("  3. Result aggregation patterns (DICT, LIST, CONCATENATE, FIRST)")
    print("  4. Caching and artifact storage")
    print("  5. Complex multi-agent workflows")
    print("  6. Production-ready configuration")
    
    await example_dynamic_prompts()
    await example_error_handling_retry()
    await example_aggregation_patterns()
    await example_caching_artifacts()
    await example_complex_workflow()
    await example_production_config()
    
    print("\n" + "="*80)
    print("KEY ADVANCED PATTERNS")
    print("="*80)
    print("\n1. Dynamic Prompts")
    print("   - with_system_prompt_builder() for context-aware prompts")
    print("   - Access ExecutionState for prior results")
    print("   - Adapt based on input parameters")
    
    print("\n2. Error Resilience")
    print("   - Combine error strategies with retry logic")
    print("   - Exponential backoff for rate limiting")
    print("   - Different strategies for different criticality")
    
    print("\n3. Result Aggregation")
    print("   - DICT: Full control, dict-like access")
    print("   - LIST: Sequential processing")
    print("   - CONCATENATE: Report generation")
    print("   - FIRST: Fast responses, fallback chains")
    
    print("\n4. Performance Optimization")
    print("   - Multi-level caching with appropriate TTLs")
    print("   - Artifact storage for large outputs")
    print("   - Cost reduction through caching")
    
    print("\n5. Production Patterns")
    print("   - Comprehensive error handling")
    print("   - Optimal LLM configuration")
    print("   - Monitoring and observability ready")
    
    print("\n✓ See BUILDERS_GUIDE.md and CONFIGURATION_GUIDE.md for complete documentation")
    print()


if __name__ == "__main__":
    asyncio.run(main())
