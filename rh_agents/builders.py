"""
Builder factories for common agent patterns.

This module provides simplified APIs for creating agents that follow common patterns:
- StructuredAgent: Forces LLM to return structured data via tool choice
- CompletionAgent: Simple text completions without tools
- ToolExecutorAgent: LLM decides which tools to call, executes them in parallel
- DirectToolAgent: Direct tool execution without LLM

All builders return BuilderAgent instances with chainable configuration methods.
"""

import asyncio
from typing import Any, Callable, Coroutine, Union
from typing_extensions import Self
from pydantic import BaseModel

from rh_agents.core.actors import Agent, Tool, ToolSet, LLM
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.core.result_types import LLM_Result, ToolExecutionResult
from rh_agents.core.types import EventType, ErrorStrategy, AggregationStrategy
from rh_agents.openai import OpenAIRequest


class BuilderAgent(Agent):
    """
    Agent subclass with chainable builder methods.
    
    Extends Agent with fluent API for configuring common options like temperature,
    max_tokens, error handling, etc. All chainable methods return self for chaining.
    
    Example:
        >>> agent = (
        ...     await StructuredAgent.from_model(...)
        ...     .with_temperature(0.7)
        ...     .with_max_tokens(2000)
        ...     .as_cacheable()
        ... )
    """
    
    # Pydantic configuration
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize overrides dictionary for storing configuration changes
        if not hasattr(self, '_overrides'):
            object.__setattr__(self, '_overrides', {})
    
    def with_temperature(self, temperature: float) -> Self:
        """
        Override LLM temperature (0.0-2.0).
        
        Args:
            temperature: Controls randomness. 0=deterministic, 2=very random.
            
        Raises:
            ValueError: If temperature not in range [0.0, 2.0]
        """
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        self._overrides['temperature'] = temperature  # type: ignore
        return self
    
    def with_max_tokens(self, max_tokens: int) -> Self:
        """
        Override maximum tokens in response.
        
        Args:
            max_tokens: Maximum number of tokens to generate (1-128000).
            
        Raises:
            ValueError: If max_tokens not in valid range
        """
        if not (1 <= max_tokens <= 128000):
            raise ValueError(f"max_tokens must be between 1 and 128000, got {max_tokens}")
        self._overrides['max_tokens'] = max_tokens  # type: ignore
        return self
    
    def with_model(self, model: str) -> Self:
        """Override LLM model (e.g., 'gpt-4o-mini')."""
        self._overrides['model'] = model  # type: ignore
        return self
    
    def with_tools(self, tools: list[Tool]) -> Self:
        """Set or replace tools available to the agent."""
        self.tools = ToolSet(tools=tools)
        return self
    
    def with_tool_choice(self, tool_name: str) -> Self:
        """Force LLM to call a specific tool (for structured output)."""
        self._overrides['tool_choice'] = {"type": "function", "function": {"name": tool_name}}  # type: ignore
        return self
    
    def with_context_transform(self, fn: Callable[[str], str]) -> Self:
        """Custom function to format context before appending to system prompt."""
        self._overrides['context_transform'] = fn  # type: ignore
        return self
    
    def with_system_prompt_builder(self, fn: Callable[[Any, str, ExecutionState], Coroutine[Any, Any, str]]) -> Self:
        """Dynamic prompt builder function with access to input, context, and state."""
        self._overrides['system_prompt_builder'] = fn  # type: ignore
        return self
    
    def with_error_strategy(self, strategy: ErrorStrategy) -> Self:
        """Set error handling strategy."""
        self._overrides['error_strategy'] = strategy  # type: ignore
        return self
    
    def with_retry(self, max_attempts: int, initial_delay: float = 1.0, **kwargs) -> Self:
        """
        Configure retry behavior for failures.
        
        Args:
            max_attempts: Maximum number of retry attempts (1-10)
            initial_delay: Initial delay in seconds between retries (> 0)
            **kwargs: Additional retry configuration
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        if not (1 <= max_attempts <= 10):
            raise ValueError(f"max_attempts must be between 1 and 10, got {max_attempts}")
        if initial_delay <= 0:
            raise ValueError(f"initial_delay must be positive, got {initial_delay}")
        
        from rh_agents.core.retry import RetryConfig
        self.retry_config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            **kwargs
        )
        return self
    
    def with_first_result_only(self) -> Self:
        """Stop execution after first successful tool call (for ToolExecutorAgent)."""
        self._overrides['first_result_only'] = True  # type: ignore
        return self
    
    def with_aggregation(self, strategy: 'AggregationStrategy', separator: str = "\n\n") -> Self:
        """
        Set result aggregation strategy (for ToolExecutorAgent).
        
        Args:
            strategy: How to aggregate multiple tool results
            separator: Separator for CONCATENATE strategy (default: double newline)
            
        Returns:
            Self for chaining
        """
        from rh_agents.core.types import AggregationStrategy
        self._overrides['aggregation_strategy'] = strategy  # type: ignore
        self._overrides['aggregation_separator'] = separator  # type: ignore
        return self
    
    def as_artifact(self) -> Self:
        """Mark agent output as artifact for separate storage."""
        self.is_artifact = True
        return self
    
    def as_cacheable(self, ttl: Union[int, None] = None) -> Self:
        """
        Enable caching of agent results.
        
        Args:
            ttl: Time-to-live in seconds (None = no expiration, > 0 for expiration)
            
        Raises:
            ValueError: If ttl is negative
        """
        if ttl is not None and ttl < 0:
            raise ValueError(f"TTL must be non-negative, got {ttl}")
        
        self.cacheable = True
        if ttl is not None:
            self.cache_ttl = ttl
        return self


class StructuredAgent:
    """
    Builder for agents that force LLM to return structured data.
    
    Creates an agent that uses tool choice to guarantee structured output matching
    a Pydantic model. The LLM is forced to call a specific tool that validates
    and returns an instance of the output_model.
    
    Usage Pattern:
        Use when you need the LLM to return typed, validated data structures
        rather than free-form text. Perfect for parsing user input into workflows,
        extracting information, or any scenario requiring schema compliance.
    
    Examples:
        Basic usage with mandatory parameters:
        >>> agent = await StructuredAgent.from_model(
        ...     name="DoctrineParser",
        ...     llm=llm,
        ...     input_model=Message,
        ...     output_model=Doctrine,
        ...     system_prompt="Parse user request into structured plan."
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await StructuredAgent.from_model(
        ...         name="DoctrineParser",
        ...         llm=llm,
        ...         input_model=Message,
        ...         output_model=Doctrine,
        ...         system_prompt="Parse user request into structured plan."
        ...     )
        ...     .with_tools([DoctrineTool()])
        ...     .with_tool_choice("DoctrineTool")
        ...     .with_temperature(0.7)
        ...     .as_artifact()
        ...     .as_cacheable()
        ... )
    """
    
    @staticmethod
    async def from_model(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str
    ) -> BuilderAgent:
        """
        Create a StructuredAgent that forces structured output.
        
        Args:
            name: Unique identifier for the agent
            llm: LLM instance to use for generation
            input_model: Pydantic model for input validation
            output_model: Pydantic model for output validation (enforced via tool choice)
            system_prompt: Base system prompt for the agent
            
        Returns:
            BuilderAgent instance ready for optional configuration
        """
        # Closure variables for overrides
        overrides = {}
        
        async def handler(input_data, context: str, execution_state: ExecutionState):
            """Generated handler that uses ExecutionEvent and enforces structured output."""
            # Get context transform function
            context_transform = overrides.get(
                'context_transform',
                lambda ctx: f"\n\nContext: {ctx}" if ctx else ""
            )
            
            # Build system prompt (check for dynamic builder)
            if 'system_prompt_builder' in overrides:
                prompt = await overrides['system_prompt_builder'](input_data, context, execution_state)
            else:
                prompt = system_prompt + context_transform(context)
            
            # Get input content
            input_content = getattr(input_data, 'content', str(input_data))
            
            # Create ExecutionEvent for LLM
            llm_event = ExecutionEvent(actor=llm)
            
            # Get tool set
            tools = overrides.get('tools', ToolSet(tools=[]))
            
            # Prepare LLM request with overrides
            llm_input = OpenAIRequest(
                system_message=prompt,
                prompt=input_content,
                model=overrides.get('model', 'gpt-4o'),
                max_completion_tokens=overrides.get('max_tokens', 2500),
                temperature=overrides.get('temperature', 1.0),
                tools=tools,
                tool_choice=overrides.get('tool_choice')
            )
            
            # Execute with ExecutionEvent
            execution_result = await llm_event(llm_input, context, execution_state)
            
            # Handle errors based on strategy
            error_strategy = overrides.get('error_strategy', ErrorStrategy.RAISE)
            if not execution_result.ok or execution_result.result is None:
                error_msg = f"LLM execution failed: {execution_result.erro_message}"
                if error_strategy == ErrorStrategy.RAISE:
                    raise Exception(error_msg)
                elif error_strategy == ErrorStrategy.RETURN_NONE:
                    return ExecutionResult[output_model](result=None, ok=False, erro_message=error_msg)
                elif error_strategy == ErrorStrategy.LOG_AND_CONTINUE:
                    await execution_state.log(f"Warning: {error_msg}")
                    return output_model()  # Return empty instance
                else:  # SILENT
                    return None
            
            result = execution_result.result
            
            # Check if it's a content response (fallback)
            if result.is_content:
                from rh_agents.models import Message, AuthorType
                return Message(content=result.content, author=AuthorType.ASSISTANT)
            
            # Ensure we got a tool call
            if not (result.is_tool_call and result.tools and result.tools[0]):
                error_msg = "LLM did not return a valid tool call for structured output."
                if error_strategy == ErrorStrategy.RAISE:
                    raise Exception(error_msg)
                else:
                    return None
            
            # Extract and validate tool call
            tool_call = result.tools[0]
            return output_model.model_validate_json(tool_call.arguments)
        
        # Create BuilderAgent
        agent = BuilderAgent(
            name=name,
            description=system_prompt,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools=[])
        )
        
        # Store overrides reference so chainable methods can modify it
        object.__setattr__(agent, '_overrides', overrides)  # type: ignore
        
        return agent


class CompletionAgent:
    """
    Builder for agents that perform simple text completions.
    
    Creates an agent that calls the LLM without tool usage, returning text content
    directly. Ideal for summarization, analysis, generation, and any task where
    free-form text output is desired.
    
    Usage Pattern:
        Use when you need natural language output rather than structured data.
        Perfect for summarization, creative writing, analysis, Q&A, and text
        transformation tasks.
    
    Examples:
        Basic usage:
        >>> agent = await CompletionAgent.from_prompt(
        ...     name="Summarizer",
        ...     llm=llm,
        ...     input_model=Message,
        ...     output_model=Message,
        ...     system_prompt="Summarize the provided text concisely."
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await CompletionAgent.from_prompt(
        ...         name="LegalReviewer",
        ...         llm=llm,
        ...         input_model=Doctrine,
        ...         output_model=Message,
        ...         system_prompt="Generate comprehensive legal analysis."
        ...     )
        ...     .with_temperature(0.8)
        ...     .with_max_tokens(3000)
        ...     .with_system_prompt_builder(dynamic_context_fn)
        ...     .as_cacheable(ttl=1800)
        ... )
    """
    
    @staticmethod
    async def from_prompt(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str
    ) -> BuilderAgent:
        """
        Create a CompletionAgent for text generation.
        
        Args:
            name: Unique identifier for the agent
            llm: LLM instance to use for generation
            input_model: Pydantic model for input validation
            output_model: Pydantic model for output (typically Message)
            system_prompt: Base system prompt for the agent
            
        Returns:
            BuilderAgent instance ready for optional configuration
        """
        # Closure variables for overrides
        overrides = {}
        
        async def handler(input_data, context: str, execution_state: ExecutionState):
            """Generated handler for simple completions."""
            # Get context transform function
            context_transform = overrides.get(
                'context_transform',
                lambda ctx: f"\n\nContext: {ctx}" if ctx else ""
            )
            
            # Build system prompt
            if 'system_prompt_builder' in overrides:
                prompt = await overrides['system_prompt_builder'](input_data, context, execution_state)
            else:
                prompt = system_prompt + context_transform(context)
            
            # Get input content
            input_content = getattr(input_data, 'content', str(input_data))
            
            # Create ExecutionEvent for LLM
            llm_event = ExecutionEvent(actor=llm)
            
            # Prepare LLM request (no tools for completion)
            llm_input = OpenAIRequest(
                system_message=prompt,
                prompt=input_content,
                model=overrides.get('model', 'gpt-4o'),
                max_completion_tokens=overrides.get('max_tokens', 2500),
                temperature=overrides.get('temperature', 1.0)
            )
            
            # Execute with ExecutionEvent
            execution_result = await llm_event(llm_input, context, execution_state)
            
            # Handle errors
            error_strategy = overrides.get('error_strategy', ErrorStrategy.RAISE)
            if not execution_result.ok or execution_result.result is None:
                error_msg = f"LLM execution failed: {execution_result.erro_message}"
                if error_strategy == ErrorStrategy.RAISE:
                    raise Exception(error_msg)
                elif error_strategy == ErrorStrategy.RETURN_NONE:
                    return ExecutionResult[output_model](result=None, ok=False, erro_message=error_msg)
                elif error_strategy == ErrorStrategy.LOG_AND_CONTINUE:
                    await execution_state.log(f"Warning: {error_msg}")
                    return output_model()
                else:  # SILENT
                    return None
            
            response = execution_result.result
            
            # Return message with content
            from rh_agents.models import Message, AuthorType
            return Message(content=response.content or "", author=AuthorType.ASSISTANT)
        
        # Create BuilderAgent
        agent = BuilderAgent(
            name=name,
            description=system_prompt,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools=[])
        )
        
        # Store overrides reference
        object.__setattr__(agent, '_overrides', overrides)  # type: ignore
        
        return agent


class ToolExecutorAgent:
    """
    Builder for agents that execute tools based on LLM decisions.
    
    Creates an agent that gives the LLM access to tools, lets it decide which to call,
    executes them in parallel, and aggregates results. Handles tool call parsing,
    execution, error handling, and result combination automatically.
    
    Usage Pattern:
        Use when the LLM needs to interact with external systems, fetch data,
        perform computations, or take actions. The LLM decides which tools to call
        and with what parameters based on the input and context.
    
    Examples:
        Basic usage:
        >>> agent = await ToolExecutorAgent.from_tools(
        ...     name="DataFetcher",
        ...     llm=llm,
        ...     input_model=Query,
        ...     output_model=Result,
        ...     system_prompt="Fetch and analyze data using available tools.",
        ...     tools=[DatabaseTool(), APITool(), CalculatorTool()]
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await ToolExecutorAgent.from_tools(
        ...         name="StepExecutor",
        ...         llm=llm,
        ...         input_model=DoctrineStep,
        ...         output_model=StepResult,
        ...         system_prompt="Execute step using available tools.",
        ...         tools=[ListTool(), GetTool(), ProcessTool()]
        ...     )
        ...     .with_temperature(0.9)
        ...     .with_system_prompt_builder(build_step_context)
        ...     .with_first_result_only()
        ...     .with_retry(max_attempts=3)
        ... )
    """
    
    @staticmethod
    async def from_tools(
        name: str,
        llm: LLM,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        system_prompt: str,
        tools: list[Tool]
    ) -> BuilderAgent:
        """
        Create a ToolExecutorAgent with parallel tool execution.
        
        Args:
            name: Unique identifier for the agent
            llm: LLM instance to use for tool selection
            input_model: Pydantic model for input validation
            output_model: Pydantic model for output
            system_prompt: Base system prompt for the agent
            tools: List of Tool instances available to the LLM
            
        Returns:
            BuilderAgent instance ready for optional configuration
        """
        # Closure variables for overrides
        overrides = {}
        tool_set = ToolSet(tools=tools)
        
        async def handler(input_data, context: str, execution_state: ExecutionState):
            """Generated handler for tool execution."""
            # Get context transform function
            context_transform = overrides.get(
                'context_transform',
                lambda ctx: f"\n\nContext: {ctx}" if ctx else ""
            )
            
            # Build system prompt
            if 'system_prompt_builder' in overrides:
                prompt = await overrides['system_prompt_builder'](input_data, context, execution_state)
            else:
                prompt = system_prompt + context_transform(context)
            
            # Get input content
            input_content = getattr(input_data, 'description', getattr(input_data, 'content', str(input_data)))
            
            # Create ExecutionEvent for LLM
            llm_event = ExecutionEvent(actor=llm)
            
            # Prepare LLM request with tools
            llm_input = OpenAIRequest(
                system_message=prompt,
                prompt=input_content,
                model=overrides.get('model', 'gpt-4o'),
                max_completion_tokens=overrides.get('max_tokens', 2500),
                temperature=overrides.get('temperature', 1.0),
                tools=tool_set
            )
            
            # Execute LLM with ExecutionEvent
            execution_result = await llm_event(llm_input, context, execution_state)
            
            # Handle LLM errors
            error_strategy = overrides.get('error_strategy', ErrorStrategy.RAISE)
            if not execution_result.ok or execution_result.result is None:
                error_msg = f"LLM execution failed: {execution_result.erro_message}"
                if error_strategy == ErrorStrategy.RAISE:
                    raise Exception(error_msg)
                else:
                    return None
            
            response = execution_result.result
            
            # Prepare result storage
            all_results = {}
            execution_order = []
            errors = {}
            
            # Handle tool calls if present
            if response.is_tool_call:
                async def execute_tool(tool_call):
                    """Execute a single tool with ExecutionEvent."""
                    tool = tool_set[tool_call.tool_name]
                    if tool is None:
                        return (tool_call.tool_name, None, f"Tool '{tool_call.tool_name}' not found.")
                    
                    try:
                        # Create ExecutionEvent for tool
                        tool_event = ExecutionEvent(actor=tool)
                        
                        # Parse tool arguments
                        tool_input = tool.input_model.model_validate_json(tool_call.arguments)
                        
                        # Execute with ExecutionEvent
                        tool_result = await tool_event(tool_input, context, execution_state)
                        
                        if not tool_result.ok or tool_result.result is None:
                            return (tool_call.tool_name, None, tool_result.erro_message)
                        
                        # Extract output from Tool_Result
                        output = getattr(tool_result.result, 'output', tool_result.result)
                        return (tool_call.tool_name, output, None)
                    except Exception as e:
                        return (tool_call.tool_name, None, str(e))
                
                # Execute all tools in parallel
                tasks = [execute_tool(tc) for tc in response.tools]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for first_result_only mode
                first_only = overrides.get('first_result_only', False)
                
                # Process results
                for item in results:
                    # Skip exceptions from gather
                    if isinstance(item, Exception):
                        continue
                    
                    # Unpack tuple (Type checker doesn't know item is tuple here)
                    tool_name, output, error = item  # type: ignore[misc]
                    execution_order.append(tool_name)
                    
                    if error:
                        errors[tool_name] = error
                    else:
                        all_results[tool_name] = output
                        # If first_result_only, stop after first success
                        if first_only:
                            break
            else:
                # No tool calls, return content as result
                all_results['_content'] = response.content or ""
                execution_order.append('_content')
            
            # Create ToolExecutionResult
            result = ToolExecutionResult(
                results=all_results,
                execution_order=execution_order,
                errors=errors
            )
            
            # Apply aggregation strategy if specified
            aggregation_strategy = overrides.get('aggregation_strategy')
            if aggregation_strategy:
                if aggregation_strategy == AggregationStrategy.LIST:
                    return result.to_list()
                elif aggregation_strategy == AggregationStrategy.CONCATENATE:
                    separator = overrides.get('aggregation_separator', "\n\n")
                    return result.to_concatenated(separator)
                elif aggregation_strategy == AggregationStrategy.FIRST:
                    return result.first()
                # DICT is default, return ToolExecutionResult as-is
            
            return result
        
        # Create BuilderAgent
        agent = BuilderAgent(
            name=name,
            description=system_prompt,
            input_model=input_model,
            output_model=ToolExecutionResult,  # Always returns ToolExecutionResult
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=tool_set
        )
        
        # Store overrides reference
        object.__setattr__(agent, '_overrides', overrides)  # type: ignore
        
        return agent


class DirectToolAgent:
    """
    Builder for agents that execute a single tool directly without LLM.
    
    Creates an agent that bypasses the LLM entirely and directly invokes a tool's
    handler. Useful for deterministic operations, data transformations, or when
    LLM reasoning is not needed.
    
    Usage Pattern:
        Use when you have a specific operation that doesn't require LLM reasoning,
        such as data validation, transformation, database queries, or API calls
        where the parameters are already known.
    
    Examples:
        Basic usage:
        >>> agent = await DirectToolAgent.from_tool(
        ...     name="Validator",
        ...     tool=validation_tool
        ... )
        
        Advanced usage with optional configuration:
        >>> agent = (
        ...     await DirectToolAgent.from_tool(
        ...         name="DatabaseQuery",
        ...         tool=db_tool
        ...     )
        ...     .with_error_strategy(ErrorStrategy.RETURN_NONE)
        ...     .with_retry(max_attempts=3, initial_delay=0.5)
        ...     .as_cacheable(ttl=300)
        ... )
    """
    
    @staticmethod
    async def from_tool(
        name: str,
        tool: Tool
    ) -> BuilderAgent:
        """
        Create a DirectToolAgent for tool execution without LLM.
        
        Args:
            name: Unique identifier for the agent
            tool: Tool instance to execute directly
            
        Returns:
            BuilderAgent instance ready for optional configuration
        """
        # Closure variables for overrides
        overrides = {}
        
        async def handler(input_data, context: str, execution_state: ExecutionState):
            """Generated handler for direct tool execution."""
            # Create ExecutionEvent for tool
            tool_event = ExecutionEvent(actor=tool)
            
            # Execute tool with ExecutionEvent
            tool_result = await tool_event(input_data, context, execution_state)
            
            # Handle errors
            error_strategy = overrides.get('error_strategy', ErrorStrategy.RAISE)
            if not tool_result.ok or tool_result.result is None:
                error_msg = f"Tool execution failed: {tool_result.erro_message}"
                if error_strategy == ErrorStrategy.RAISE:
                    raise Exception(error_msg)
                elif error_strategy == ErrorStrategy.RETURN_NONE:
                    return ExecutionResult[tool.output_model](result=None, ok=False, erro_message=error_msg)
                elif error_strategy == ErrorStrategy.LOG_AND_CONTINUE:
                    await execution_state.log(f"Warning: {error_msg}")
                    return None
                else:  # SILENT
                    return None
            
            return tool_result.result
        
        # Create BuilderAgent
        agent = BuilderAgent(
            name=name,
            description=tool.description,
            input_model=tool.input_model,
            output_model=tool.output_model,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=None,  # No LLM for direct tool execution
            tools=ToolSet(tools=[tool])
        )
        
        # Store overrides reference
        object.__setattr__(agent, '_overrides', overrides)  # type: ignore
        
        return agent
