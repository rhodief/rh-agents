import asyncio
from typing import Callable, Union
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.result_types import LLM_Result
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.execution import EventBus, ExecutionState
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message
from rh_agents.openai import OpenAIRequest, openai_handler


# ═══════════════════════════════════════════════════════════════════════════════
# Beautiful Event Printer
# ═══════════════════════════════════════════════════════════════════════════════

class EventPrinter:
    """Pretty printer for execution events with colors and formatting."""
    
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # Status colors and symbols
    STATUS_CONFIG = {
        ExecutionStatus.STARTED: ("▶", CYAN, "STARTED"),
        ExecutionStatus.COMPLETED: ("✔", GREEN, "COMPLETED"),
        ExecutionStatus.FAILED: ("✖", RED, "FAILED"),
        ExecutionStatus.AWAITING: ("⏳", YELLOW, "AWAITING"),
        ExecutionStatus.HUMAN_INTERVENTION: ("👤", MAGENTA, "HUMAN"),
    }
    
    # Event type icons
    EVENT_ICONS = {
        EventType.AGENT_CALL: "🤖",
        EventType.TOOL_CALL: "🔧",
        EventType.LLM_CALL: "🧠",
    }
    
    def __init__(self, show_timestamp: bool = True, show_address: bool = True):
        self.show_timestamp = show_timestamp
        self.show_address = show_address
        self.indent_cache: dict[str, int] = {}
    
    def _get_indent_level(self, address: str) -> int:
        """Calculate indentation based on address depth."""
        if not address:
            return 0
        return address.count("::")
    
    def _format_time(self, execution_time: float | None) -> str:
        """Format execution time nicely."""
        if execution_time is None:
            return ""
        if execution_time < 0.001:
            return f"{execution_time * 1_000_000:.0f}μs"
        elif execution_time < 1:
            return f"{execution_time * 1000:.1f}ms"
        else:
            return f"{execution_time:.2f}s"
    
    def _truncate(self, text: str, max_len: int = 50) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def print_event(self, event: ExecutionEvent):
        """Print a beautifully formatted event."""
        status = event.execution_status
        symbol, color, status_text = self.STATUS_CONFIG.get(
            status, ("?", self.WHITE, "UNKNOWN")
        )
        
        event_icon = self.EVENT_ICONS.get(event.actor.event_type, "📌")
        indent_level = self._get_indent_level(event.address)
        indent = "  │ " * indent_level
        
        # Build the output
        lines = []
        
        # Main event line
        actor_name = event.actor.name
        time_str = self._format_time(event.execution_time)
        time_display = f" {self.GRAY}({time_str}){self.RESET}" if time_str else ""
        
        main_line = (
            f"{self.GRAY}{indent}{self.RESET}"
            f"{color}{self.BOLD}{symbol}{self.RESET} "
            f"{event_icon} "
            f"{self.BOLD}{actor_name}{self.RESET} "
            f"{color}[{status_text}]{self.RESET}"
            f"{time_display}"
        )
        lines.append(main_line)
        
        # Address line (if enabled and has content)
        if self.show_address and event.address:
            address_line = (
                f"{self.GRAY}{indent}  ├─ 📍 {event.address}{self.RESET}"
            )
            lines.append(address_line)
        
        # Timestamp line (if enabled)
        if self.show_timestamp:
            timestamp = event.datetime[:19].replace("T", " ")  # Trim to readable format
            time_line = (
                f"{self.GRAY}{indent}  ├─ 🕐 {timestamp}{self.RESET}"
            )
            lines.append(time_line)
        
        # Error message (if failed)
        if status == ExecutionStatus.FAILED and event.message:
            error_msg = self._truncate(event.message, 80)
            error_line = (
                f"{self.GRAY}{indent}  {self.RESET}"
                f"{self.RED}└─ ⚠️  {error_msg}{self.RESET}"
            )
            lines.append(error_line)
        else:
            # Closing line
            lines.append(f"{self.GRAY}{indent}  └{'─' * 40}{self.RESET}")
        
        # Print all lines
        print("\n".join(lines))
    
    def __call__(self, event: ExecutionEvent):
        """Allow using the printer as a callback."""
        self.print_event(event)


def create_event_handler(printer: EventPrinter | None = None) -> Callable:
    """Factory to create an event handler with optional custom printer."""
    if printer is None:
        printer = EventPrinter()
    return printer


class DoctrineStep(BaseModel):    
    index: int
    description: str
    feasible: bool
    required_steps: list[int] = Field(default_factory=list)
    
class Doctrine(BaseModel):
    goal: str
    constraints: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)
    steps: list[DoctrineStep] = Field(default_factory=list)
    

class OpenAILLM(LLM[OpenAIRequest]):
    """OpenAI LLM Actor with function calling support"""
    
    def __init__(
        self,
        name: str = "OpenAI-LLM",
        description: str = "OpenAI GPT model with function calling capabilities",        
    ):
        async def handler_wrapper(request: OpenAIRequest, extra_context: str, execution_state) -> LLM_Result:
            return await openai_handler(request)
        
        super().__init__(
            name=name,
            description=description,
            input_model=OpenAIRequest,
            output_model=LLM_Result,
            handler=handler_wrapper,
            event_type=EventType.LLM_CALL
        )
        
     

    
class DoctrineTool(Tool):
    def __init__(self) -> None:
        DOCTRINE_TOOL_PROMPT = '''
        Analisa o pedido do usuário e gera um plano estruturado
        com objetivo e passos executáveis por subagentes.
        Cada passo deve conter uma única ação clara e objetiva.
        O índice de cada passo deve ser único e sequencial, começando em 0.
        '''
        
        super().__init__(
            name="DoctrineTool",
            description=DOCTRINE_TOOL_PROMPT,
            input_model=Message,
            output_model=Doctrine,
            handler=lambda args: args,
            event_type=EventType.TOOL_CALL
        )
    

class DoctrineReceverAgent(Agent):
    def __init__(self,
                 llm: LLM,
                 tools: Union[list[Tool], None] = None
                 ) -> None:
        INTENT_PARSER_PROMPT = '''
        Você é um analisador de intenções. 
        Use a ferramenta disponível para estruturar o pedido do usuário.
        Considere que cada passo será passado para um subagente, utilize linguagem clara e objetiva.
        Gere uma resposta estruturada com:
        - goal: objetivo principal (string)
        - steps: lista de passos com índice sequencial, descrição clara e viabilidade (array de objetos)
        - constraints: limitações se houver (SEMPRE array de strings, mesmo se vazio)
        - guidelines: diretrizes se necessário (SEMPRE array de strings, mesmo se vazio)
        
        IMPORTANTE: constraints e guidelines DEVEM ser arrays/listas, nunca strings simples.
        Exemplo: "constraints": ["limitação 1", "limitação 2"] ou "constraints": []
    '''
    
        async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Union[Doctrine, Message]:
            llm_event = ExecutionEvent[llm.output_model](
                actor=llm
            )            
            # Execute LLM to parse the user input into a Doctrine
            llm_input = OpenAIRequest(
                system_message=INTENT_PARSER_PROMPT + f'\nContexto de Execuções anteriores: {context}',
                prompt=input_data.content,
                model="gpt-5-mini",
                tools=ToolSet([DoctrineTool()]),
            )
            execution_result = await llm_event(llm_input, context, execution_state)
            
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            result = execution_result.result
            if result.is_content:
                return Message(content=result.content, author=AuthorType.ASSISTANT)
            
            if not (result.is_tool_call and result.tools and result.tools[0]):
                raise Exception("LLM did not return a valid tool call for DoctrineTool.")
                
            tool_call = result.tools[0]
            return Doctrine.model_validate_json(tool_call.arguments)
        
        super().__init__(
            name="DoctrineReceverAgent",
            description=INTENT_PARSER_PROMPT,
            input_model=Message,
            output_model=Doctrine,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools) if tools else ToolSet()
        )


if __name__ == "__main__":
    llm = OpenAILLM()
    agent = DoctrineReceverAgent(llm=llm)
    message = Message(content="Organize uma festa de aniversário surpresa com decoração, comida e música.", author=AuthorType.USER)
    
    # Create beautiful event printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    
    bus = EventBus()
    bus.subscribe(printer)  # Use the beautiful printer
    agent_execution_state = ExecutionState(event_bus=bus)
    
    async def streamer():
        async for event in bus.event_stream():
            pass  # Events already printed by subscriber
    
    async def main():
        print(f"\n{'═' * 60}")
        print(f"{'🚀 EXECUTION STARTED':^60}")
        print(f"{'═' * 60}\n")
        
        await asyncio.gather(
            ExecutionEvent(actor=agent)(message, "", agent_execution_state),
            streamer()        
        )
        
        print(f"\n{'═' * 60}")
        print(f"{'✅ EXECUTION FINISHED':^60}")
        print(f"{'═' * 60}\n")
    
    asyncio.run(main())
        