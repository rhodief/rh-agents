import asyncio
from typing import Callable, Union
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.result_types import LLM_Result, Tool_Result
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import EventBus, ExecutionState
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message
from rh_agents.openai import OpenAIRequest, openai_handler


DOC_LIST = {
    "DEC_ADM": [{"nome": "Decisão de Admissibilidade - João da Silva vs Banco Alfa S/A", "id": "DEC_ADM_0001"}],
    "ARESP": [{"nome": "Agravo em Recurso Especial - João da Silva vs Banco Alfa S/A", "id": "ARESP_0001"}]
}

DOCS = {
    "DEC_ADM_0001": '''
Decisão de Admissibilidade (Tribunal de Origem) n. DEC_ADM_0001

Recorrente: João da Silva
Recorrido: Banco Alfa S/A

DECISÃO DE ADMISSIBILIDADE DE RECURSO ESPECIAL

Vistos.

Trata-se de Recurso Especial interposto por João da Silva, com fundamento no art. 105, III, “a” e “c”, da Constituição Federal, contra acórdão proferido pela Xª Câmara de Direito Privado deste Tribunal.

No recurso, sustenta o recorrente, em síntese:
(i) violação aos arts. 421 e 422 do Código Civil, sob o argumento de que o acórdão recorrido teria afastado a boa-fé objetiva na interpretação contratual;
(ii) divergência jurisprudencial quanto à possibilidade de revisão de cláusula contratual em contrato bancário.

É o relatório.

FUNDAMENTAÇÃO

O recurso não comporta admissibilidade.

Quanto à alegada violação aos arts. 421 e 422 do Código Civil, verifica-se que o exame da pretensão recursal demandaria revolvimento do conjunto fático-probatório, notadamente quanto à análise das cláusulas contratuais e da conduta das partes, providência vedada em sede de Recurso Especial, nos termos da Súmula 7 do STJ.

No que tange à divergência jurisprudencial, observa-se que o recorrente não realizou o necessário cotejo analítico, limitando-se à transcrição de ementas, sem demonstrar a similitude fática entre os julgados confrontados, em afronta ao disposto no art. 1.029, §1º, do CPC e ao art. 255 do RISTJ.

Ademais, o acórdão recorrido encontra-se em consonância com a jurisprudência dominante do Superior Tribunal de Justiça, incidindo, por analogia, o óbice da Súmula 83 do STJ.

DISPOSITIVO

Ante o exposto, NEGO SEGUIMENTO AO RECURSO ESPECIAL.

Intime-se.

São Paulo, 10 de março de 2024.

Desembargador Fulano de Tal
Vice-Presidente do Tribunal de Justiça
    ''',
    "ARESP_0001": '''
Agravo em Recurso Especial (AREsp) – Análise Cotejada n. ARESP_0001

Agravante: João da Silva
Agravado: Banco Alfa S/A

AGRAVO EM RECURSO ESPECIAL

(art. 1.042 do CPC)

EGRÉGIO SUPERIOR TRIBUNAL DE JUSTIÇA

João da Silva, já qualificado nos autos, inconformado com a decisão que negou seguimento ao Recurso Especial, vem interpor o presente AGRAVO EM RECURSO ESPECIAL, pelas razões a seguir expostas.

I – DA DECISÃO AGRAVADA

A decisão agravada negou seguimento ao Recurso Especial sob os fundamentos de:
(a) incidência da Súmula 7/STJ;
(b) ausência de cotejo analítico;
(c) aplicação da Súmula 83/STJ.

Todavia, tais fundamentos não se sustentam, conforme se demonstrará.

II – DO NÃO CABIMENTO DA SÚMULA 7/STJ

O Recurso Especial não pretende o reexame de fatos ou provas, mas tão somente a revaloração jurídica de fatos incontroversos, expressamente reconhecidos no acórdão recorrido.

O Tribunal de origem reconheceu que:

“as cláusulas contratuais impõem obrigações excessivamente onerosas ao consumidor” (fl. XXX).

Ainda assim, afastou a aplicação dos arts. 421 e 422 do Código Civil, o que configura erro de subsunção jurídica, plenamente revisável em Recurso Especial, conforme jurisprudência pacífica do STJ.

III – DO DEVIDO COTEJO ANALÍTICO (DIVERGÊNCIA JURISPRUDENCIAL)

Diferentemente do afirmado na decisão agravada, o recorrente realizou cotejo analítico adequado, conforme se observa:

Acórdão recorrido: afastou a revisão contratual mesmo diante de desequilíbrio reconhecido.

Acórdão paradigma (REsp nº 1.234.567/RS): admitiu a revisão contratual em hipótese idêntica, com base nos arts. 421 e 422 do CC.

Ambos os julgados tratam de contrato bancário, com cláusulas de idêntica natureza, e discutem a incidência da boa-fé objetiva, estando configurada a similitude fática exigida pelo art. 1.029, §1º, do CPC.

IV – DA INAPLICABILIDADE DA SÚMULA 83/STJ

A Súmula 83/STJ não se aplica ao caso, pois há divergência atual e específica no âmbito do próprio STJ acerca da extensão da revisão contratual em contratos bancários, especialmente quando reconhecido o desequilíbrio contratual no acórdão recorrido.

V – DO PEDIDO

Diante do exposto, requer-se:

a) o conhecimento e provimento do presente Agravo em Recurso Especial, para que seja destrancado o Recurso Especial;
b) o posterior provimento do Recurso Especial, reformando-se o acórdão recorrido.

Termos em que,
Pede deferimento.

Brasília, 25 de março de 2024.
'''
}




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

MODEL = 'gpt-4o'
MAX_TOKENS = 2500

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
    
class StepResult(BaseModel):
    step_index: int
    result: ExecutionResult[str]

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
            input_model=Doctrine,
            handler=lambda args: args
        )

class ListPecasArgs(BaseModel):
    processo: int = Field(..., description="Número do processo judicial")
    tipo_peca: str = Field(..., description="Tipo da peça judicial, ex: DEC_ADM, ARESP")

class GetTextoPecaArgs(BaseModel):
    id_peca: str = Field(..., description="ID da peça")


 
class ListPecasTool(Tool):
    def __init__(self) -> None:
        LISTA_PECAS_TOOL_PROMPT = '''
        Obtém uma lista de peças (nome e id) baseado no tipo_peca que pode ser: DEC_ADM para decisão de admissibilidade e ARESP para agravo em recurso especial
        '''
        
        async def handler(args: ListPecasArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOC_LIST.get(args.tipo_peca, [])
            return Tool_Result(output=result, tool_name="lista_pecas_por_tipo")
        
        super().__init__(
            name="lista_pecas_por_tipo",
            description=LISTA_PECAS_TOOL_PROMPT,
            input_model=ListPecasArgs,
            handler=handler
        )
    
class GetTextoPecaTool(Tool):
    def __init__(self) -> None:
        GET_TEXTO_PECA_TOOL_PROMPT = '''
        Obtém o texto completo e alguns metadados de uma peça judicial baseado no id da peça
        Utilize essa ferramenta quando precisar acessar o inteiro teor do texto para alguma análise ou sumarização.
        '''
        
        async def handler(args: GetTextoPecaArgs, context: str, execution_state: ExecutionState) -> Tool_Result:
            result = DOCS.get(args.id_peca, "Peça não encontrada.")
            return Tool_Result(output=result, tool_name="get_texto_peca")
        
        super().__init__(
            name="get_texto_peca",
            description=GET_TEXTO_PECA_TOOL_PROMPT,
            input_model=GetTextoPecaArgs,
            handler=handler
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
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                tools=ToolSet(tools if tools else []),
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

class StepExecutorAgent(Agent):
    def __init__(self,
                 llm: LLM,
                 tools: Union[list[Tool], None] = None
                 ) -> None:
        STEP_EXECUTOR_PROMPT = '''
            Você é um executor de passos.
            Execute o passo fornecido de acordo com o plano de execução e o objetivo geral.
            '''
        tool_set = ToolSet(tools) if tools else ToolSet()
        async def handler(input_data: DoctrineStep, context: str, execution_state: ExecutionState) -> StepResult:
            llm_event = ExecutionEvent[llm.output_model](
                actor=llm
            )           
            # Retrieve dependencies from the datastore (execution_state)
            dependencies_list = execution_state.get_steps_result(input_data.required_steps)
            #print('STEP_EXECUTOR_AGENT - DEPENDENCIES LIST', input_data.required_steps, dependencies_list)
            dependencies_context = '\n'.join(dependencies_list) if dependencies_list else 'Nenhuma execução anterior.'
            
            # Execute LLM to execute the step
            system_context = STEP_EXECUTOR_PROMPT + f'\nCONTEXTO: O Processo corrente é o 123456789\n\n{context}\n\nExecuções anteriores:\n{dependencies_context}'
            #print('SYSTEM CONTEXT', system_context)
            #print('USER_PROMPT', input_data.description)
            llm_input = OpenAIRequest(
                system_message=system_context,
                prompt=input_data.description,
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                tools=tool_set
            )
            execution_result = await llm_event(llm_input, context, execution_state)
            #print('STEP_EXECUTOR_AGENT - LLM EXECUTION RESULT', execution_result)
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            
            response = execution_result.result
            all_outputs = []
            errors = []
            if response.is_tool_call:
                for tool_call in response.tools:
                    tool = tool_set[tool_call.tool_name]
                    if tool is None:
                        errors.append(f"Tool '{tool_call.tool_name}' not found.")
                        continue                    
                    try:
                        tool_event = ExecutionEvent(
                            actor=tool
                        )
                        tool_input = tool.input_model.model_validate_json(tool_call.arguments)
                        tool_result = await tool_event(tool_input, context, execution_state)
                        
                        if not tool_result.ok or tool_result.result is None:
                            errors.append(f"Tool '{tool_call.tool_name}' execution failed: {tool_result.erro_message}")
                        else:
                            # Extract the output from Tool_Result
                            output = tool_result.result.output if hasattr(tool_result.result, 'output') else tool_result.result
                            all_outputs.append(str(output))
                    except Exception as e:
                        errors.append(f"Error in {tool_call.tool_name}: {str(e)}")
                # Return combined results
            else:
                all_outputs.append(response.content or "")
            if errors and not all_outputs:
                return StepResult(
                    step_index=input_data.index,
                    result=ExecutionResult[str](
                        ok=False,
                        erro_message="; ".join(errors)
                    )
                )
            
            combined_output = "\n".join(all_outputs)
            if errors:
                combined_output += f"\nErrors: {'; '.join(errors)}"
            
            return StepResult(
                step_index=input_data.index,
                result=ExecutionResult[str](
                    result=combined_output,
                    ok=True
                )
            )
        
        super().__init__(
            name="StepExecutorAgent",
            description=STEP_EXECUTOR_PROMPT,
            input_model=Doctrine,
            output_model=StepResult,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools) if tools else ToolSet()
        )


class ReviewerAgent(Agent):
    def __init__(self,
                 llm: LLM,
                 tools: Union[list[Tool], None] = None
                 ) -> None:
        REVIEWER_AGENT_PROMPT = '''
            Você é um revisor especializado em análise jurídica.
            Com base nos resultados das etapas anteriores, elabore uma resposta final completa e bem estruturada.
            Sintetize as informações coletadas e apresente um relatório coeso ao usuário.
            Use linguagem clara e técnica apropriada ao contexto jurídico.
            '''
        tool_set = ToolSet(tools) if tools else ToolSet()
        
        async def handler(input_data: Doctrine, context: str, execution_state: ExecutionState) -> Message:
            llm_event = ExecutionEvent[llm.output_model](
                actor=llm
            )
            
            # Get ALL results from the store
            all_step_results = execution_state.get_all_steps_results()
            
            # Build comprehensive context from all step results
            results_context = "\n\n".join([
                f"Resultado da Etapa {idx}:\n{result}"
                for idx, result in all_step_results.items()
            ]) if all_step_results else "Nenhum resultado anterior disponível."
            
            # Build system context with goal, guidelines, constraints, and all results
            system_context = (
                f"{REVIEWER_AGENT_PROMPT}\n\n"
                f"OBJETIVO GERAL: {input_data.goal}\n\n"
                f"DIRETRIZES: {', '.join(input_data.guidelines) if input_data.guidelines else 'Nenhuma'}\n\n"
                f"RESTRIÇÕES: {', '.join(input_data.constraints) if input_data.constraints else 'Nenhuma'}\n\n"
                f"RESULTADOS DAS ETAPAS EXECUTADAS:\n{results_context}"
            )
            
            user_prompt = (
                f"Com base nos resultados das {len(all_step_results)} etapas executadas acima, "
                f"elabore um relatório final completo e estruturado sobre: {input_data.goal}"
            )
            
            llm_input = OpenAIRequest(
                system_message=system_context,
                prompt=user_prompt,
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                tools=tool_set
            )
            
            execution_result = await llm_event(llm_input, context, execution_state)
            
            if not execution_result.ok or execution_result.result is None:
                raise Exception(f"LLM execution failed: {execution_result.erro_message}")
            
            response = execution_result.result
            final_content = response.content or "Não foi possível gerar uma revisão final."
            
            return Message(content=final_content, author=AuthorType.ASSISTANT)
        
        super().__init__(
            name="ReviewerAgent",
            description=REVIEWER_AGENT_PROMPT,
            input_model=Doctrine,
            output_model=Message,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools) if tools else ToolSet()
        )



if __name__ == "__main__":
    llm = OpenAILLM()
    tools = [DoctrineTool(), ListPecasTool(), GetTextoPecaTool()]
    tools_2 = [ListPecasTool(), GetTextoPecaTool()]
    doctrine_receiver_agent = DoctrineReceverAgent(llm=llm, tools=tools)
    step_executor_agent = StepExecutorAgent(llm=llm, tools=tools_2)
    reviewer_agent = ReviewerAgent(llm=llm, tools=[])
    msg = 'Faça um relatório para a Análise da Admissibilidade Cotejada de modo a extrair os óbices jurídicos da decisão de Admissibilidade e verificar o respectivo rebatimento no agravo de Recurso Especial correspondente'
    message = Message(content=msg, author=AuthorType.USER)
    
    # Create beautiful event printer
    printer = EventPrinter(show_timestamp=True, show_address=True)
    
    bus = EventBus()
    bus.subscribe(printer)  # Use the beautiful printer
    agent_execution_state = ExecutionState(event_bus=bus)
    
    async def streamer():
        async for event in bus.event_stream():
            pass  # Events already printed by subscriber
    
    async def run_agent():
        result = await ExecutionEvent[Union[Doctrine, Message]](actor=doctrine_receiver_agent)(message, "", agent_execution_state)
        if not result.ok or result.result is None:
            raise Exception(f"Agent execution failed: {result.erro_message}")
        if isinstance(result.result, Message):
            return result.result
        doctrine = result.result
        for step in doctrine.steps:
            if not step.feasible:
                raise Exception(f"Step {step.index} not feasible")
            # Build context from the goal, guidelines, and constraints (not dependencies yet)
            #context = f'Goal: {doctrine.goal}\nGuidelines: {doctrine.guidelines}\nConstraints: {doctrine.constraints}'
            context = ''
            # Execute step - the handler will retrieve dependencies from execution_state
            step_result = await ExecutionEvent[StepResult](actor=step_executor_agent, tag=f'step_{step.index}-{len(doctrine.steps) - 1}')(step, context, agent_execution_state)
            if not step_result.ok or step_result.result is None:
                raise Exception(f"Step execution failed: {step_result.erro_message}")
            agent_execution_state.add_step_result(step.index, step_result.result)
        
        # After all steps complete, call the ReviewerAgent to generate final results
        print("\n" + "═" * 60)
        print("🔍 Iniciando Revisão Final...")
        print("═" * 60 + "\n")
        
        review_result = await ExecutionEvent[Message](actor=reviewer_agent, tag='final_review')(doctrine, '', agent_execution_state)
        if not review_result.ok or review_result.result is None:
            raise Exception(f"Review execution failed: {review_result.erro_message}")
        
        final_message = review_result.result
        print("\n" + "═" * 60)
        print("📄 RELATÓRIO FINAL")
        print("═" * 60)
        print(final_message.content)
        print("═" * 60 + "\n")
        
        return final_message
        
        
            
            
        
        
    
    async def main():
        print(f"\n{'═' * 60}")
        print(f"{'🚀 EXECUTION STARTED':^60}")
        print(f"{'═' * 60}\n")
        
        await asyncio.gather(
            run_agent(),
            streamer()        
        )
        
        print(f"\n{'═' * 60}")
        print(f"{'✅ EXECUTION FINISHED':^60}")
        print(f"{'═' * 60}\n")
    
    asyncio.run(main())
        