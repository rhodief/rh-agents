import asyncio
from typing import Union
from rh_agents.core.actors import LLM, Agent, Tool, ToolSet
from rh_agents.core.result_types import LLM_Result, Tool_Result
from rh_agents.core.types import EventType
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from pydantic import BaseModel, Field
from rh_agents.models import AuthorType, Message
from rh_agents.openai import OpenAIRequest, openai_handler



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

class OpenAILLM(LLM):
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
        
        async def doctrine_handler(args: Doctrine, context: str, state: ExecutionState) -> Doctrine:
            return args
        
        super().__init__(
            name="DoctrineTool",
            description=DOCTRINE_TOOL_PROMPT,
            input_model=Doctrine,
            handler=doctrine_handler,
            cacheable=True            
        )

class ListPecasArgs(BaseModel):
    processo: int = Field(..., description="Número do processo judicial")
    tipo_peca: str = Field(..., description="Tipo da peça judicial, ex: DEC_ADM, ARESP")

class GetTextoPecaArgs(BaseModel):
    id_peca: str = Field(..., description="ID da peça")


 

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
            llm_event = ExecutionEvent(
                actor=llm
            )            
            # Execute LLM to parse the user input into a Doctrine
            llm_input = OpenAIRequest(
                system_message=INTENT_PARSER_PROMPT + f'\nContexto de Execuções anteriores: {context}',
                prompt=input_data.content,
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                temperature=1,
                tools=ToolSet(tools=tools if tools else []),
                tool_choice={"type": "function", "function": {"name": "DoctrineTool"}}
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
            tools=ToolSet(tools=tools) if tools else ToolSet(tools=[]),
            is_artifact=True,
            cacheable=True
        )

class StepExecutorAgent(Agent):
    def __init__(self,
                 llm: LLM,
                 tools: Union[list[Tool], None] = None
                 ) -> None:
        STEP_EXECUTOR_PROMPT = '''
            Você é um executor de passos.
            Execute o passo fornecido de acordo com o plano de execução e o objetivo geral.
            Se você já tiver informações necessárias em seu contexto, não chame ferramentas desnecessariamente.
            '''
        tool_set = ToolSet(tools=tools) if tools else ToolSet(tools=[])
        async def handler(input_data: DoctrineStep, context: str, execution_state: ExecutionState) -> StepResult:
            llm_event = ExecutionEvent(
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
                temperature=1,
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
                            output = getattr(tool_result.result, 'output', tool_result.result)
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
                        result=None,
                        execution_time=None,
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
                    execution_time=None,
                    ok=True,
                    erro_message=None
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
            tools=ToolSet(tools=tools) if tools else ToolSet(tools=[])
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
        tool_set = ToolSet(tools=tools) if tools else ToolSet(tools=[])
        
        async def handler(input_data: Doctrine, context: str, execution_state: ExecutionState) -> Message:
            llm_event = ExecutionEvent(
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
                temperature=1,
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
            tools=ToolSet(tools=tools) if tools else ToolSet(tools=[])
        )

class OmniAgent(Agent):
    def __init__(self,
                 receiver_agent: Agent,
                 step_executor_agent: Agent,
                 reviewer_agent: Agent,
                 ) -> None:
        
        
        async def handler(input_data: Message, context: str, execution_state: ExecutionState) -> Message:
            result = await ExecutionEvent(actor=receiver_agent)(input_data, "", execution_state)
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
                step_result = await ExecutionEvent(actor=step_executor_agent, tag=f'step_{step.index}-{len(doctrine.steps) - 1}')(step, context, execution_state)
                if not step_result.ok or step_result.result is None:
                    raise Exception(f"Step execution failed: {step_result.erro_message}")
                execution_state.add_step_result(step.index, step_result.result)
            
            # After all steps complete, call the ReviewerAgent to generate final results
            print("\n" + "═" * 60)
            print("🔍 Iniciando Revisão Final...")
            print("═" * 60 + "\n")
            
            review_result = await ExecutionEvent(actor=reviewer_agent, tag='final_review')(doctrine, '', execution_state)
            if not review_result.ok or review_result.result is None:
                raise Exception(f"Review execution failed: {review_result.erro_message}")
            
            final_message = review_result.result
            print("\n" + "═" * 60)
            print("📄 RELATÓRIO FINAL")
            print("═" * 60)
            print(final_message.content)
            print("═" * 60 + "\n")
            
            return final_message
            
        
        super().__init__(
            name="OmniAgent",
            description='Omni Agent que orquestra os subagentes de recepção, execução de passos e revisão final',
            input_model=Message,
            output_model=Message,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            tools=ToolSet(tools=[]),
            llm=None
        )
