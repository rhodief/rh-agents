from pydantic import BaseModel
from rh_agents.core.actors import BaseActor
from rh_agents.agents import Doctrine
from rh_agents.core.execution import ExecutionState


class OrchestrationState(BaseModel):
    doctrine: Doctrine
    execution_state: ExecutionState | None = None
    
    
class Orchestrator(BaseActor):
    name: str
    description: str
    state: OrchestrationState | None = None
    
    async def __call__(self, input_data, orchestration_state: OrchestrationState):
        raise NotImplementedError("Orchestrator must implement its own __call__ method.")
    
    