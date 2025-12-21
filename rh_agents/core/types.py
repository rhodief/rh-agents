from enum import Enum


class EventType(str, Enum):
    AGENT_CALL = 'agent_call'
    TOOL_CALL = 'tool_call'
    LLM_CALL = 'llm_call'


class ExecutionStatus(str, Enum):
    STARTED = 'started'
    COMPLETED = 'completed'
    FAILED = 'failed'
    AWAITING = 'awaiting'
    HUMAN_INTERVENTION = 'human_intervention'
