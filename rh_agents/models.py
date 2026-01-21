from enum import Enum
from pydantic import BaseModel

class AuthorType(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

class ArtifactRef(BaseModel):
    id_ref: str
    type_ref: str
    title: str
    value: str | None = None
    tags: list[str] = []
    description: str | None = None

class Message(BaseModel):
    content: str
    author: AuthorType
    artifact_refs: list[ArtifactRef] | None = None
    
