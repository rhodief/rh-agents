from enum import Enum
from pydantic import BaseModel

class AuthorType(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

class Message(BaseModel):
    content: str
    author: AuthorType