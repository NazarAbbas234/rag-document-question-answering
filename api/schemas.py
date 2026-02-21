
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = "openai"


class QueryResponse(BaseModel):
    summary: str
    clause_type: str
    risk_level: str
    explanation: str
    model_used: str


class IngestResponse(BaseModel):
    status: str
    message: str