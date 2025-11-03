from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class InstructionRecord(BaseModel):
    id: str
    timestamp: Optional[str] = None
    source: Optional[str] = None
    instruction: str
    input: Optional[str] = ""
    output: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FaultRecord(BaseModel):
    id: str
    timestamp: Optional[str] = None
    source: Optional[str] = None
    text: str
    fault_type: Optional[str] = None
    labels: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
