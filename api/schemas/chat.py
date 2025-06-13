"""
Chat request and response schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
from datetime import datetime

class ChatRequest(BaseModel):
    """Schema for chat requests."""
    message: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[Union[datetime, str]] = None

class ChatResponse(BaseModel):
    """Schema for chat responses."""
    response: str
    session_id: str
    timestamp: Optional[datetime] = None
