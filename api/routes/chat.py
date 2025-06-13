"""
Chat API routes.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import uuid
import json

from ..schemas.chat import ChatRequest, ChatResponse
from ..dependencies.chat_service import get_chat_service

router = APIRouter()

# In-memory storage for chat sessions (in production, use a proper database)
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

@router.post("/chat/debug")
async def debug_chat(request: Request):
    """Debug endpoint to see raw request data."""
    body = await request.body()
    try:
        json_data = json.loads(body)
        return {"received": json_data, "body_type": type(json_data).__name__}
    except Exception as e:
        return {"error": str(e), "raw_body": body.decode()}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests and return AI responses."""
    try:
        print(f"Received chat request: {request}")  # Debug logging
        print(f"Request data: message='{request.message}', session_id='{request.session_id}'")
        
        chat_service = get_chat_service()
        
        # Get or create session
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        
        # Add user message to session
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": request.timestamp
        }
        chat_sessions[request.session_id].append(user_message)
        
        # Get AI response
        response = await chat_service.get_response(
            message=request.message,
            session_id=request.session_id,
            context=request.context
        )
        
        # Add AI response to session
        ai_message = {
            "role": "assistant",
            "content": response,
            "timestamp": None  # Will be set by the frontend
        }
        chat_sessions[request.session_id].append(ai_message)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
        
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid request data: {str(e)}")
    except Exception as e:
        print(f"Chat service error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    return chat_sessions.get(session_id, [])

@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": "Chat history cleared"}
