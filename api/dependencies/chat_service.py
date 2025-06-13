"""
Chat service dependency.
"""
from functools import lru_cache
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional, Dict, Any

from .config import get_settings
from src.models.agent import Agent
from src.models.tools_new import predict_enrollment

class ChatService:
    """Service for handling chat interactions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.memory = InMemorySaver()
        self.model = ChatGroq(
            temperature=0,
            model=self.settings.model_name,
            api_key=self.settings.groq_api_key
        )
        self.tools = [predict_enrollment]
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            memory=self.memory,
            system="You are a helpful AI assistant specializing in enrollment predictions. "
                   "You have access to machine learning tools to help users predict "
                   "admission chances based on academic credentials. Be helpful, "
                   "accurate, and provide clear explanations of your predictions."
        )
    
    async def get_response(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get a response from the AI agent."""
        try:
            # Use the agent to process the message
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Extract the response content
            if response and "messages" in response:
                last_message = response["messages"][-1]
                return last_message.content
            
            return "I'm sorry, I couldn't process your request at the moment."
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

@lru_cache()
def get_chat_service() -> ChatService:
    """Get cached chat service instance."""
    return ChatService()
