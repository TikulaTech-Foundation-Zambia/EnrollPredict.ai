"""
Application configuration and settings.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_title: str = "EnrollPredict.ai"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Model Settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = "llama-3.3-70b-versatile"
    
    # LangSmith Settings
    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    
    # Tavily API Key
    tavily_api_key: str = ""
    
    # File Paths
    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"
    web_dir: Path = BASE_DIR / "web"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
