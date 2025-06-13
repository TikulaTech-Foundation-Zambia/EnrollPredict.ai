"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .routes import chat, web
from .dependencies.config import get_settings

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="EnrollPredict.ai",
        description="AI-powered enrollment prediction application",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Get base directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
    
    # Include routers
    app.include_router(web.router, tags=["web"])
    app.include_router(chat.router, tags=["chat"])
    
    return app

app = create_app()
