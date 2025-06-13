"""
Web page routes for the application.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ..dependencies.config import get_settings

router = APIRouter()
settings = get_settings()

# Setup templates
templates = Jinja2Templates(directory=str(settings.web_dir / "templates"))

@router.get("/")
async def root(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/landing")
async def landing(request: Request):
    """Serve the landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})

@router.get("/home")
async def home(request: Request):
    """Serve the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/about")
async def about_page(request: Request):
    """Serve the about page."""
    return templates.TemplateResponse("about.html", {"request": request})

@router.get("/chat")
async def chat_page(request: Request):
    """Serve the chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})

@router.get("/developers")
async def developers_page(request: Request):
    """Serve the developers page."""
    return templates.TemplateResponse("developers.html", {"request": request})
