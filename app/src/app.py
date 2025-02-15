from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from .agent import Agent
from pathlib import Path


app = FastAPI()

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent



# Make sure this line is before any route definitions
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup template and static directories
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize your agent
model = ChatGroq(temperature=0)
tools = [TavilySearchResults(max_results=1)]
agent = Agent(model=model, tools=tools)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/landing")
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message")
    
    # Process the message through your agent
    result = agent.graph.invoke({"messages": [{"role": "user", "content": message}]})
    
    return JSONResponse(content={"response": result["messages"][-1].content})

@app.get("/developers")
async def developers_page(request: Request):
    return templates.TemplateResponse("developers.html", {"request": request})




