from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .agent import Agent
from .tools import  predict_enrollment
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()


app = FastAPI()

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Make sure this line is before any route definitions
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup template and static directories
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize your agent
model = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
tools = [predict_enrollment]  # Using the more reliable tool for now
agent = Agent(model=model, tools=tools, 
              memory=memory,
              system="You are a helpful assistant, assist the user with their questions. Use have access to tools to solve the users queries. ")

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
    
    # Create a human message from the input
    human_message = HumanMessage(content=message)
    
    try:
        # Process the message through your agent
        result = agent.graph.invoke({
            "messages": [human_message]
        }, config={"configurable": {"thread_id": "1"}})
        
        
        # Extract the last message content
        final_message = result["messages"][-1].content if result["messages"] else "I couldn't process your request."
        
    except Exception as e:
        print(f"Error in processing: {e}")
        final_message = "Sorry, I encountered an error while processing your request. Please try again with different inputs."
    
    return JSONResponse(content={"response": final_message})

@app.get("/developers")
async def developers_page(request: Request):
    return templates.TemplateResponse("developers.html", {"request": request})




