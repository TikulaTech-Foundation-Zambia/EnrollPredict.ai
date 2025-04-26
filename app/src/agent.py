from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import add_messages
from typing import Annotated, TypedDict, Literal
from langchain_groq import ChatGroq
from langchain_community.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


import pandas as pd

load_dotenv()



class State(TypedDict):
    messages: Annotated[list, add_messages]


class Agent:
    def __init__(self, model:ChatGroq, tools, memory,system=""):
        self.system = system
        self.model = model.bind_tools(tools)
        self.checkpointer = memory
        self.tools_node = ToolNode(tools)

        workflow = StateGraph(State)
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("tools", self.tools_node)

        workflow.add_edge(START, "llm")
        workflow.add_edge("tools", "llm")

        self.graph = workflow.compile(checkpointer=self.checkpointer)



    def call_llm(self, state:State)->Command[Literal["tools", "__end__"]]:
        messages=[SystemMessage(content=self.system)]+state["messages"]
        config = {"configurable": {"thread_id": "1"}}
        response = self.model.invoke(messages, config=config)
       
        if len(response.tool_calls) > 0:
            next_node = "tools"
        else:
            next_node = "__end__"
        
        return Command(goto=next_node, update={"messages":response})


        
            
            
    

    

