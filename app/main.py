import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.agent import agent_main

load_dotenv()

REQUIRED_FILES = [
    "Project_Descriptions1.pdf", 
    "Meeting_Notes1.pdf", 
    "Meeting_Notes2.pdf", 
    "Sample_Employee_Data.xlsx"
]

app = FastAPI(
    title="AlgoRhythm Agent API",
    description="""An agent that interacts with multiple 
                    sub agents and can read the company
                    projects, meeting notes, and employee
                    data to answer questions and assign the
                    taskspowered  by LangGraph""",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def verify_files():
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            print(f"CRITICAL ERROR: {file} is missing! Agent will fail.")
        else:
            print(f"SUCCESS: {file} detected.")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # config is important because it allows agent to remember the history
        config = {"configurable": {"thread_id": "default_user"}}
        result = agent_main.invoke(
            {"messages": [HumanMessage(content=request.message)]}, 
            config=config
        )
        return ChatResponse(response=result["messages"][-1].content)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
