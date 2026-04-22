import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.agent import agent_main
from fastapi.responses import RedirectResponse

load_dotenv()

REQUIRED_FILES = [
    "Project_Descriptions1.pdf", 
    "Meeting_Notes1.pdf", 
    "Meeting_Notes2.pdf", 
    "Sample_Employee_Data.xlsx"
]

app = FastAPI(
    title="SecureMemo AI",
    description="""An agent that interacts with multiple 
                    sub agents and can read the company
                    projects, meeting notes, and employee
                    data to answer questions and assign the
                    tasks powered by LangGraph""",
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

# Updated response model (IMPORTANT)
class ChatResponse(BaseModel):
    response: str
    assignments: list = []

@app.on_event("startup")
async def verify_files():
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            print(f"CRITICAL ERROR: {file} is missing! Agent will fail.")
        else:
            print(f"SUCCESS: {file} detected.")

@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": "default_user"}}

        result = agent_main.invoke(
            {"messages": [HumanMessage(content=request.message)]}, 
            config=config
        )

        full_text = result["messages"][-1].content

        # Extract JSON block
        json_match = re.search(r'\[\s*{.*}\s*\]', full_text, re.DOTALL)
        
        # Remove JSON from visible text
        clean_text = re.sub(r'\[\s*{.*}\s*\]', '', full_text, flags=re.DOTALL).strip()
        
        assignments = []
        if json_match:
            try:
                assignments = json.loads(json_match.group())
            except Exception as e:
                print("JSON parsing failed:", e)
                assignments = []
                
        return ChatResponse(
            response=clean_text,
            assignments=assignments
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
