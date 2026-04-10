# SecureMemo AI Agentic Workflow

#------------------------
#Imports
import os
# PyPDF and Text Splitting/Chunking Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and Storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Generation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Tool Decorator
from langchain_core.tools import tool

# Create Agent
from langchain.agents import create_agent

from rank_bm25 import BM25Okapi
import re
from dotenv import load_dotenv

load_dotenv()

#------------------------
# Configure the API key as an environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGSMITH_PROJECT"] = "SecureMemo_AI_SubAgent_Tests"

#------------------------
# Shared Resources

# Helper function to load content of file
def load_pdf(file_name):
   """
   Performs:
   1. loading of data from pdf file using pyPDFLoader
   2. converts all pages into one string

   input: file_name
   output: full_text string
   """
   # loading
   loader = PyPDFLoader(file_name)
   document = loader.load()

   # combining pages into one string
   full_text = "\n\n".join([page.page_content for page in document])
   return full_text

# Helper function to create chunks
def get_chunks(full_text, chunk_size, chunk_overlap):
   """
   Creates chunks and returns them
   Input:
   1. text string
   2. chunk_size for recursive splitting
   3. chunk_overlap for recursive splitting
   """
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
   return text_splitter.split_text(full_text)

# BM25 precomputation for Keyword Search
def build_bm25_index(chunks):
   # Simple Tokenization/Preprocessing (Low-level cleaning)
   def tokenize(text):
      return re.sub(r'[^\w\s]', '', text.lower()).split()
   tokenized = [tokenize(c) for c in chunks]
   return BM25Okapi(tokenized), tokenize

# Keyword Search Function
def keyword_search(query, chunks, bm25, tokenizer, top_k=3):
    """
    Performs BM25 keyword search over a list of text chunks.

    Args:
        query (str): The user's search query.
        chunks (list): List of strings (the text chunks).
        bm25 (object): To run the keyword search.
        tokenizer (function): To tokenize the user query.
        top_k (int): Number of results to return.

    Returns:
        list: The top_k most relevant chunks.
    """
    tokenized_query = tokenizer(query)
    return bm25.get_top_n(tokenized_query, chunks, n=top_k)

# All the RAG tools will use the same embedding version to convert text into embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

#------------------------
# RAG Tool For Project Description

# Company Project Notes - Document Loading and Chunking
full_text_pd1 = load_pdf("Project_Descriptions1.pdf")
chunks_pd1 = get_chunks(full_text_pd1, 2000, 200)
bm25_pd1, tokenize_pd1 = build_bm25_index(chunks_pd1)

# Embeddings and Storage
# Use Chroma.from_texts() with chunks and embedding model
vectorstore_pd1 = Chroma.from_texts(
    texts=chunks_pd1,
    embedding=embeddings,
    collection_name="projects_collection"
)

# Generation

llm_pd1 = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template_pd1 = """You are a helpful project description notes agent. Answer the question in a concise and straightforward way. 
Provide more details about the projects when asked for them. Only use information from this context. 
If query is out of context, respond that you cannot provide the response.

Context:
{context}

Question: {question}

Answer:"""

prompt_pd1 = ChatPromptTemplate.from_template(template_pd1)

# Defining chain beforehand
chain_pd1 = prompt_pd1 | llm_pd1 | StrOutputParser()

@tool
def process_project_description(query: str) -> str:
  """Search and retrieve specific information about company projects,
    including project titles, summaries, Employee Involvement and Access Levels."""
  # Running semantic search
  vector_docs = vectorstore_pd1.similarity_search(query, k=5)
  vector_texts = [doc.page_content for doc in vector_docs]

  # Running Keyword search
  keyword_texts = keyword_search(query, chunks_pd1, bm25_pd1, tokenize_pd1)
  
  # Merging results from Keyword and Semantic search
  seen = set()
  merged = []
  for t in (vector_texts + keyword_texts):
    if t not in seen:
      seen.add(t)
      merged.append(t)
  context_string = "\n\n---\n\n".join(merged[:8])

  return chain_pd1.invoke({"context": context_string, "question": query})

agent_llm_pd1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

system_prompt_pd1 = """You are a helpful project description notes assistant.
Your goal is to help users understand company projects using the 'get_project_details' tool.
Always verify project facts using the tool before answering.
If a user asks for sensitive financial info which user is not authorized to access or things not in the documents,
politely decline as per company policy."""

agent_pd1 = create_agent(
    model=agent_llm_pd1,
    tools=[process_project_description],
    system_prompt=system_prompt_pd1
)

#------------------------
#RAG Tool for Meeting Notes

# First Meeting Notes to be stored in meeting notes storage

full_text_mn1 = load_pdf("Meeting_Notes1.pdf")
chunks_mn1 = get_chunks(full_text_mn1, 200, 50)

# Second Meeting Notes
full_text_mn2 = load_pdf("Meeting_Notes2.pdf")
chunks_mn2 = get_chunks(full_text_mn2, 200, 50)

# Embeddings and Storage
meeting_notes_chunks = chunks_mn1 + chunks_mn2

bm25_mn, tokenize_mn = build_bm25_index(meeting_notes_chunks)

# Use Chroma.from_texts() with chunks and embedding model
vectorstore_mn = Chroma.from_texts(
    texts=meeting_notes_chunks,
    embedding=embeddings,
    collection_name="meeting_notes_collection"
)

# Generation

# Initialize the LLM (matching the exact format from Project Description)
llm_mn = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template_mn = """You are a helpful assistant analyzing meeting notes from a company.

Use the following context from meeting notes to answer the question. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

prompt_mn = ChatPromptTemplate.from_template(template_mn)

chain_mn = prompt_mn | llm_mn | StrOutputParser()

@tool
def process_meeting_notes(query: str) -> str:
  """Process and Search the uploaded Meeting Notes for tasks mentioned in company meetings. Tasks can include information about action items for projects, types of employees and access levels required, and deadines.
    Use this tool when asked about tasks mentioned in meeting notes."""
  # Running semantic search
  vector_docs = vectorstore_mn.similarity_search(query, k=5)
  vector_texts = [doc.page_content for doc in vector_docs]

  # Running Keyword search
  keyword_texts = keyword_search(query, meeting_notes_chunks, bm25_mn, tokenize_mn)
  
  # Merging results from Keyword and Semantic search
  seen = set()
  merged = []
  for t in (vector_texts + keyword_texts):
    if t not in seen:
      seen.add(t)
      merged.append(t)
  context_string = "\n\n---\n\n".join(merged[:8])

  return chain_mn.invoke({"context": context_string, "question": query})

# Meeting Notes Agent with RAG Tool

agent_llm_mn = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

system_prompt_mn = """You are a helpful meeting notes assistant for processing and analyzing company tasks in the provided meeting notes.

You have one tool:
1. process_meeting_notes: Use this tool for any questions about the tasks in the meeting notes.
These questions can include, but are not limited to, knowing the tasks for a specific project, finding
types of tasks that require a certain type of employee position and/or employee access level, or knowing the deadlines for tasks.

Only use this tool for answering questions about what is in the provided meeting notes."""

agent_mn = create_agent(
    model=agent_llm_mn,
    tools=[process_meeting_notes],
    system_prompt=system_prompt_mn
)

#------------------------
# RAG Tool for Employee Data

# Employee Records - Document Loading and Chunking
import pandas as pd

# Load the Excel file
employee_df = pd.read_excel("Sample_Employee_Data.xlsx")

# Convert the DataFrame to text format for RAG
# Each employee becomes a document chunk
employee_texts = []
for _, row in employee_df.iterrows():
    employee_text = f"""Employee: {row['Name']}
Email: {row['Email']}
Employee ID: {row['Employee ID']}
Team: {row['Team']}
Position: {row['Position']}"""
    employee_texts.append(employee_text)

# Create additional contextual chunks about teams and positions
# This helps with queries like "who are the engineers?" or "who are the managers?"
team_summary = employee_df.groupby('Team').size().to_dict()
position_summary = employee_df.groupby('Position').size().to_dict()

team_text = "Team Summary:\n" + "\n".join([f"- {team}: {count} employees" for team, count in team_summary.items()])
position_text = "Position Summary:\n" + "\n".join([f"- {position}: {count} employees" for position, count in position_summary.items()])

# Add summary texts to help with aggregate queries
employee_texts.append(team_text)
employee_texts.append(position_text)

bm25_ed, tokenize_ed = build_bm25_index(employee_texts)

# Embeddings and Storage
# Create vector store for employee data

# Create the vector store
vectorstore_ed = Chroma.from_texts(
    texts=employee_texts,
    embedding=embeddings,
    collection_name="employee_data_collection"
)

# Generation

# Initialize LLM for employee queries
llm_ed = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template_ed = """You are a helpful HR assistant with access to company employee records.

Use the following employee information to answer the question. Be concise and accurate.
If the question asks about multiple people (e.g., "who are the managers?"), list all relevant employees.
If you cannot find the information, say so clearly.

Employee Information:
{context}

Question: {question}

Answer:"""

prompt_ed = ChatPromptTemplate.from_template(template_ed)

chain_ed = prompt_ed | llm_ed | StrOutputParser()

@tool
def process_employee_data(query: str) -> str:
  """Process and Search the uploaded Employee Data for information about employees, teams, positions, and clearance levels.
    Use this tool when asked about employees, their roles, team composition, or access levels."""
  # Running semantic search
  vector_docs = vectorstore_ed.similarity_search(query, k=5)
  vector_texts = [doc.page_content for doc in vector_docs]

  # Running Keyword search
  keyword_texts = keyword_search(query, employee_texts, bm25_ed, tokenize_ed)

  # Merging duplicate chunks
  seen = set()
  merged = []
  for t in (vector_texts + keyword_texts):
    if t not in seen:
      seen.add(t)
      merged.append(t)

  context_string = "\n\n---\n\n".join(merged[:8])
  return chain_ed.invoke({"context": context_string, "question": query})

# Employee Data Agent with RAG Tool

agent_llm_ed = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

system_prompt_ed = """You are a helpful HR assistant for N 'nd R Associates with access to employee records.

You have one tool:
1. process_employee_data: Use this tool for any questions about employees, teams, positions, or organizational structure.
These questions can include, but are not limited to, finding specific employees by name or ID, finding employees by team or position,
determining who has certain clearance levels, or understanding the organizational structure.

Only use this tool for answering questions about what is in the employee database."""

agent_ed = create_agent(
    model=agent_llm_ed,
    tools=[process_employee_data],
    system_prompt=system_prompt_ed
)

#------------------------

"""#Main Orchestration Agent

Create Agent with the three RAG Agents as Sub-Agents
This approach references: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
- Create Subagents (Done in above code)
- Create tool functions to define when the main agent should call/invoke the other agents
- Use the main agent to answer a query like "Please assign employees to the tasks outlined in the meeting notes. Reference our project description document for finding the best matches.".
"""

# Tool for project description agent
@tool
def project_description_agent(request:str) -> str:
  """ Call the agent for processing the project description.

  Use this agent when the main agent wants to gather information about the projects in the company, the details of each project,
  and the employee positions involved in each project.

  Input: Natural language project information gathering request (e.g., 'Get the project title, description, and employee positions for each project.')

  """
  result = agent_pd1.invoke({
        "messages": [{"role": "user", "content": request}]
    })

  return result["messages"][-1].content

# Tool for meeting notes agent
@tool
def meeting_notes_agent(request:str) -> str:
  """ Call the agent for processing the meeting notes.

  Use this agent when the main agent wants to get the tasks in the meeting notes.
  Input: Natural language tasks request (e.g., 'Get the tasks mentioned in the meeting notes.')

  """
  result = agent_mn.invoke({
        "messages": [{"role": "user", "content": request}]
    })

  return result["messages"][-1].content

# Tool for employee data agent
@tool
def employee_data_agent(request:str) -> str:
  """ Call the agent for processing the employee data.

  Use this agent when the main agent wants to find an employee for a task based on employee position.
  Input: Natural language employee request (e.g., 'Find the name and email of an employee that is a Finance Intern with low-level access.')

  """
  result = agent_ed.invoke({
        "messages": [{"role": "user", "content": request}]
    })

  return result["messages"][-1].content

# Create the main agent
agent_llm_main = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

system_prompt_agent_main = """
You are a helpful main assistant for N nd' R Associates.

Your primary responsibility is to assign employees to company tasks outlined in meeting notes using information from three sources:
1. Project descriptions
2. Meeting notes
3. Employee data

You have access to three tools:
- project_description_agent: Use this FIRST to understand projects and required roles
- meeting_notes_agent: Use this SECOND to extract ALL tasks from meeting notes
- employee_data_agent: Use this THIRD to find employees that match the required roles

You MUST use all three tools in this order.

---

### CRITICAL WORKFLOW (DO NOT SKIP)

You MUST follow this exact reasoning process:

STEP 1:
Extract a COMPLETE list of ALL projects from the meeting notes.

You MUST identify multiple distinct projects if they exist.
Do NOT stop after finding one project.
If only one project is returned but multiple exist in the meeting notes,
you MUST continue searching and include the missing projects.

STEP 2:
For EACH project, extract ALL tasks associated with it.
- Do NOT summarize
- Do NOT skip any tasks
- Capture EVERY task explicitly

STEP 3:
For EACH task, determine the required role using project descriptions.

STEP 4:
For EACH task, find ALL relevant employees using employee data.
- Do NOT limit to one employee
- Include ALL reasonable matches

STEP 5:
Repeat until EVERY task across ALL projects is processed.

---

### COMPLETENESS REQUIREMENTS (STRICT)

Your response is NOT complete unless:

- ALL projects from the meeting notes are included
- ALL tasks under each project are included
- EVERY task has either:
  - one or more matched employees, OR
  - is explicitly marked as "No match found"

If anything is missing, you MUST continue working before responding.

---

### OBJECTIVE

Match employees to tasks based on:
- Required role or position
- Relevance to project context
- Available employee data

IMPORTANT:
- Do NOT generate fake or example employees
- ONLY use real employees returned by the employee_data_agent
- If no employee exists, say "No match found"

---

### OUTPUT FORMAT (VERY IMPORTANT)

You MUST return your final answer in TWO parts:

---

### PART 1: NATURAL LANGUAGE SUMMARY

Provide a short, clear explanation of the assignments.

Example:
"Here are the recommended employee assignments based on the meeting notes and project requirements."

---

### PART 2: STRUCTURED JSON ARRAY

Immediately after the explanation, output a JSON array.

DO NOT include labels like "JSON:" or any extra text before the array.

---

### JSON FORMAT

Each task MUST be represented as ONE object with MULTIPLE employees:

[
  {
    "sector": "Sector Name",
    "project": "Project Name",
    "task": "Task description",
    "employees": [
      {
        "name": "Employee Name",
        "position": "Employee Position",
        "email": "employee@email.com"
      },
      {
        "name": "Employee Name",
        "position": "Employee Position",
        "email": "employee@email.com"
      }
    ]
  }
]

---

### JSON RULES

- JSON must be VALID
- Use double quotes for all keys and values
- Include ALL tasks
- Each task = one object
- Each task can have MULTIPLE employees
- Do NOT include comments
- Do NOT include explanation inside JSON
- If no employees match:
  - set employees = []

---

### FINAL RESPONSE STRUCTURE

Your response MUST look like this:

Here are the recommended employee assignments based on the meeting notes and project requirements.

[
  { ... },
  { ... }
]

---

Failure to follow this format will break the system.
"""

agent_main = create_agent(
    model=agent_llm_main,
    tools=[project_description_agent, meeting_notes_agent, employee_data_agent],
    system_prompt=system_prompt_agent_main
)