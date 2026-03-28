# SecureMemo AI 🔐

**Intelligent Workflow Management with Confidentiality Control**

SecureMemo AI is an agentic system that automatically processes meeting notes, extracts action items, and distributes appropriately filtered information to team members based on their security clearance levels—preventing confidential information leaks while ensuring employees receive the tasks and information they're authorized to see.

---

## 🎯 Problem Statement

Finance professionals managing projects struggle to track action items across scattered meeting notes and documents, often missing critical tasks or accidentally sharing confidential information with team members who lack proper clearance. This leads to missed deadlines, forgotten commitments, and potential compliance violations.

---

## 🤖 Why an Agent?

SecureMemo AI uses agentic AI because it requires sophisticated reasoning that simple automation cannot provide:

- **Context-aware extraction**: Identifying action items, decisions, and assumptions from unstructured meeting notes
- **Intelligent classification**: Determining what information is confidential based on content and context
- **Multi-tier generation**: Creating appropriate versions of the same content for different clearance levels
- **Smart assignment**: Matching tasks to employees based on both their role and clearance level

---

## ✨ Key Features

### Core Capabilities
- 📝 **Meeting Notes Processing**: Automatically extracts action items, decisions, and key information from meeting transcripts
- 🔍 **Employee Lookup**: Quick access to employee information including roles, teams, and contact details
- 🎯 **Task Assignment**: Intelligently matches tasks to qualified employees
- 🔒 **Confidentiality Management**: Generates different versions of content based on clearance levels
- 📧 **Email Distribution**: Sends personalized task summaries to appropriate team members

### Intelligent Analysis
- Extract action items with assigned owners
- Identify assumptions and dependencies
- Classify information by sensitivity level
- Generate role-appropriate summaries

---

## 🏗️ Architecture

SecureMemo AI uses a RAG (Retrieval-Augmented Generation) architecture with three main components:

### 1. **Meeting Notes RAG Tool**
- Processes meeting transcripts using PyPDF
- Chunks and embeds content using Google's Gemini embeddings
- Stores in ChromaDB vector database
- Retrieves relevant context for queries

### 2. **Employee Data RAG Tool**
- Loads employee information from Excel files
- Creates searchable employee profiles
- Enables queries about team composition and roles
- Supports aggregate queries (e.g., "who are the managers?")

### 3. **Generation Pipeline**
- Uses Google's Gemini 1.5 Flash model
- Applies context-aware prompts
- Generates accurate, relevant responses
- Maintains conversation context

---

## 🛠️ Technology Stack

- **Language**: Python 3.12
- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB
- **LLM**: Google Gemini (gemini-flash-latest)
- **Embeddings**: Google Generative AI Embeddings (gemini-embedding-001)
- **Document Processing**: PyPDF, Pandas, openpyxl
- **Development Environment**: Google Colab

---

## 📋 Project Structure

```
SecureMemo AI
├── README.md                        # This file
├── SecureMemo_AI_Coding1.ipynb      # Main notebook with RAG implementation
├── main.py                          
├── app/securememo_ai_coding.py      # Agent logic                     
├── requirements.txt                 # All dependencies listed
├── .gitignore                       
└── data/
    ├── Project_Descriptions1.pdf    # Company project information
    ├── Meeting_Notes1.pdf           # Sample meeting notes
    ├── Meeting_Notes2.pdf           # Sample meeting notes
    └── Sample_Employee_Data.xlsx    # Employee clearance data
```

---

## 🚀 Getting Started

### Prerequisites
- Google Colab account (or local Python 3.12+ environment)
- Google API key for Gemini access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/reeyasingh021/Secure-Memo-AI-BreakThroughTech.git
   cd Secure-Memo-AI-BreakThroughTech
   ```

2. **Open in Google Colab**
   - Upload `SecureMemo_AI_Coding.ipynb` to Google Colab
   - Or run locally with Jupyter Notebook

3. **Set up API Key**
   ```python
   from google.colab import userdata
   import os
   os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
   ```

4. **Upload Data Files**
   - Upload the PDF and Excel files to your Colab session
   - Or place them in the appropriate directory for local development

5. **Run the Notebook**
   - Execute cells sequentially
   - The notebook will guide you through:
     - Setting up embeddings and vector stores
     - Creating RAG pipelines
     - Testing with sample queries

### Sample Usage

```python
# Query meeting notes
response = rag_chain_mn.invoke("What tasks are related to the Small Business Loan Support Program?")
print(response)

# Query employee data
response = rag_chain_ed.invoke("Who are the project managers?")
print(response)
```

---

## 🎯 Use Cases

### Current Implementation
- Extract action items from financial services meeting notes
- Identify employees by role, team, or clearance level
- Answer questions about project tasks and responsibilities

### Future Capabilities (Roadmap)
- **Multi-tier document generation**: Automatically create confidential, internal, and public versions
- **Email automation**: Send personalized task lists to employees
- **Calendar integration**: Add tasks and deadlines to employee calendars
- **Follow-up reminders**: Track task completion and send reminders
- **Risk detection**: Identify blocked tasks and timeline risks


---

## 👥 Team

**Team Members**
- **Nivi Munjal** - Research and tool implementation, documentation creation
- **Reeya Singh** - Task organization and project timeline management
- **Naman Bagga** - AI agent architecture and system integration

**Mentor**
- **Aram Ramos** - Project guidance and technical mentorship

**Program**
- **Break Through Tech AI** - Agentic AI Program (Fall 2026)
  - Affinity Category: Productivity & Workplace

---

## 🙏 Acknowledgments

This project was developed as part of the **Break Through Tech AI Agentic AI Specialization Program** (Winter 2026). We are grateful to:

- **Aram Ramos**, our mentor, for invaluable guidance and support throughout the project
- **Break Through Tech AI**, for providing the framework, resources, and opportunity to build this project
- **Anthropic** and **Google**, whose AI technologies power SecureMemo AI

---

## 📝 License

This project is created for educational purposes as part of the Break Through Tech AI program.

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/reeyasingh021/Secure-Memo-AI-BreakThroughTech](https://github.com/reeyasingh021/Secure-Memo-AI-BreakThroughTech)

---

## 📧 Contact

For questions or collaboration opportunities, please reach out to the team through our GitHub repository.

---

**Built with ❤️ by the SecureMemo AI Team**

*Empowering finance professionals with intelligent workflow management and confidentiality protection.*
