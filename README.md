# Aura — Multi-Agent AI Assistant

Aura is a Streamlit-based AI assistant that automatically routes each user query to the right capability:

- **General chat** (Groq Llama)
- **Document Q&A (RAG)** over uploaded PDFs
- **Web search** with Google Search grounding
- **Image analysis / OCR** with Gemini Vision

No manual mode switching is required.

---

## Table of Contents

- [1) Project Demo (What it does)](#1-project-demo-what-it-does)
- [2) Core Features](#2-core-features)
- [3) Architecture Overview](#3-architecture-overview)
- [4) Repository Structure](#4-repository-structure)
- [5) Prerequisites](#5-prerequisites)
- [6) Installation (Windows / macOS / Linux)](#6-installation-windows--macos--linux)
- [7) Environment Configuration](#7-environment-configuration)
- [8) Run the App](#8-run-the-app)
- [9) How to Use (Step-by-step)](#9-how-to-use-step-by-step)
- [10) Optional K-12 Standards Script](#10-optional-k-12-standards-script)
- [11) Troubleshooting](#11-troubleshooting)
- [12) Security Notes](#12-security-notes)

---

## 1) Project Demo (What it does)

### End-to-end flow

```text
User query
   │
   ▼
Router (Groq llama-3.1-8b-instant)
   │
   ├── GENERAL     → Chat response
   ├── RAG         → Retrieve from PDF index + answer with sources
   ├── WEB_SEARCH  → Gemini + Google Search grounded answer
   └── OCR         → Analyze uploaded image + answer
```

### Real usage scenarios

1. **Ask a coding question**  
   Example: `Explain Python lists with a class-6 level example.`  
   Aura routes to **GENERAL**.

2. **Ask a current-events question**  
   Example: `What happened in AI news this week?`  
   Aura routes to **WEB_SEARCH** and includes source links.

3. **Upload a PDF and ask about it**  
   Example: upload notes, then ask: `Summarize chapter 2.`  
   Aura routes to **RAG** and returns contextual answer + document sources.

4. **Upload an image and ask about it**  
   Example: upload screenshot/photo, then ask: `What text do you see?`  
   Aura routes to **OCR**.

---

## 2) Core Features

- **Automatic intent routing** via lightweight Groq router model.
- **Streaming responses** in the chat UI.
- **RAG with FAISS + HuggingFace embeddings** for local PDF knowledge.
- **Web grounding** with Gemini search tools.
- **Vision/OCR pipeline** for image understanding.
- **LangSmith tracing support** for end-to-end chain/graph observability.
- **Session history** in sidebar with quick restore.
- **Source display**:
  - inline icon links for URL/web sources
  - chips for non-URL document sources

---

## 3) Architecture Overview

### Main components

- `app.py`: Streamlit app entrypoint and orchestration glue.
- `graph.py`: LangGraph nodes + routing logic.
- `router.py`: intent classifier.
- `chat_chain.py`: general conversational response chain.
- `rag_chain.py`: retrieval + grounded generation.
- `search_agent.py`: Gemini search-grounded responses.
- `vision_service.py`: image/OCR analysis.
- `ingestion.py`: PDF parsing/chunking/indexing.
- `sidebar.py`, `ui_helpers.py`, `styles.css`: UI and rendering utilities.

### Models and engines used

- **Groq**
  - `llama-3.3-70b-versatile` (general + RAG generation)
  - `llama-3.1-8b-instant` (intent router)
- **Google Gemini**
  - `gemini-3.1-flash-lite-preview` (primary web search model)
  - `gemini-2.5-flash-lite` (fallback for quota issues)
  - `gemini-2.5-flash` / fallback lite (vision paths)
- **Embeddings**: `all-MiniLM-L6-v2`
- **Vector store**: FAISS
- **Observability**: LangSmith tracing (optional via environment variables)

---

## 4) Repository Structure

```text
Aura/
├── app.py
├── graph.py
├── router.py
├── chat_chain.py
├── rag_chain.py
├── search_agent.py
├── vision_service.py
├── ingestion.py
├── ingest_k12_standards.py
├── config.py
├── sidebar.py
├── ui_helpers.py
├── styles.css
├── Aura_Icon.png
├── requirements.txt
└── README.md
```

---

## 5) Prerequisites

- Python **3.10+** (recommended: 3.10 or 3.11)
- A **Groq API key**
- A **Google AI API key**
- Git

Recommended system resources:

- 8 GB RAM minimum
- Stable internet connection for model APIs

---

## 6) Installation (Windows / macOS / Linux)

## Step A — Clone from GitHub

```bash
git clone <YOUR_REPO_URL>
cd Aura
```

## Step B — Create virtual environment and install dependencies

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Command Prompt / CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7) Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_ai_key_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=Aura
```

If tracing is enabled, also set `LANGCHAIN_API_KEY`.

### LangSmith Traceability

Aura can be traced in LangSmith for debugging and observability.

Enable these in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=Aura
```

What you can observe in LangSmith:

- Router decisions and fallback behavior
- Chain/LLM latency and token usage
- Prompt/response spans for debugging
- End-to-end execution paths across the app

---

## 8) Run the App

From project root (with virtual environment activated):

```bash
streamlit run app.py
```

Open the local URL shown in terminal (typically `http://localhost:8501`).

---

## 9) How to Use (Step-by-step)

### A) General chat

- Ask any non-document, non-image, non-live query.
- Example: `Teach me loops like I am in class 5.`

### B) Web search mode

- Ask real-time or recent-info questions.
- Example: `Latest Python release highlights?`
- Aura auto-routes to web search and returns grounded output with sources.

### C) Document Q&A (RAG)

- Use chat input attachment (`+`) to add one or more PDFs.
- Wait until indexing finishes.
- Ask document-specific questions.
- Example: `Summarize the key rules from section 3.`

### D) Image analysis / OCR

- Use chat input attachment (`+`) to upload image.
- Ask questions about text/content in that image.
- Example: `Extract all text from this image.`

### E) Chat sessions

- Use **+ New chat** in sidebar to start a fresh conversation.
- Prior sessions appear in **Chat history** and can be restored.

---

## 10) Optional K-12 Standards Script

The repository includes `ingest_k12_standards.py` for building/querying a grade-aware FAISS index outside the main UI flow.

### Ingest

```bash
python ingest_k12_standards.py ingest \
  --codeorg ./data/codeorg_standards.pdf \
  --ncert12 ./data/ncert_class12_cs.pdf \
  --index-dir ./faiss_k12
```

### Query (explicit level)

```bash
python ingest_k12_standards.py query \
  --index-dir ./faiss_k12 \
  --question "Explain loops with an example" \
  --difficulty "Middle"
```

---

## 11) Troubleshooting

### 1) `python` command not found

- Windows: install Python and check **Add Python to PATH**.
- macOS/Linux: try `python3` instead of `python`.

### 2) PowerShell blocks activation

Run once in PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then reactivate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) `ModuleNotFoundError`

- Ensure virtual environment is active.
- Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### 4) API errors / empty responses

- Verify `.env` keys are valid and non-empty.
- Restart terminal and rerun app after editing `.env`.
- Check Groq/Google quota limits.

### 5) Streamlit port already in use

```bash
streamlit run app.py --server.port 8502
```

---

## 12) Security Notes

- Never commit `.env` or API keys to GitHub.
- Keep `.gitignore` enabled for local and secret files.
- Rotate keys immediately if exposed.

---

If you are using this project from GitHub and want to customize models/chunking, update values in `config.py` and restart Streamlit.
