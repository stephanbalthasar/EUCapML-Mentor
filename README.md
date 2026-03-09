# Balthasar's Code - Your European Capital Markets Law AI Mentor
An app for students in the Bayreuth University EU Capital Markets Law course
Balthasar's Code is a modular, privacy‑respecting digital tutor for European and German capital markets law.  
It provides two distinct learning modes:

1. **Exam‑style self‑testing** using sample cases and AI‑generated feedback  
2. **RAG‑augmented legal tutoring** using the course booklet + selected EU/DE sources

The system is designed to be lightweight, explainable, and easily maintainable.  
This repository contains a complete redesign of a previous version, based on modular, testable components.

---

## 🧭 Architecture Overview

EUCapML Mentor is built from three conceptual layers:

### **1. Engines (top‑level orchestration)**
These implement the high‑level behaviours:
- **`FeedbackEngine`** → evaluates student answers against model solutions **and** provides:
  - **Plan**: help students plan an answer before writing (outline, topics, anchors)
  - **Evaluate**: assess a finished answer (similarity, coverage, structured feedback)
  - **Explain**: answer follow‑up “why” questions about the feedback
- **`ChatEngine`** → RAG‑augmented legal tutor (booklet + optional web), independent of model answers

Each engine has its own prompts, retrieval strategy, and post‑processing.

### **2. Shared subsystems**
Reusable modules used by both engines:

- **RAG subsystem** (booklet index, retrieval, filtering)
- **LLM subsystem** (model‑agnostic interface)
- **Booklet parsing** (DOCX → semantic chunks)
- **Core utilities** (keyword extraction, similarity, citation helpers)
- **Prompts** (separate prompt logic for both engines)

### **3. UI layer**
A thin Streamlit UI (`streamlit_app.py`) calling into the engines.

### **4. Assets**
Assets (e.g., course booklet, sample cases, model answers) will be sitting in a private repo.

---

## 📁 Repository Structure
```text
eucapml-mentor/
├── .github/workflows/build-index.yml  Workflow to launch indexing booklet (booklet and index sitting in private repo)
└── app/
    ├── __init__.py
    ├── bootstrap_booklet.py
    └── bootstrap_cases.py
└── assets/
    ├── Notice.md            # Privacy and AI Notice
└── mentor/
    ├── booklet/
        ├── parse.py             # I wonder whether this is dead code
        ├── index.py             # I wonder whether this is dead code
        ├── retriever.py         # Query → relevant snippets  
        └── build_booklet_index.py           # Builds index from booklet in private repo and stores json in private repo
    ├── engines/
        ├── feedback_engine.py    Exam-style evaluator: plan/evaluate/explain
        └── chat_engine.py       # RAG-based tutor
    ├── llm/
        ├── __init__.py          
        ├── groq.py            
        └── client.py          # I wonder whether this is dead code           
    ├── rag/
        ├── __init__.py
        └── retrieve.py          Is a placeholder
    ├── __init__.py              
    ├── core.py                  # Shared utilities
    └── prompts.py               # Prompt templates for both engines
└── tests/
    ├── __init__.py
├── README.md
├── requirements.txt
└── streamlit_app.py         # Streamlit UI wrapper
```
---

## ⚙️ Module Responsibilities
### **Modules at a glance**
- `app/bootstrap_booklet.py` — Pulls booklet json from private repo.
- `app/bootstrap_cases.py` — Pulls cases with model answers (json) from private repo.

- `mentor/engines/feedback_engine.py` — Plan / Evaluate / Explain a case.
- `mentor/engines/chat_engine.py` — RAG‑augmented tutor (booklet + optional web).
- `mentor/rag/` — booklet index + filters + snippet retrieval.
- `mentor/llm/` — model‑agnostic client (e.g., Groq).
- `mentor/booklet/` — parse DOCX booklet, extract anchors.
- `mentor/core.py` — shared helpers (keywords, similarity, citations).
- `mentor/prompts.py` — prompts (evaluator, planner, explainer, chat).

### **engines/**
#### `feedback_engine.py`
Implements the exam-style evaluation workflow:
- slice correct model answer  
- extract issues  
- compute similarity & coverage  
- retrieve relevant booklet/web snippets  
- generate structured feedback  
- enforce consistency with model answer  

#### `chat_engine.py`
Implements the RAG‑augmented legal tutor:
- extract legal keywords  
- retrieve relevant booklet snippets  
- (optionally) retrieve selected EU/DE web sources  
- assemble a grounded LLM answer  
- no model‑answer logic, no exam structure  

---

### **llm/**
#### `client.py`
Defines an abstract interface:

python
class LLMClient:
    def chat(self, messages: list[dict], **kwargs) -> str:
        ...

This allows swapping Groq, OpenRouter, or local models without changing engines.
groq.py
Concrete implementation for Groq’s Llama models.

---

### **rag/**
#### `index.py`
Builds and stores the semantic index of booklet chunks:

chunking
embeddings
metadata
caching

#### `filters.py`
Pre‑retrieval keyword and heuristic filtering:

case numbers
paragraph numbers
legal references (MAR, PR, §33 WpHG, etc.)

#### `retrieve.py`
Combines:

semantic similarity
filters
ranking
snippet grouping

Used by both engines with different settings.

---

### **booklet/**
#### `parse.py`
Reads the course booklet DOCX file and outputs:

text chunks
metadata
case/paragraph anchors

#### `anchors.py`
Utility functions:

detect “Case Study 30”
detect “para. 115”
handle citation strings

---

#### `core.py`
Shared utilities:

keyword extraction
safe text normalization
cosine similarity wrappers
citation number detection
truncation helpers

Simple, reusable functions with no Streamlit or LLM dependencies.

---

#### `prompts.py`
Contains two distinct prompt sets:


Feedback Evaluator prompts
Strict, rule-based, structured (five sections, citations, model alignment)


Tutor Chat prompts
Conversational, explanatory, RAG‑grounded, no exam structure


Optionally includes shared guardrails.

---

### **🧪 Testing Strategy (recommended)**
Add tests under tests/:

booklet parsing
RAG retrieval
similarity/coverage metrics
LLM client mock responses


### **🚀 Run the App**
In development:
Shellstreamlit run streamlit_app.py
Secrets needed:
BOOKLET_REPO = ""
BOOKLET_REF = ""
BOOKLET_PATH = ""
CASES_PATH = ""
GITHUB_TOKEN = ""
GROQ_API_KEY = ""
LOG_GIST_TOKEN = ""
GIST_ID = ""
STUDENT_PIN = ""
TUTOR_PIN = ""

### **📜 License**
TBD.

### **👤 Author**
Stephan Balthasar (Allianz SE)
