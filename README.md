# EUCapML-Mentor
An app for students in the Bayreuth University EU Capital Markets Law course
EUCapML Mentor is a modular, privacy‑respecting digital tutor for European and German capital markets law.  
It provides two distinct learning modes:

1. **Exam‑style self‑testing** using sample cases and AI‑generated feedback  
2. **RAG‑augmented legal tutoring** using the course booklet + selected EU/DE sources

The system is designed to be lightweight, explainable, and easily maintainable.  
This repository contains **v2**, a complete redesign based on modular, testable components.

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
├── streamlit_app.py         # Streamlit UI wrapper
├── README.md
├── requirements.txt
└── app/
    ├── __init__.py
    ├── bootstrap_booklet.py
    └── bootstrap_cases.py
└── mentor/
    ├── engines/
    │   ├── feedback_engine.py    Exam-style evaluator: plan/evaluate/explain
    │   └── chat_engine.py       # RAG-based tutor
    │
    ├── rag/
    │   ├── __init__.py
    │   ├── index.py             # Build and store booklet embeddings
    │   └── filters.py           # Keyword & heuristic narrowing
    │
    ├── llm/
    │   ├── client.py            # Model-agnostic LLM interface
    │   └── groq.py              # Groq implementation
    │
    ├── booklet/
    │   ├── parse.py             # Parse DOCX booklet into chunks
    │   ├── retrieve.py          # Query → relevant snippets  
    │   └── anchors.py           # Detect paras, cases, sections
    │
    ├── core.py                  # Shared utilities
    └── prompts.py               # Prompt templates for both engines
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

### Improvement proposals by M365 Copilot (4 Mar 2026)
1) Make the evaluator deterministic, grounded, and verifiable
Right now the “Evaluate” path returns free‑form markdown from the LLM. Tighten this into a structured, grounded pipeline so students get consistent results and you can audit outcomes.
What to do

Structured JSON output: Have FeedbackEngine.evaluate_answer() ask the model for a JSON object (overall_assessment, rubric_scores, missing_issues, citations, action_items). Parse/validate it before rendering markdown. This prevents “wandering” feedback and makes it testable. [allianzms-...epoint.com]
Rubric with weights: Encode your exam criteria (issue spotting, legal basis, application, structure, conclusions) with weights; compute a composite score locally and render a clean panel from the JSON. [allianzms-...epoint.com]
RAG grounding for Evaluate: You already have a booklet index and retrievers; use them to anchor Evaluate with 3–6 snippets per question, not just the model answer slice. Show inline citations (“MAR Art. 7(1) / Booklet §2.3”) and links to the trusted EU/DE sources you listed for the tutor engine (EUR‑Lex, CURIA, BaFin, etc.). It aligns with your architecture where the RAG subsystem is shared between engines. [allianzms-...epoint.com]
Guardrails: If JSON parsing fails, retry once with a “you produced invalid JSON—fix it” system message; if still invalid, fall back to a clearly labeled “Unstructured feedback” box.

Why it matters

Students get consistent, comparable feedback.
You gain explainability, which is critical in legal education.
You can add unit tests for the evaluator without mocking markdown. [allianzms-...epoint.com]


2) Persistence, IDs, and audit trails (per case/question attempt)
You’ve got session‑based state for answers, feedback, and chat; add lightweight persistence so a student can resume later, and you get a log book for course analytics—without storing personal data.
What to do

Attempt IDs: On each evaluation, generate a run_id like caseId:question:timestamp. Store {run_id, case_id, q_index, answer, feedback, chat_history, model, temperature} in a tiny local store (JSON or SQLite). Add “Continue last attempt” and “Start new attempt” buttons on the Evaluate screen. [allianzms-...epoint.com]
PIN or pseudonymous tag: If you later re‑enable the PIN concept, namespace attempts by PIN without collecting personal data. (You previously wanted a log book of submissions; this completes that story.)
Export with metadata: Your .docx export works—great. Move the make_docx() helper out of the UI into a small utils/exports.py, and include metadata (case title, question, run_id, model, temperature, timestamp). Offer PDF as an opt‑in (use reportlab) once stable. [allianzms-...epoint.com]
Recovery UX: On app reload, detect unfinished attempts in st.session_state / local store and offer to restore the last run’s answer/feedback/chat into the screen automatically. [allianzms-...epoint.com]

Why it matters

Students don’t lose work; you get traceability.
Clean separation of concerns: UI ↔ engines ↔ persistence.
Sets you up for anonymized course analytics (e.g., common missing issues per question).


3) Polish the student UX for long-form answers
You already switched to a single‑column flow—nice. Now make the writing and review experience excellent for exam‑length answers.
What to do

Better editor ergonomics: add word/character counts, a “time spent” indicator, and optional autosave every 10–15 seconds (just to session/local store). Add keyboard shortcuts (Ctrl/Cmd+Enter to Evaluate; Enter to send in chat). [allianzms-...epoint.com]
Stable layout: Keep the case description in an expander (“Show/Hide case”), and render the submitted answer & feedback in collapsible sections with sticky subheadings. This avoids vertical jumps on re‑runs. [allianzms-...epoint.com]
Streaming replies (optional): Stream the LLM output in both Evaluate (for the long feedback block) and follow‑up chat, so students see progress on slower models. (Your Groq client likely supports streaming; expose it through the LLMClient.) [allianzms-...epoint.com]
Consistent state keys: You already tripped over mixed keys earlier. Normalize to a single helper like _key(case_id, q_index) and reuse it for answer, feedback, chat, and downloads. This prevents “missing output after re‑run” surprises. [allianzms-...epoint.com]
Error UX: Replace raw exceptions with friendly toasts (e.g., “Model is busy, retrying…”) and auto‑retry once on transient errors; log details server‑side.

Why it matters

Students focus on legal reasoning, not UI friction.
Fewer support pings from “my text disappeared” or “nothing happens when I click.”


Bonus quick wins (low effort, high payoff)

Centralize constants (default temperature, max_tokens, model names) in one settings.py. You already pass these around in multiple places. [allianzms-...epoint.com]
Module hygiene: Move inline helpers like make_docx() into a utilities module; keep the Streamlit file a thin orchestrator. [allianzms-...epoint.com]
Tests: Add a tiny tests/ suite for: model‑slice selection by q_index, JSON schema validation of evaluator output, and RAG retrieval sanity checks. Your README already outlines a testing strategy—use it. [allianzms-...epoint.com]

End of README.md
