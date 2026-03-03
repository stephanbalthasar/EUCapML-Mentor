# streamlit_app.py
# Minimal UI to exercise both engines using the booklet index from the private repo.

import streamlit as st

# --- Load booklet index (server-side; users never see this file) ---
from app.bootstrap_booklet import load_booklet_index
INDEX = load_booklet_index()  # {"paragraphs": [...], "chapters": [...]}

from app.bootstrap_cases import load_cases
CASES = load_cases()

from mentor.booklet.retriever import ParagraphRetriever, ChapterRetriever
from mentor.engines.chat_engine import ChatEngine
from mentor.engines.feedback_engine import FeedbackEngine
from mentor.llm.groq import GroqClient

st.set_page_config(page_title="EUCapML Mentor", page_icon="⚖️", layout="wide")

# Optional tiny banner (remove once you’re confident)
st.caption(f"📖 Booklet loaded — {len(INDEX['chapters'])} chapters, {len(INDEX['paragraphs'])} numbered paragraphs.")

# --- Build retrievers once ---
para_retriever = ParagraphRetriever(INDEX["paragraphs"])
chap_retriever = ChapterRetriever(INDEX["chapters"])

# --- LLM client ---
llm_api_key = st.secrets.get("GROQ_API_KEY")
if not llm_api_key:
    st.error("Missing GROQ_API_KEY in secrets.")
    st.stop()
llm = GroqClient(api_key=llm_api_key)

# --- Engines ---
chat_engine = ChatEngine(
    llm=llm,
    booklet_index=INDEX,
    booklet_retriever=chap_retriever,  # ChatEngine uses chapter-level grounding
    web_retriever=None                 # add later if you want web RAG
)
feedback_engine = FeedbackEngine(llm=llm)

# --- Sidebar controls ---
with st.sidebar:
    model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    temp  = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.button("Reload booklet index (server cache)"):
        st.cache_data.clear()
        st.success("Re-loaded. Re-run the action to use the latest JSON.")

# --- Tabs: Feedback + Tutor chat ---
tab_feedback, tab_chat = st.tabs(["📝 Feedback", "💬 Tutor chat"])

# Small helper: persist latest run per case+question
def _key(case_id: str, q_label: str) -> str:
    return f"{case_id}::{q_label}"

with tab_feedback:
    st.subheader("Exam Feedback")

    # --- Left and Right columns for feedback workflow ---
    left, right = st.columns([1, 1], gap="large")

    # --- (A) LEFT: inputs ---
    with left:
        # Use real cases from CASES
        case_titles = [c.get("title", c.get("id", "Untitled case")) for c in CASES]
        sel_case_title = st.selectbox("Select exam case", case_titles, index=0)
        sel_case = next(c for c in CASES if c.get("title", c.get("id")) == sel_case_title)
        sel_case_id = sel_case.get("id", "unknown")
        
        # Show the full case description (which already contains the numbered questions)
        st.markdown("**Case description**")
        st.write(sel_case.get("description", "—"))
        
        # Let the user pick which question number they are answering
        q_count = int(sel_case.get("question_count", 1))
        q_labels = [f"Question {i+1}" for i in range(max(1, q_count))]
        q_label  = st.selectbox("Which question are you answering?", q_labels, index=0)
        q_index  = q_labels.index(q_label)
        
        # Your existing 'Task' radio stays the same
        mode = st.radio("Task", ["Plan", "Evaluate", "Follow‑up"], horizontal=True)
        
        # Inputs (keep exactly as you had them)
        ans = st.text_area(
            "Student answer (paste or write here)",
            height=220,
            key=f"answer::{sel_case.get('id','unknown')}::{q_label}"
        )
        
        run = st.button(
            "Run task",
            type="primary",
            key=f"run_btn::{sel_case_id}::{q_index}::{mode}"   # <-- unique per case/question/mode
        )

    # --- (B) RIGHT: output stays on screen + download buttons ---
    with right:
        st.markdown("**Latest feedback**")

        # Retrieve any previously stored output for this case/question
        storage_key = _key(sel_case["id"], q_label)
        last = st.session_state.get("feedback_store", {}).get(storage_key, {})

        if run:
            if mode == "Plan":
                # (a) model-answer slice for the selected question
                sections = sel_case.get("model_answer_sections") or []
                model_slice = sections[q_index] if (0 <= q_index < len(sections)) else ""
            
                # (b) best booklet chapter (query: case title + question label + short anchor from model slice)
                #     You already have chap_retriever in scope.
                #     Keep the query short to avoid noise.
                anchor = (model_slice or "")[:400]
                query  = f"{sel_case.get('title','')} {q_label} {anchor}"
                best_chapter = chap_retriever.retrieve_best(query) if chap_retriever else None
                booklet_text = (best_chapter or {}).get("text", "")
            
                # call Plan with all 3 inputs
                with st.spinner("Planning..."):
                    plan = feedback_engine.plan_answer(
                        case_text=sel_case.get("description", f"[{sel_case['title']}]"),
                        question=q_label,
                        model_answer_slice=model_slice,
                        booklet_text=booklet_text,
                        model=model,
                        temperature=temp
                    )

                # persist on the right, as you already do
                st.session_state.setdefault("feedback_store", {})
                st.session_state["feedback_store"][f"{sel_case_id}::{q_index}"] = {
                    "mode": "Plan",
                    "answer": ans,
                    "feedback": plan
                }
                last = st.session_state["feedback_store"][f"{sel_case_id}::{q_index}"]
                        
            elif mode == "Evaluate":
                # Prefer JSON slice; fall back to the textarea if provided
                sections = sel_case.get("model_answer_sections") or []
                auto_slice = sections[q_index] if (0 <= q_index < len(sections)) else None
                
                effective_model_answer = (auto_slice or model_answer_slice or "").strip()
                if not effective_model_answer or not ans.strip():
                    st.warning("Missing model answer slice (JSON or pasted) or student answer.")
                else:
                    with st.spinner("Evaluating..."):
                        fb = feedback_engine.evaluate_answer(
                            student_answer=ans,
                            model_answer=effective_model_answer,
                            model=model,
                            temperature=temp
                        )
                    st.session_state.setdefault("feedback_store", {})
                    st.session_state["feedback_store"][f"{sel_case.get('id','unknown')}::{q_label}"] = {
                        "mode": "Evaluate",
                        "answer": ans,
                        "feedback": fb
                    }
    
            elif mode == "Follow‑up":
                if not last or not last.get("feedback"):
                    st.warning("No previous feedback found for this case/question. Run **Evaluate** first.")
                else:
                    follow_q = st.text_area("Follow‑up question", height=120, key=f"fu::{storage_key}")
                    if follow_q.strip():
                        with st.spinner("Answering follow‑up..."):
                            fu = feedback_engine.follow_up(
                                question=follow_q,
                                previous_feedback=last["feedback"],
                                model=model,
                                temperature=temp
                            )
                        st.session_state.setdefault("feedback_store", {})
                        st.session_state["feedback_store"][storage_key] = {
                            "mode": "Follow‑up",
                            "answer": last.get("answer", ""),
                            "feedback": fu
                        }
                        last = st.session_state["feedback_store"][storage_key]
                    else:
                        st.info("Type your follow‑up question above and click **Run task** again.")

        # Render the persisted output (survives re-runs)
        if last and last.get("feedback"):
            st.markdown(last["feedback"])
            st.divider()
            # Downloads (txt / md)
            txt = f"# Feedback ({sel_case['title']} – {q_label})\n\n{last['feedback']}\n\n---\nStudent answer:\n{last.get('answer','')}"
            md_bytes = txt.encode("utf-8")
            st.download_button(
                "⬇️ Download feedback (.txt)",
                data=md_bytes,
                file_name=f"feedback_{sel_case['id']}_{q_label.replace(' ','_')}.txt",
                mime="text/plain",
            )
            st.download_button(
                "⬇️ Download feedback + answer (.md)",
                data=md_bytes,
                file_name=f"feedback_{sel_case['id']}_{q_label.replace(' ','_')}.md",
                mime="text/markdown",
            )
        else:
            st.info("No feedback yet. Choose a task and click **Run task**.")

# --- Tutor chat (separate, uncluttered) ---
with tab_chat:
    st.subheader("Tutor chat (booklet‑grounded)")
    q = st.text_area("Your question", height=140, placeholder="e.g., What is 'inside information' under MAR?")
    if st.button("Ask", key="chat_btn"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                reply = chat_engine.answer(q, model=model, temperature=temp, max_tokens=800)
            st.markdown(reply)
