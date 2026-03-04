# streamlit_app.py
# Minimal UI to exercise both engines using the booklet index from the private repo.


import json, time, os  # time is used by your existing call sites
import requests  
import streamlit as st

# === PATCH 1: sticky footer ===
def render_sticky_footer():
    st.markdown(
        """
<style>
.stApp { padding-bottom: 64px; }
section.main > div { padding-bottom: 64px; }
.eucapml-fixed-footer {
  position: fixed; left: 0; right: 0; bottom: 0;
  z-index: 99999;
  background: #0e1117; color: rgba(255,255,255,.92);
  border-top: 1px solid rgba(255,255,255,.12);
  padding: .70rem 1rem; font-size: .92rem; line-height: 1.35rem;
}
.eucapml-fixed-footer .inner {
  max-width: 1200px; margin: 0 auto;
  display: flex; gap: .75rem; align-items: center; flex-wrap: wrap;
}
.eucapml-fixed-footer .spacer { flex: 1; }
.eucapml-fixed-footer a.btn {
  display: inline-block; background: #1a73e8; color: #fff !important;
  padding: .35rem .75rem; border-radius: 6px; font-weight: 600; text-decoration: none;
}
.eucapml-fixed-footer a.btn:hover { background: #165fc1; }
</style>
<div class="eucapml-fixed-footer" role="contentinfo" aria-label="Legal note and privacy">
  <div class="inner">
    <span>
      ℹ️ <strong>Notes</strong>:
      (c) 2026 by Stephan Balthasar. This app uses AI & LLMs. Output may be inaccurate, and no liability is accepted.
      App feedback is no indicator for grades in a real examination.
    </span>
    <span class="spacer"></span>
    <a class="btn" href="?show_privacy=1" title="View AI & Privacy Notice">AI & Privacy Notice</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# === PATCH 1: load the notice (Markdown file) ===
def load_privacy_notice():
    file_path = os.path.join("assets", "Notice.md")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "# Privacy Notice\n\nThe notice could not be loaded. Please contact the administrator."

# === PATCH 1: privacy overlay (opens via ?show_privacy=1) ===
def _get_query_params():
    import streamlit as st
    try:
        return st.experimental_get_query_params()  # legacy API
    except Exception:
        try:
            return dict(st.query_params)  # newer API
        except Exception:
            return {}

def render_privacy_overlay_if_requested():
    qp = _get_query_params()
    val = str(qp.get("show_privacy", ["0"])[0]).lower()
    show = val in ("1", "true", "yes", "y", "on")
    if not show:
        return

    notice_md = load_privacy_notice()
    st.title("AI & Privacy Notice")
    st.markdown(notice_md)
    st.markdown("[← Back to the app](?)")
    render_sticky_footer()
    st.stop()

# --- minimalist logger: uses only LOG_GIST_TOKEN + GIST_ID ---
def update_gist(new_entry):
    """
    Append [timestamp, event, role] to EUCapML_Mentor_Log.csv in a GitHub Gist.
    Uses a dedicated token only: st.secrets['LOG_GIST_TOKEN'].
    If not configured, this function silently no-ops.
    """
    import streamlit as st

    token = st.secrets.get("LOG_GIST_TOKEN")
    gist_id = st.secrets.get("GIST_ID")
    if not token or not gist_id:
        return  # no-op if logging is not configured

    url = f"https://api.github.com/gists/{gist_id}"
    headers = {"Authorization": f"token {token}"}

    # 1) Read current EUCapML_Mentor_Log.csv (or start with the header)
    try:
        r = requests.get(url, headers=headers, timeout=8)
        files = r.json().get("files", {}) if r.status_code == 200 else {}
        content = files.get("EUCapML_Mentor_Log.csv", {}).get("content", "")
        lines = [ln for ln in content.splitlines() if ln.strip()] or ["timestamp,event,role"]
    except Exception:
        lines = ["timestamp,event,role"]

    # 2) Append the new entry and push
    lines.append(",".join(new_entry))
    payload = {"files": {"EUCapML_Mentor_Log.csv": {"content": "\n".join(lines)}}}
    try:
        requests.patch(url, headers=headers, data=json.dumps(payload), timeout=8)
    except Exception:
        # Best-effort logging — ignore network/api errors to keep UX smooth
        pass

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

# === PATCH 2: always-on footer and optional overlay ===
render_privacy_overlay_if_requested()
render_sticky_footer()

# === PATCH 3: session flags ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = None

# === PATCH 3: login gate ===
if not st.session_state.authenticated:
    st.title("European Capital Markets Law – Digital Tutor")

    STUDENT_PIN = st.secrets.get("STUDENT_PIN")
    TUTOR_PIN   = st.secrets.get("TUTOR_PIN")

    pin = st.text_input("Enter password", type="password")

    # Show the consent checkbox only after a correct PIN is typed
    role_detected = None
    if pin and pin == STUDENT_PIN:
        role_detected = "student"
        st.success("Password accepted.")
    elif pin and pin == TUTOR_PIN:
        role_detected = "tutor"
        st.success("PIN accepted (tutor).")
    elif pin:
        st.error("Incorrect PIN. Please try again.")

    if role_detected:
        agree = st.checkbox(
            "I confirm I have read the AI & Privacy Notice (see the blue footer button)."
        )
        st.caption("You must accept to continue.")
        if st.button("Continue", type="primary", disabled=not agree):
            st.session_state.authenticated = True
            st.session_state.role = role_detected
            if role_detected == "student":
                # log student login
                update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "LOGIN", "student"])
            st.rerun()

    # Stop rendering the rest of the app until authenticated
    st.stop()

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

# --- REPLACEMENT FEEDBACK TAB (full block) ----

with tab_feedback:

    st.subheader("Exam Assistant")

    # -----------------------------
    # 1. CASE SELECTION (kept unchanged)
    # -----------------------------
    case_titles = [c.get("title", c.get("id", "Untitled case")) for c in CASES]
    sel_case_title = st.selectbox("Select exam case", case_titles, index=0)
    sel_case = next(c for c in CASES if c.get("title", c.get("id")) == sel_case_title)
    sel_case_id = sel_case.get("id", "unknown")

    q_count = int(sel_case.get("question_count", 1))
    q_labels = [f"Question {i+1}" for i in range(max(1, q_count))]
    q_label = st.selectbox("Which question are you working on?", q_labels, index=0)
    q_index = q_labels.index(q_label)

    # -----------------------------
    # 2. CASE DESCRIPTION
    # -----------------------------
    st.markdown("### Case description")
    st.write(sel_case.get("description", "—"))

    st.divider()

    # -----------------------------
    # 3. WORKFLOW CHOICE
    # -----------------------------
    workflow = st.radio(
        "Choose your workflow:",
        ["Help me prepare an answer", "I have an answer ready to submit"],
        horizontal=False
    )

    st.divider()


    # -----------------------------
    # 4. PLAN WORKFLOW
    # -----------------------------
    if workflow == "Help me prepare an answer":
        st.markdown("## Plan your answer")
        st.markdown("The app will help you build a structured outline based on the case and model solution.")

        # Load model answer slice
        sections = sel_case.get("model_answer_sections") or []
        model_slice = sections[q_index] if (0 <= q_index < len(sections)) else ""

        if st.button("Generate plan", type="primary"):
            if st.session_state.get("role") == "student":
                update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "PLAN", "student"])
            with st.spinner("Thinking..."):
                plan = feedback_engine.plan_answer(
                    case_text=sel_case.get("description", ""),
                    question=q_label,
                    model_answer_slice=model_slice,
                    booklet_text="",       # your design: no booklet grounding in plan
                    model=model,
                    temperature=temp
                )
            st.session_state["plan_output"] = plan

        # Display generated plan
        if "plan_output" in st.session_state:
            st.markdown("### Suggested solution structure")
            st.markdown(st.session_state["plan_output"])


    # -----------------------------
    # 5. EVALUATE WORKFLOW
    # -----------------------------
    if workflow == "I have an answer ready to submit":

        st.markdown("## Submit your exam answer")

        # text input
        answer = st.text_area(
            "Your answer",
            height=240,
            key=f"answer::{sel_case_id}::{q_label}"
        )

        # on evaluate
        if st.button("Evaluate my answer", type="primary"):
            if st.session_state.get("role") == "student":
                update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "EVALUATE", "student"])
            sections = sel_case.get("model_answer_sections") or []
            auto_slice = sections[q_index] if (0 <= q_index < len(sections)) else None
            effective_model = (auto_slice or "").strip()

            if not effective_model or not answer.strip():
                st.warning("Missing model answer slice or student answer.")
            else:
                with st.spinner("Evaluating..."):
                    fb = feedback_engine.evaluate_answer(
                        student_answer=answer,
                        model_answer=effective_model,
                        model=model,
                        temperature=temp
                    )

                # persist all relevant state
                st.session_state["exam_answer"] = answer
                st.session_state["exam_feedback"] = fb
                st.session_state["chat_history"] = []    # reset chat thread

        # SHOW RESULTS AFTER EVALUATION
        if "exam_feedback" in st.session_state:

            st.markdown("## Your submitted answer")
            st.markdown(st.session_state["exam_answer"])

            st.markdown("## Structured feedback")
            st.markdown(st.session_state["exam_feedback"])

            # -----------------------------
            # DOCX DOWNLOAD
            # -----------------------------
            from docx import Document
            from io import BytesIO

            def make_docx():
                doc = Document()
                doc.add_heading(f"Feedback – {sel_case_title} – {q_label}", level=1)

                doc.add_heading("Student Answer", level=2)
                doc.add_paragraph(st.session_state["exam_answer"])

                doc.add_heading("Feedback", level=2)
                doc.add_paragraph(st.session_state["exam_feedback"])

                buf = BytesIO()
                doc.save(buf)
                buf.seek(0)
                return buf

            st.download_button(
                "📄 Download feedback (.docx)",
                data=make_docx(),
                file_name=f"feedback_{sel_case_id}_{q_label.replace(' ','_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            st.divider()

            # -----------------------------
            # 6. FOLLOW-UP CHAT
            # -----------------------------
            st.markdown("## Follow-up discussion")

            st.session_state.setdefault("chat_history", [])

            # show history
            for role, msg in st.session_state["chat_history"]:
                if role == "student":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**Tutor:** {msg}")

            follow_q = st.text_area("Your follow-up question", height=120)

            if st.button("Send follow-up"):
                if follow_q.strip():

                    # add user's message
                    st.session_state["chat_history"].append(("student", follow_q))

                    # build conversation context
                    context = {
                        "student_answer": st.session_state["exam_answer"],
                        "feedback": st.session_state["exam_feedback"],
                        "history": st.session_state["chat_history"],
                    }

                    # new engine call
                    reply = feedback_engine.follow_up_with_history(
                        question=follow_q,
                        context=context,
                        model=model,
                        temperature=temp
                    )

                    # store bot reply
                    st.session_state["chat_history"].append(("tutor", reply))

                    # force re-render
                    st.rerun()

# --- Tutor chat (separate, uncluttered) ---
with tab_chat:
    st.subheader("Tutor chat (booklet‑grounded)")
    q = st.text_area("Your question", height=140, placeholder="e.g., What is 'inside information' under MAR?")
    if st.button("Ask", key="chat_btn"):
        if st.session_state.get("role") == "student":
            update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "CHAT", "student"])
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                reply = chat_engine.answer(q, model=model, temperature=temp, max_tokens=800)
            st.markdown(reply)
