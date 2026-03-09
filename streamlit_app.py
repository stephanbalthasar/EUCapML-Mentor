# streamlit_app.py
# Minimal UI to exercise both engines using the booklet index from the private repo.

# ── imports (top of file) ──────────────────────────────────────────────────────
import base64
import json
import mimetypes
import os
import requests
import streamlit as st
import time
from typing import Callable, List, Dict, Any
# ───────────────────────────────────────────────────────────────────────────────

# === HELPERS ===
# === APP BAR ===
def render_brand_bar_aligned(
    icon_src: str = "assets/logo_128.png",
    title: str = "Balthasar's Code",
    subhead: str = "Your European Capital Markets Law AI Mentor.",
    bar_height_desktop: int = 44,
    bar_height_mobile: int = 38,
    logo_top_nudge_px: int = 0,
    title_nudge_px: int = 0,
    sub_nudge_px: int = 0
) -> None:
    # embed image as data-URI
    try:
        mime, _ = mimetypes.guess_type(icon_src)
        if not mime:
            mime = "image/png"
        with open(icon_src, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        img_data_uri = f"data:{mime};base64,{b64}"
    except Exception:
        img_data_uri = icon_src

    st.markdown(
        f"""
<style>
  .lc-appbar {{
    background: #F6F8FC;
    border: 1px solid #E7EAF0;
    border-radius: 10px;
    padding: 8px 10px;
    margin: 6px 0 12px 0;
  }}
  .lc-appbar-row {{
    display: flex;
    align-items: flex-start;             /* lock logo top to text top */
    gap: 10px;
  }}
  .lc-appbar-logo {{
    height: {bar_height_desktop}px;
    width: auto;
    display: block;
    position: relative;
    top: {-logo_top_nudge_px}px;
  }}
  .lc-appbar-text {{
    display: flex;
    flex-direction: column;
    height: {bar_height_desktop}px;      /* SAME height as logo */
    min-height: {bar_height_desktop}px;
  }}
  .lc-appbar-title {{
    margin: 0;
    font-weight: 700;
    font-size: 1.12rem;
    line-height: 1.08;
    position: relative;
    top: {title_nudge_px}px;
    color: #0B1F3B;
  }}
  .lc-appbar-sub {{
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.08;
    margin-top: auto;                    /* pin subtitle to bottom edge */
    position: relative;
    top: {sub_nudge_px}px;
    color: #0B1F3B; opacity: 0.90;
  }}
  @media (max-width: 680px) {{
    .lc-appbar-logo {{ height: {bar_height_mobile}px; }}
    .lc-appbar-text {{ height: {bar_height_mobile}px; min-height: {bar_height_mobile}px; }}
    .lc-appbar-title {{ font-size: 1.02rem; }}
    .lc-appbar-sub {{ font-size: 0.9rem; }}
  }}
</style>

<div class="lc-appbar">
  <div class="lc-appbar-row">
    <img class="lc-appbar-logo" src="{img_data_uri}" alt="Balthasar's Code icon"/>
    <div class="lc-appbar-text">
      <div class="lc-appbar-title">{title}</div>
      <div class="lc-appbar-sub">{subhead}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
# =============================
# Conversation utilities (DRY)
# =============================
def _approx_tokens(s: str) -> int:
    """Rough token estimate (~4 chars / token)."""
    return max(1, len(s) // 4)

def _build_history_preamble(history: List[Dict[str, str]], max_tokens: int = 6000) -> str:
    """
    Create a compact transcript (User/Assistant lines) from the most recent turns
    within a token budget; keeps newest content.
    """
    if not history:
        return ""
    acc, used = [], 0
    for turn in reversed(history):
        line = f'{"User" if turn["role"]=="user" else "Assistant"}: {turn["content"]}'.strip()
        t = _approx_tokens(line)
        if used + t > max_tokens:
            break
        acc.append(line); used += t
    acc.reverse()
    return "\n".join(acc)

def _ensure_thread(state_key: str) -> List[Dict[str, Any]]:
    """
    Ensure st.session_state[state_key] is a list[dict] with keys: role in {"user","assistant"}, content, ts.
    Migrates old tuple-based history [('student'|'tutor', 'msg')] if present.
    Returns the normalized list.
    """
    st.session_state.setdefault(state_key, [])
    thread = st.session_state[state_key]
    if thread and isinstance(thread[0], tuple):
        # migrate tuples -> dicts; keep a best-effort role mapping
        mapped: List[Dict[str, Any]] = []
        for role, msg in thread:
            r = "user" if role in ("user", "student") else "assistant"
            mapped.append({"role": r, "content": msg, "ts": time.time()})
        st.session_state[state_key] = mapped
        thread = mapped
    return thread

def render_conversation(
    *,
    state_key: str,
    title: str,
    placeholder: str,
    on_ask: Callable[[str, List[Dict[str, Any]]], str],
    clear_label: str = "🗑️ Clear chat",
    before_input: Callable[[], None] | None = None,
) -> None:
    """
    Generic chat renderer:
    - Keeps a thread in st.session_state[state_key]
    - Renders bubbles with st.chat_message + st.chat_input
    - Calls on_ask(user_msg, history_without_current) -> assistant_reply
    - Appends both turns back to the thread
    - Guarantees that the input box appears at the bottom after the latest answer
    """
    st.subheader(title)
    thread = _ensure_thread(state_key)

    # Optional top controls (e.g., info boxes)
    if before_input:
        before_input()

    # --- Transcript (messages) ---
    for msg in thread:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # --- Bottom-placed clear button (appears only when there is history) ---
    if thread:
        spacer, btn_col = st.columns([4, 1])
        with btn_col:
            if st.button(clear_label, key=f"clear_bottom__{state_key}"):
                st.session_state[state_key] = []
                st.rerun()

    # --- Input (unique key per thread avoids collisions when used in several places) ---
    user_q = st.chat_input(placeholder, key=f"chat_input__{state_key}")
    if not user_q:
        return

    # Show user turn immediately
    user_turn = {"role": "user", "content": user_q, "ts": time.time()}
    thread.append(user_turn)
    with st.chat_message("user"):
        st.markdown(user_q)

    # Call the app-provided handler with history (excluding the new turn)
    prior = thread[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = on_ask(user_q, prior)
            st.markdown(answer)

    # Append assistant turn
    thread.append({"role": "assistant", "content": answer, "ts": time.time()})

    # Re-run so Streamlit re-renders the full transcript and places a fresh input box
    # *after* the last answer (so the input is always at the bottom).
    st.rerun()

# --- HERO (flat navy) ---
def render_sticky_footer():
    st.markdown(
        """
        <style>
        /* --- Sticky footer --- */
        .sb-sticky-foot{
          position: fixed; bottom: 0; left: 0; right: 0;
          background: #ffffff;
          border-top: 1px solid #E7EAF0;
          box-shadow: 0 -2px 8px rgba(5,16,28,0.06);
          z-index: 9999;
          /* Respect iOS home indicator / safe area */
          padding-bottom: env(safe-area-inset-bottom, 0px);
        }

        .sb-sticky-foot .sb-inner{
          max-width: 1120px;
          margin: 0 auto;
          display: flex; align-items: center;
          gap: 12px; padding: 8px 12px;
          /* Allow wrapping on small screens */
          white-space: normal;
          flex-wrap: wrap;
        }

        .sb-foot-btn{
          display: inline-block;
          background: #123B7A; color: #fff !important;
          border: 1px solid #123B7A;
          padding: 6px 12px; border-radius: 999px;
          text-decoration: none; font-weight: 600; font-size: 0.9rem;
        }
        .sb-foot-btn:hover{ background:#0F2D5A; border-color:#0F2D5A; }
        .sb-foot-btn:focus-visible{ outline: 2px solid #1464A5; outline-offset: 2px; }

        .sb-footnote{
          color: #5B677A; font-size: 0.9rem;
        }

        /* --- Reserve space for the sticky footer, dynamically --- */
        :root{
          /* Updated by JS below */
          --sb-foot-h: 72px;
        }

        /* Pad the main content so it never hides behind the footer.
           Apply to both common Streamlit containers to be robust across versions. */
        .block-container,
        section.main > div {
          padding-bottom: calc(var(--sb-foot-h, 72px) + env(safe-area-inset-bottom, 0px)) !important;
        }

        /* A little extra bottom air for very small viewports */
        @media (max-width: 480px) {
          .block-container,
          section.main > div {
            padding-bottom: calc(var(--sb-foot-h, 72px) + env(safe-area-inset-bottom, 0px) + 12px) !important;
          }
        }
        </style>

        <div class="sb-sticky-foot">
          <div class="sb-inner">
            <a class="sb-foot-btn" href="?show_privacy=1">AI & Privacy Notice</a>
            <div class="sb-footnote">
              © 2026 Stephan Balthasar · This app uses AI & LLMs; outputs may be inaccurate; no liability.
              Feedback is not a grade predictor.
            </div>
          </div>
        </div>

        <script>
          (function() {
            function setFootHeight(){
              try{
                var f = document.querySelector('.sb-sticky-foot');
                var h = f ? f.offsetHeight : 72;
                document.documentElement.style.setProperty('--sb-foot-h', h + 'px');
              } catch(e) {}
            }
            window.addEventListener('load', setFootHeight);
            window.addEventListener('resize', setFootHeight);
            window.addEventListener('orientationchange', setFootHeight);

            // If Streamlit re-renders parts of the DOM, keep the measurement fresh
            var obs = new MutationObserver(setFootHeight);
            obs.observe(document.body, {childList: true, subtree: true});
          })();
        </script>
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

st.set_page_config(
    page_title="Balthasar's Code",
    page_icon="assets/logo_256.png",
    layout="wide",
    initial_sidebar_state="collapsed"  # NEW: collapse sidebar by default
)

# Global width cap for a professional look (applies to all pages)
st.markdown("""
<style>
  /* Constrain the main content and center it */
  .block-container, section.main > div {
    max-width: 1120px !important;   /* pick 1040–1160px if you prefer */
    margin: 0 auto !important;
  }
</style>
""", unsafe_allow_html=True)

# === Load brand CSS (scoped details not covered by theme) ===
try:
    with open("assets/theme.css", "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

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
    # Hide the sidebar on the landing page only
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] { display: none !important; }
        /* Slightly tighten top/bottom padding while the sidebar is hidden */
        .block-container { padding-top: 0.75rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Landing
    render_brand_bar_aligned(
        icon_src="assets/logo_128.png",
        title="Balthasar's Code",
        subhead="Your European Capital Markets Law AI Mentor.",
        bar_height_desktop=44, bar_height_mobile=38,
        logo_top_nudge_px=0, title_nudge_px=3, sub_nudge_px=-3
    )
           
    STUDENT_PIN = st.secrets.get("STUDENT_PIN")
    TUTOR_PIN   = st.secrets.get("TUTOR_PIN")

    # --- Simple, robust login form ---
    # Requirements: show password + checkbox at the same time; allow Enter OR button click.
    with st.form(key="login_form", clear_on_submit=False):
        pin = st.text_input("Enter password", type="password")
        agree = st.checkbox(
            "Confirm AI & Privacy Notice (see blue button in footer)."
        )
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        role_detected = None
        if pin and STUDENT_PIN and pin == STUDENT_PIN:
            role_detected = "student"
        elif pin and TUTOR_PIN and pin == TUTOR_PIN:
            role_detected = "tutor"

        # Collect validation messages per spec
        messages = []
        if role_detected is None:
            messages.append("Incorrect password")
        if not agree:
            messages.append("Tick the box first")

        if messages:
            for m in messages:
                st.error(m)
        else:
            # Successful login
            st.session_state.authenticated = True
            st.session_state.role = role_detected
            if role_detected == "student":
                # log student login
                update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "LOGIN", "student"])
            st.rerun()

    # Stop rendering the rest of the app until authenticated
    st.stop()

# Compact brand bar (authenticated pages only)
render_brand_bar_aligned(
    icon_src="assets/logo_128.png",
    title="Balthasar's Code",
    subhead="Your European Capital Markets Law AI Mentor.",
    bar_height_desktop=44, bar_height_mobile=38,
    logo_top_nudge_px=0, title_nudge_px=3, sub_nudge_px=-3
)

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
    booklet_retriever=para_retriever,  
    web_retriever=None                 
)
feedback_engine = FeedbackEngine(llm=llm)

# --- Sidebar controls ---
with st.sidebar:
    st.caption(f"📖 Booklet loaded — {len(INDEX['chapters'])} chapters / {len(INDEX['paragraphs'])} paragraphs")

    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
        help=(
            "Model choice:\n"
            "• llama‑3.1‑8b‑instant → faster, cheaper; good for drafts and everyday Q&A.\n"
            "• llama‑3.3‑70b‑versatile → slower, more capable; better for nuanced legal analysis."
        ),
    )

    temp = st.slider(
        "Temperature",
        0.0, 1.0, 0.2, 0.05,
        help=(
            "Controls randomness.\n"
            "• 0.0–0.3 → deterministic, best for legal reasoning.\n"
            "• 0.4–0.7 → more creative.\n"
            "Higher values may produce inconsistent answers."
        ),
    )

    if st.button("Reload booklet index (server cache)"):
        st.cache_data.clear()
        st.success("Re-loaded. Re-run the action to use the latest JSON.")

# --- Tabs: Feedback + Tutor chat ---
tab_feedback, tab_chat = st.tabs(["📝 Sample Exam Cases", "💬 General Tutor Chat"])

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

    # -----------------------------
    # 2. CASE DESCRIPTION
    # -----------------------------
    st.markdown("### Case description")
    st.write(sel_case.get("description", "—"))

    # -----------------------------
    # 3. QUESTION SELECTION
    # -----------------------------
    q_count = int(sel_case.get("question_count", 1))
    q_labels = [f"Question {i+1}" for i in range(max(1, q_count))]
    q_label = st.selectbox("Which question are you working on?", q_labels, index=0)
    q_index = q_labels.index(q_label)

    st.divider()
    
    # -----------------------------
    # 4. WORKFLOW CHOICE
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
            # --- Follow-up discussion (reuses the same conversation component) ---
            with st.container():
                def on_ask_followup(user_q: str, history: List[Dict[str, Any]]) -> str:
                    # Build the history in the legacy tuple format that your feedback_engine expects
                    tuple_hist = [
                        ("student" if m["role"] == "user" else "tutor", m["content"])
                        for m in history
                    ]
                    context = {
                        "student_answer": st.session_state.get("exam_answer", ""),
                        "feedback": st.session_state.get("exam_feedback", ""),
                        "history": tuple_hist,
                    }
                    # Keep your audit behavior: log only for students
                    if st.session_state.get("role") == "student":
                        update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "FOLLOW_UP", "student"])
            
                    return feedback_engine.follow_up_with_history(
                        question=user_q,
                        context=context,
                        model=model,
                        temperature=temp,
                    )
            
                # Use the same state key you already reset after evaluation to avoid surprises
                render_conversation(
                    state_key="chat_history",  # we keep the same key to preserve behavior / resets
                    title="Follow-up discussion",
                    placeholder="Add your follow-up question…",
                    on_ask=on_ask_followup,
                    clear_label="🗑️ Clear follow-up thread",
                )

# --- Tutor chat (separate, uncluttered) ---
# --- General Tutor Chat (conversation mode + booklet grounding) ---
with tab_chat:
    def on_ask_tutor(user_q: str, history: List[Dict[str, Any]]) -> str:
        # audit log for students (existing behavior)
        if st.session_state.get("role") == "student":
            update_gist([time.strftime("%Y-%m-%d %H:%M:%S"), "CHAT", "student"])

        # Build a compact preamble from prior turns; keep retrieval focused on CURRENT question
        preamble = _build_history_preamble(history, max_tokens=6000)
        composed_q = (
            "You are in an ongoing conversation. Use the 'Conversation so far' only for context; "
            "when searching the booklet, focus on the CURRENT question. "
            "If you cite legal norms, prefer short forms (e.g., 'MAR Art. 17'). "
            "Avoid fabricating case law.\n\n"
            f"Conversation so far (most recent last):\n{preamble}\n\n"
            f"CURRENT question:\n{user_q}"
        ).strip()

        try:
            # Slightly lower max_tokens to ease rate pressure (optional)
            return chat_engine.answer(
                composed_q,
                model=model,
                temperature=temp,
                max_tokens=700,  # was 800; small reduction to ease 429s
            )
        except Exception as e:
            # Show a friendly, actionable message in the transcript
            msg = str(e)
            if "rate" in msg.lower() or "429" in msg:
                return (
                    "⏳ We’re hitting the provider’s rate limit right now. "
                    "Please wait ~10–20 seconds and ask again."
                )
            # Fallback general message
            return "Sorry—there was a temporary issue. Please try again in a few seconds."

    # 👉 This call actually renders the chat UI inside the tab
    render_conversation(
        state_key="tutor_chat",
        title="Tutor chat (conversation mode)",
        placeholder="Ask the tutor…",
        on_ask=on_ask_tutor,
        clear_label="🗑️ Clear chat",
    )
