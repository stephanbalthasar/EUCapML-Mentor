# mentor/engines/feedback_engine.py

from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
    build_sources_gate_messages,
)

from mentor.rag.booklet_retriever import fetch_booklet_chunks_for_prompt

from mentor.rag.supporting_sources_selector import select_supporting_paragraphs

class FeedbackEngine:
    def __init__(self, llm, booklet_retriever=None):
        self.llm = llm
        self.booklet_retriever = booklet_retriever
        if self.booklet_retriever is None:
            try:
                import streamlit as st
                st.sidebar.warning(
                    "[FE] No booklet retriever attached – prompt grounding and source selection are disabled."
                )
            except Exception:
                pass

    # -------------------------------------------------------
    # (i) PLAN  ---- CHANGED SIGNATURE ----
    # -------------------------------------------------------
    def plan_answer(self, *,
                    case_text: str,
                    question: str,
                    model_answer_slice: str | None,
                    booklet_text: str | None,
                    model: str,
                    temperature: float) -> str:
        messages = build_plan_messages(
            case_text=case_text,
            question_label=question,
            model_answer_slice=model_answer_slice,
            booklet_text=booklet_text
        )
        # For planning, prefer tight settings
        return self.llm.chat(messages=messages, model=model, temperature=min(temperature, 0.2), max_tokens=350)

    # -------------------------------------------------------
    # (ii) Evaluate a submitted answer  — use the prompt builder
    # -------------------------------------------------------
    def evaluate_answer(self, *, student_answer, model_answer, model, temperature, max_words=300):
        # Build the structured, five‑heading prompt with a word ceiling
        messages = build_evaluate_messages(
            student_answer=student_answer,
            model_answer=model_answer,
            max_words=max_words
        )
        # Optional one‑time fingerprint for debugging which path runs:
        # messages.insert(0, {"role": "system", "content": "[EVAL_PATH=v2]"})
        raw = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=900  # generous but bounded; headings + content fit comfortably
        )
        return raw if isinstance(raw, str) else str(raw)

    # -------------------------------------------------------
    # (iii) Follow-up questions about the feedback
    # -------------------------------------------------------

    def follow_up_with_history(self, question, context, model, temperature):
        """
        Follow-up chat (exam module):
        - Ground the prompt with booklet snippets (from question + prior feedback).
        - Generate reply.
        - Gate (YES/NO) whether to show sources (temp=0).
        - If YES: run answer-driven selector (retrieves with answer text).
          If that picks nothing: fallback once by retrieving with (question + feedback)
          and re-scoring those hits against the final answer.
        - Append footer if we have picks.
        - Show gate + number of sources in the Streamlit sidebar.
        - Return a plain string (no trailing comma).
        """
        # 0) Base messages (kept)
        messages = []
        messages.append({"role": "system", "content": f"Student exam answer:\n{context['student_answer']}"})
        messages.append({"role": "system", "content": f"Feedback:\n{context['feedback']}"})
    
        # 1) Retrieve booklet paragraphs using BOTH the follow-up question AND the feedback text (for prompt grounding)
        booklet_chunks: list[str] = []
        if getattr(self, "booklet_retriever", None) is not None:
            try:
                # A) from follow-up question
                _hits_q, chunks_q = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever, question or "", top_k=15
                    # , truncate_chars=700  # optional
                )
                # B) from prior feedback text
                _hits_fb, chunks_fb = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever, (context.get("feedback") or ""), top_k=15
                    # , truncate_chars=700
                )
                # C) merge + dedupe; cap to 12 for prompt brevity
                merged, seen = [], set()
                for t in (chunks_q + chunks_fb):
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    merged.append(t)
                    if len(merged) == 12:
                        break
                booklet_chunks = merged
            except Exception:
                booklet_chunks = []
    
        # 2) Inject excerpts as a system block (only if we have any)
        if booklet_chunks:
            block = "Relevant booklet excerpts:\n" + "\n\n".join(f"- {c}" for c in booklet_chunks)
            messages.append({"role": "system", "content": block})
    
        # 3) Prior chat turns (kept)
        for role, msg in context["history"]:
            messages.append({
                "role": "user" if role == "student" else "assistant",
                "content": msg
            })
    
        # 4) Current question (kept)
        messages.append({"role": "user", "content": question})
    
        # 5) Call LLM and coerce to plain string (no early return)
        raw = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=800,
        )
        reply_text = raw if isinstance(raw, str) else str(raw)
    
        # 6) Gate: should we attach sources for THIS answer? (deterministic, temp=0)
        try:
            gate_msgs = build_sources_gate_messages(user_query=question, answer_text=reply_text)
            gate_raw = self.llm.chat(messages=gate_msgs, model=model, temperature=0.0, max_tokens=4)
            gate_txt = gate_raw if isinstance(gate_raw, str) else str(gate_raw)
            show_sources = gate_txt.strip().upper().startswith("YES")
        except Exception:
            gate_txt, show_sources = "EXC", False  # conservative + defined for debug
    
        # 7) If YES, run the selector so sources correspond to the *answer*
        picked: list[str] = []
        if show_sources and getattr(self, "booklet_retriever", None) is not None:
            try:
                # 7a) Primary path: answer-driven retrieval (selector retrieves with answer_text)
                picked = select_supporting_paragraphs(
                    answer_text=reply_text,
                    hits=None,                     # selector retrieves with answer_text as query
                    booklet_retriever=self.booklet_retriever,
                    top_k=15,
                    max_n=5,
                )
            except Exception:
                picked = []
    
            # 7b) Fallback: if nothing was picked, re-retrieve once using (question + feedback)
            if not picked:
                try:
                    qf = f"{question or ''}\n{context.get('feedback') or ''}".strip()
                    cand_hits, _ = fetch_booklet_chunks_for_prompt(
                        self.booklet_retriever, qf, top_k=15
                        # , truncate_chars=700
                    )
                    picked = select_supporting_paragraphs(
                        answer_text=reply_text,
                        hits=cand_hits,              # re-score these against the final answer
                        booklet_retriever=self.booklet_retriever,
                        max_n=5,
                    )
                except Exception:
                    pass
    
            if picked:
                reply_text += "\n\n---\n" + "_Key paragraphs: " + ", ".join(picked) + "._"
    
        # 8) Minimal Streamlit sidebar debug (always visible in Streamlit runs)
        try:
            import streamlit as st
            st.sidebar.write(f"[FE] gate: {'YES' if show_sources else 'NO'}")
            st.sidebar.write(f"[FE] sources: {len(picked)}")
        except Exception:
            # If Streamlit isn't available (e.g., during CLI unit tests), ignore
            pass
    
        # 9) Return a plain string (no trailing comma)
        return reply_text
