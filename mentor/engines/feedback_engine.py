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
          If none picked: fallback once using (question + feedback) candidates and re-score.
        - Append footer if we have picks.
        - Emit sidebar diagnostics.
        - Return a plain string.
        """
        # --- Diagnostics: prove which file/class is running ---
        try:
            import streamlit as st, inspect, time
            st.sidebar.info(f"[FE] live: {time.time():.0f}")
            st.sidebar.write(f"[FE] module: {FeedbackEngine.__module__}")
            st.sidebar.write(f"[FE] file: {inspect.getsourcefile(FeedbackEngine)}")
        except Exception:
            pass
    
        # 0) Base messages
        messages = []
        messages.append({"role": "system", "content": f"Student exam answer:\n{context['student_answer']}"})
        messages.append({"role": "system", "content": f"Feedback:\n{context['feedback']}"})
    
        # 1) Prompt grounding: question + prior feedback
        booklet_chunks: list[str] = []
        if getattr(self, "booklet_retriever", None) is not None:
            try:
                _hits_q, chunks_q = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever, question or "", top_k=15
                )
                _hits_fb, chunks_fb = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever, (context.get("feedback") or ""), top_k=15
                )
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
        try:
            import streamlit as st
            st.sidebar.write(f"[FE] booklet_chunks_in_prompt: {len(booklet_chunks)}")
        except Exception:
            pass
    
        if booklet_chunks:
            block = "Relevant booklet excerpts:\n" + "\n\n".join(f"- {c}" for c in booklet_chunks)
            messages.append({"role": "system", "content": block})
    
        # 3) Prior chat turns
        for role, msg in context["history"]:
            messages.append({
                "role": "user" if role == "student" else "assistant",
                "content": msg
            })
    
        # 4) Current question
        messages.append({"role": "user", "content": question})
    
        # 5) LLM answer
        raw = self.llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=800)
        reply_text = raw if isinstance(raw, str) else str(raw)
    
        # 6) Gate (deterministic)
        try:
            gate_msgs = build_sources_gate_messages(user_query=question, answer_text=reply_text)
            gate_raw = self.llm.chat(messages=gate_msgs, model=model, temperature=0.0, max_tokens=4)
            gate_txt = gate_raw if isinstance(gate_raw, str) else str(gate_raw)
            show_sources = gate_txt.strip().upper().startswith("YES")
        except Exception:
            gate_txt, show_sources = "EXC", False
    
        # 7) Select supporting paragraphs (answer-driven), with fallback
        picked: list[str] = []
        if show_sources and getattr(self, "booklet_retriever", None) is not None:
            try:
                picked = select_supporting_paragraphs(
                    answer_text=reply_text,
                    hits=None,
                    booklet_retriever=self.booklet_retriever,
                    top_k=15,
                    max_n=5,
                )
            except Exception:
                picked = []
            if not picked:
                try:
                    qf = f"{question or ''}\n{context.get('feedback') or ''}".strip()
                    cand_hits, _ = fetch_booklet_chunks_for_prompt(self.booklet_retriever, qf, top_k=15)
                    picked = select_supporting_paragraphs(
                        answer_text=reply_text,
                        hits=cand_hits,
                        booklet_retriever=self.booklet_retriever,
                        max_n=5,
                    )
                except Exception:
                    pass
    
        if picked:
            reply_text += "\n\n---\n" + "_Key paragraphs: " + ", ".join(picked) + "._"
    
        # 8) Sidebar debug
        try:
            import streamlit as st
            st.sidebar.write(f"[FE] gate_raw: {gate_txt!r}")
            st.sidebar.write(f"[FE] gate_parsed: {'YES' if show_sources else 'NO'}")
            st.sidebar.write(f"[FE] sources_selected: {len(picked)}")
        except Exception:
            pass
    
        return reply_text
