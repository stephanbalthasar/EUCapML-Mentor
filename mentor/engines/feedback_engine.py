# mentor/engines/feedback_engine.py
from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
)
from mentor.rag.booklet_references_selector import rank_paragraphs_by_text, pick_para_nums


class FeedbackEngine:
    def __init__(self, llm, booklet_retriever=None):
        self.llm = llm
        self.booklet_retriever = booklet_retriever

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
    # mentor/engines/feedback_engine.py (snippets)
    def evaluate_answer(self, *, student_answer: str, model_answer: str, model: str, temperature: float):
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
        if self.booklet_retriever is not None:
            try:
                # retrieve 15 candidates against student's answer (broad)
                hits = self.booklet_retriever.retrieve(student_answer or "", top_k=15) or []
                ranked = rank_paragraphs_by_text(feedback_text, hits, booklet_retriever=self.booklet_retriever)
                picked = pick_para_nums(ranked, max_n=5)
                if picked:
                    feedback_text = feedback_text.rstrip() + "\n\n---\n" + "_Key paragraphs: " + ", ".join(picked) + "._"
            except Exception:
                pass

        return feedback_text
   
    # -------------------------------------------------------
    # (iii) Follow-up questions about the feedback
    # -------------------------------------------------------
    # mentor/engines/feedback_engine.py (snippets)
from mentor.prompts import build_followup_messages
from mentor.rag.booklet_references_selector import compact_chunks, rank_paragraphs_by_text, pick_para_nums

def follow_up_with_history(self, *, question: str, context: dict, model: str, temperature: float):
    prev_fb = (context or {}).get("feedback","")
    # 1) Retrieve 12–15 booklet paragraphs (anchor on the follow-up question)
    hits = []
    if self.booklet_retriever is not None:
        try:
            hits = self.booklet_retriever.retrieve(question or "", top_k=15) or []
        except Exception:
            hits = []

    # 2) Build prompt WITH booklet chunks
    booklet_chunks = compact_chunks(hits, max_chars=700, max_k=12)
    messages = build_followup_messages(
        previous_feedback=prev_fb,
        followup_question=question,
        booklet_chunks=booklet_chunks,
        max_words=FOLLOWUP_MAX_WORDS,
    )

    # 3) Ask LLM
    raw = self.llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=500)
    reply_text = raw if isinstance(raw, str) else str(raw)

    # 4) Append supporting paras (answer-aware, same selector as ChatEngine)
    if hits and self.booklet_retriever is not None:
        try:
            ranked = rank_paragraphs_by_text(reply_text, hits, booklet_retriever=self.booklet_retriever)
            picked = pick_para_nums(ranked, max_n=5)
            if picked:
                reply_text += "\n\n---\n" + "_Key paragraphs: " + ", ".join(picked) + "._"
        except Exception:
            pass

    return reply_text
