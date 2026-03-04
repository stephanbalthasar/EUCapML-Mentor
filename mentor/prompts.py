# mentor/prompts.py
from __future__ import annotations

EVAL_MAX_WORDS = 300
PLAN_MAX_WORDS = 100
FOLLOWUP_MAX_WORDS = 160

def build_evaluate_messages(student_answer: str, model_answer: str,
                            max_words: int = EVAL_MAX_WORDS) -> list[dict]:
    system = (
        "You are an examiner for an EU/German capital markets law exam.\n"
        "Use the MODEL ANSWER as the authoritative benchmark.\n"
        "Do NOT disclose the model answer text. If the student conflicts with it, state the correct "
        "conclusion succinctly and explain briefly why.\n"
        "Do NOT invent legal citations or paragraph numbers. No footnotes. No web sources.\n"
        "Do NOT assign grades, scores, marks etc., provide qualitative feedback only.\n"
        f"Write clearly, ≤ {max_words} words, using the exact five headings below."
    )
    user = (
        "MODEL ANSWER (authoritative, do NOT disclose it):\n"
        f"\"\"\"{(model_answer or '').strip()}\"\"\"\n\n"
        "STUDENT ANSWER:\n"
        f"\"\"\"{(student_answer or '').strip()}\"\"\"\n\n"
        "TASK: Compare the student's reasoning to the MODEL ANSWER and return feedback with the EXACT "
        "headings and format below. Keep it concise and exam‑focused.\n\n"
        "**Student's Core Claims:**\n"
        "• <short bullet per core claim> — [Correct | Incorrect | Unclear]\n"
        "**Mistakes:**\n"
        "• <what is wrong and why> (1–2 lines)\n"
        "**Missing Aspects:**\n"
        "• <material point absent> (why it matters)\n"
        "**Suggestions:**\n"
        "• <how to improve reasoning>\n"
        "**Conclusion:**\n"
        "<one sentence summing up adequacy>\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_consistency_rewrite_messages(feedback_text: str, model_answer: str,
                                       max_words: int = EVAL_MAX_WORDS) -> list[dict]:
    system = (
        "You are an impartial checker. Ensure the FEEDBACK does not contradict the MODEL ANSWER.\n"
        f"If contradictions exist, minimally edit the FEEDBACK to align with the MODEL ANSWER while keeping the same structure and ≤ {max_words} words."
    )
    user = (
        f"MODEL ANSWER (authoritative):\n\"\"\"{(model_answer or '').strip()}\"\"\"\n\n"
        f"FEEDBACK (to check and minimally correct):\n\"\"\"{(feedback_text or '').strip()}\"\"\"\n"
        "Return only the corrected FEEDBACK text (no preface)."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_plan_messages(case_text: str,
                        question_label: str,
                        model_answer_slice: str | None = None,
                        booklet_text: str | None = None,
                        max_words: int = PLAN_MAX_WORDS) -> list[dict]:
    system = (
        "You are a tutor helping a student plan an exam answer in EU/German capital markets law.\n"
        f"Produce a lean, issue‑first outline (6–9 bullets), ≤ {max_words} words. "
        "No citations. No paragraph numbers. No web sources.\n"
        "Do NOT propose conclusions, only list topics to cover."
        "Do NOT disclose or quote the model answer text. If the correct direction differs from the student's likely path, steer it quietly in the plan."      
    )
    blocks = [
        f"CASE DESCRIPTION:\n\"\"\"{(case_text or '').strip()}\"\"\"",
        f"QUESTION: {question_label}",
    ]
    if model_answer_slice and model_answer_slice.strip():
        blocks.append(
            "AUTHORITATIVE COMPASS (do NOT disclose to student; use only to orient the plan):\n"
            f"\"\"\"{model_answer_slice.strip()}\"\"\""
        )
    if booklet_text and booklet_text.strip():
        blocks.append(
            "RELEVANT BOOKLET CHAPTER (use concepts/terms, but no verbatim quotes):\n"
            f"\"\"\"{booklet_text.strip()}\"\"\""
        )
    task = (
        "TASK: Draft a plan the student can follow under time pressure:\n"
        "• Order issues logically (IRAC‑friendly labels: Issue → Rule/Standard → Application → Mini‑conclusion).\n"
        "• Write one short clause per bullet about the expected conclusion.\n"
        "• End with a 1‑line Exam Tip."
    )
    user = "\n\n".join(blocks) + "\n\n" + task
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_followup_messages(previous_feedback: str, followup_question: str,
                            max_words: int = FOLLOWUP_MAX_WORDS) -> list[dict]:
    system = (
        "You answer follow‑up questions about previous feedback. Be precise, "
        f"≤ {max_words} words. If something depends on facts, say what you would check."
    )
    user = (
        f"PREVIOUS FEEDBACK:\n\"\"\"{(previous_feedback or '').strip()}\"\"\"\n\n"
        f"STUDENT'S FOLLOW‑UP QUESTION:\n{(followup_question or '').strip()}\n\n"
        "Answer clearly. If the student asks for the model answer, politely refuse and re‑explain the principle."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]
