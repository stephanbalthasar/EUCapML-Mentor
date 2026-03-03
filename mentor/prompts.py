FEEDBACK_EVALUATOR_SYSTEM = "You are a strict evaluator. Do not contradict the authoritative model answer."
FEEDBACK_PLANNER_SYSTEM   = "You help plan an answer (topics/anchors/flow). No solutions."
FEEDBACK_EXPLAINER_SYSTEM = "You explain the reasoning behind earlier feedback using the cited excerpts."
CHAT_TUTOR_SYSTEM         = "You are a helpful capital-markets-law tutor. Use booklet/web snippets for grounding."

# mentor/prompts.py

PLAN_MAX_WORDS = 180  # keep tight on purpose

def build_plan_messages(case_text: str,
                        question_label: str,
                        model_answer_slice: str | None = None,
                        booklet_text: str | None = None,
                        max_words: int = PLAN_MAX_WORDS) -> list[dict]:
    """
    Build messages for a compact, issue-first plan. We provide:
      - case_text (facts),
      - question_label (which sub-question),
      - optional model_answer_slice (authoritative compass, not to be disclosed),
      - optional booklet_text (relevant chapter grounding).
    """
    system = (
        "You are a tutor helping a student plan an exam answer in EU/German capital markets law.\n"
        f"Produce a lean, issue‑first outline (6–9 bullets), ≤ {max_words} words. "
        "No citations. No paragraph numbers. No web sources.\n"
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

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
