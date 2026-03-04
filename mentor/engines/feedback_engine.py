# mentor/engines/feedback_engine.py
from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
)

class FeedbackEngine:
    def __init__(self, llm):
        self.llm = llm

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
        return self.llm.chat(messages=messages, model=model, temperature=min(temperature, 0.2), max_tokens=750)

    # -------------------------------------------------------
    # (ii) Evaluate a submitted answer
    # -------------------------------------------------------
    def evaluate_answer(self, *, student_answer, model_answer, model, temperature):
        messages = [
            {"role": "system", "content": "You compare the student's answer to the authoritative model answer."},
            {"role": "user", "content":
                f"MODEL ANSWER:\n{model_answer}\n\nSTUDENT ANSWER:\n{student_answer}\n\nGive structured feedback."}
        ]
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=1500
        )

    # -------------------------------------------------------
    # (iii) Follow-up questions about the feedback
    # -------------------------------------------------------
    
    def follow_up_with_history(self, question, context, model, temperature):
        messages = []
    
        # Inject core context
        messages.append({"role": "system", "content": f"Student exam answer:\n{context['student_answer']}"})
        messages.append({"role": "system", "content": f"Feedback:\n{context['feedback']}"})
    
        # Prior chat turns
        for role, msg in context["history"]:
            messages.append({
                "role": "user" if role == "student" else "assistant",
                "content": msg
            })
    
        # Current question
        messages.append({"role": "user", "content": question})
    
        # LLM call
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=800
        )
