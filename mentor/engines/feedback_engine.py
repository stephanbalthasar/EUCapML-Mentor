# mentor/engines/feedback_engine.py

class FeedbackEngine:
    def __init__(self, llm):
        self.llm = llm

    # -------------------------------------------------------
    # (i) Help student plan an answer
    # -------------------------------------------------------
    def plan_answer(self, *, case_text, question, model, temperature):
        messages = [
            {"role": "system", "content": "You help students plan structured exam answers."},
            {"role": "user", "content": f"CASE:\n{case_text}\n\nQUESTION:\n{question}\n\nHelp me plan my answer."}
        ]
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=800
        )

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
    def follow_up(self, *, question, previous_feedback, model, temperature):
        messages = [
            {"role": "system", "content": "You answer follow-up questions about previous feedback."},
            {"role": "user", "content":
                f"STUDENT QUESTION:\n{question}\n\nYOUR PREVIOUS FEEDBACK:\n{previous_feedback}"}
        ]
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=600
        )
