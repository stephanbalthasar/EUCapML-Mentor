from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class IssueScore:
    issue: str
    importance: int
    keywords_hit: List[str]
    keywords_missing: List[str]
    score: int
    max_points: int

@dataclass
class FeedbackResult:
    similarity_pct: float
    coverage_pct: float
    final_score: float
    per_issue: List[IssueScore]
    sources: List[str]
    excerpts: List[str]
    narrative: str
    meta: Dict[str, Any]

@dataclass
class PlanOutline:
    sections: List[str]
    topics: List[str]
    legal_anchors: List[str]
    suggested_flow: List[str]
    sources: List[str]

class FeedbackEngine:
    """Three capabilities around a selected case: plan, evaluate, explain."""
    def __init__(self, llm_client, retriever, prompts):
        self.llm = llm_client
        self.retriever = retriever
        self.prompts = prompts

    def plan(self, *, case_data: dict, question_label: str, user_note: Optional[str] = None) -> PlanOutline:
        raise NotImplementedError

    def evaluate(self, *, student_answer: str, case_data: dict) -> FeedbackResult:
        raise NotImplementedError

    def explain(self, *, followup_question: str, prior_feedback: FeedbackResult) -> str:
        raise NotImplementedError
