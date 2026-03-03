class ChatEngine:
    """RAG-augmented legal tutor (booklet + optional web). No model answers."""
    def __init__(self, llm_client, retriever, prompts):
        self.llm = llm_client
        self.retriever = retriever
        self.prompts = prompts

    def answer(self, *, question: str) -> str:
        raise NotImplementedError
