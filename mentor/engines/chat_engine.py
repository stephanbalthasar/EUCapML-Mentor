# mentor/engines/chat_engine.py

class ChatEngine:
    """
    One-purpose engine:
    - Take user query
    - Find relevant booklet parts
    - Optionally find relevant web snippets
    - Ask LLM to answer grounded in those
    """

    def __init__(self, llm, booklet_index, booklet_retriever, web_retriever=None):
        self.llm = llm
        self.booklet_index = booklet_index
        self.booklet_retriever = booklet_retriever
        self.web_retriever = web_retriever  # may be None for now

    def answer(self, user_query, *, model, temperature, max_tokens=800):
    # 1) Extract keywords from query (stub for now)
    keywords = self._extract_keywords(user_query)

    # 2) Retrieve booklet context
    booklet_chunks: list[str] = []
    chapter_title = None

    # If we were given a ChapterRetriever, use retrieve_best → single chapter
    if hasattr(self.booklet_retriever, "retrieve_best"):
        chapter = self.booklet_retriever.retrieve_best(user_query)
        if chapter and isinstance(chapter, dict) and "text" in chapter:
            text = chapter["text"]
            # Light truncation to leave space for the answer
            booklet_chunks = [text if len(text) <= 6000 else text[:6000] + " …"]
            chapter_title = chapter.get("title") or f"Chapter {chapter.get('chapter_num', '—')}"
        else:
            booklet_chunks = []
    else:
        # Fallback: a paragraph‑style retriever with a generic .retrieve()
        hits = self.booklet_retriever.retrieve(
            query=user_query,
            keywords=keywords,
            top_k=6
        ) or []
        # Normalise to a list of strings
        booklet_chunks = [
            (h.get("text") if isinstance(h, dict) else str(h)) for h in hits if h
        ]

    # 3) Optionally retrieve web snippets
    web_snippets = []
    if self.web_retriever is not None:
        web_snippets = self.web_retriever.retrieve(
            query=user_query,
            keywords=keywords,
            top_k=4
        )

    # 4) Build messages for LLM (now passes chapter_title)
    messages = self._build_prompt(
        user_query=user_query,
        booklet_chunks=booklet_chunks,
        web_snippets=web_snippets,
        chapter_title=chapter_title,
    )

    # 5) Ask LLM
    return self.llm.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # -------- helpers (placeholders for now) --------------------

    def _extract_keywords(self, text):
        # TODO: later include legal keyword extraction
        # For now: split by spaces and take simple tokens
        return [w.strip() for w in text.split() if len(w) > 3]

    def _build_prompt(self, user_query, booklet_chunks, web_snippets, chapter_title=None):
    """
    Prefer prompts from mentor.prompts if available; otherwise fall back to the existing template.
    """
    # Try to import the booklet‑only tutor prompt builder
    try:
        from mentor.prompts import build_tutor_chat_prompt_booklet_only  # type: ignore
    except Exception:
        build_tutor_chat_prompt_booklet_only = None

    # If a builder is available AND we have booklet context, use it (booklet‑only, no web)
    if build_tutor_chat_prompt_booklet_only and booklet_chunks:
        excerpt = ("\n\n---\n\n").join(booklet_chunks[:3])[:3200]
        prompt_str = build_tutor_chat_prompt_booklet_only(
            booklet_excerpt=excerpt,
            user_question=user_query,
            chapter_title=chapter_title,
        )
        # Wrap as a single-user message; builder already includes the guardrails text
        return [{"role": "user", "content": prompt_str}]

    # Otherwise: keep your existing prompt structure (booklet + optional web)
    system = (
        "You are a helpful EU/German capital markets law tutor. "
        "Use the provided booklet excerpts and optional web snippets. "
        "If unsure, say what is known, and avoid fabricating structural references."
    )
    booklet_block = "\n\n".join(f"- {c}" for c in booklet_chunks[:6]) or "None"
    web_block = "\n\n".join(f"- {s}" for s in web_snippets[:4]) or "None"
    user_content = (
        f"USER QUERY:\n{user_query}\n\n"
        f"RELEVANT BOOKLET EXCERPTS:\n{booklet_block}\n\n"
        f"RELEVANT WEB SNIPPETS:\n{web_block}\n\n"
        "Please answer clearly and concisely."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]    
