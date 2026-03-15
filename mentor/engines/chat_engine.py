from mentor.prompts import build_tutor_messages

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
        self.booklet_retriever = booklet_retriever  # Prefer ParagraphRetriever
        self.web_retriever = web_retriever  # may be None for now

    def answer(self, user_query, *, model, temperature, max_tokens=800):
        """
        One-purpose engine:
        - Take user query
        - Find relevant booklet parts (prefer new BookletRetriever.search)
        - Optionally find relevant web snippets
        - Ask LLM to answer grounded in those
        """
        import os
    
        # ---- Retrieval configuration (passed into retriever; no re-filtering here) ----
        top_k = int(os.getenv("BOOKLET_TOP_K", "6"))
    
        # 1) Simple keyword extraction (kept)
        keywords = self._extract_keywords(user_query)
    
        # 2) Retrieve booklet context
        hits = []
        try:
            if hasattr(self.booklet_retriever, "search"):
                # New hybrid retriever path (BM25 + embeddings); threshold enforced inside retriever
                hits = self.booklet_retriever.search(
                    query=user_query,
                    top_k=top_k,                    
                ) or []
            elif hasattr(self.booklet_retriever, "retrieve"):
                # Legacy ParagraphRetriever-style API
                hits = self.booklet_retriever.retrieve(user_query, top_k=top_k) or []
            elif hasattr(self.booklet_retriever, "retrieve_best"):
                # Back-compat: ChapterRetriever branch (from your existing code)
                chapter = self.booklet_retriever.retrieve_best(user_query)
                if chapter and isinstance(chapter, dict):
                    from mentor.rag.booklet_retriever import ParagraphRetriever
                    chapter_num = chapter.get("chapter_num")
                    chapter_paras = [
                        p for p in (self.booklet_index.get("paragraphs") or [])
                        if p.get("chapter_num") == chapter_num
                    ]
                    if chapter_paras:
                        pr = ParagraphRetriever(chapter_paras)
                        hits = pr.retrieve(user_query, top_k=top_k) or []
                    else:
                        text = chapter.get("text", "")
                        paras = [t.strip() for t in text.split("\n\n") if t.strip()]
                        hits = [{"text": t} for t in paras[:top_k]]
            else:
                # Generic fallback (avoid passing unsupported kwargs)
                try:
                    hits = self.booklet_retriever.retrieve(user_query, top_k=top_k) or []
                except TypeError:
                    hits = self.booklet_retriever.retrieve(query=user_query, top_k=top_k) or []
        except Exception:
            hits = []
    
        # Normalize to list[str] for prompt (kept)
        booklet_chunks = [
            (h.get("text") if isinstance(h, dict) else str(h))
            for h in hits if h
        ]
    
        # --- Keep collecting paragraph numbers if present (unchanged) ---
        para_nums = []
        for h in hits:
            if isinstance(h, dict) and "para_num" in h and h["para_num"] is not None:
                para_nums.append(str(h["para_num"]))
        # keep order, remove duplicates
        seen = set()
        para_nums = [p for p in para_nums if not (p in seen or seen.add(p))]
    
        # 3) Optional web retrieval (kept)
        web_snippets = []
        if self.web_retriever is not None:
            try:
                web_snippets = self.web_retriever.retrieve(
                    query=user_query,
                    keywords=keywords,
                    top_k=4
                ) or []
            except Exception:
                web_snippets = []
    
        # 4) Build messages for LLM (kept)
        messages = build_tutor_messages(
            user_query=user_query,
            booklet_chunks=booklet_chunks,
            web_snippets=web_snippets
        )
    
        # 5) Ask LLM (kept)
        result = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
        # --- Append deterministic footer with paragraph numbers (kept) ---
        reply_text = result if isinstance(result, str) else str(result)
        if para_nums:
            footer = (
                "\n\n---\n"
                "_Also see paragraphs " + ", ".join(para_nums[:top_k]) + " in the course booklet._"
            )
            reply_text += footer
    
        return reply_text


    # -------- helpers (kept) --------------------
    def _extract_keywords(self, text):
        # TODO: later include legal keyword extraction
        # For now: split by spaces and take simple tokens
        return [w.strip() for w in text.split() if len(w) > 3]
