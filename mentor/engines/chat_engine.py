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
        self.booklet_retriever = booklet_retriever  # Prefer ParagraphRetriever
        self.web_retriever = web_retriever  # may be None for now

    def answer(self, user_query, *, model, temperature, max_tokens=800):
        """
        Generates an answer and then selects 0..5 booklet paragraphs that are
        meaningfully related to the *answer* (not the question).
        Only appends a footer if at least one paragraph clears a minimum similarity.
        """
        import string, math
        import numpy as np
    
        # ---------- local helpers (self-contained) ----------
        def _tok_keep_acronyms(text: str) -> set[str]:
            import string
            if not text:
                return set()
            punct_table = str.maketrans("", "", string.punctuation)
            toks = text.translate(punct_table).split()
            out = set()
            for tok in toks:
                upper_count = sum(1 for ch in tok if ch.isupper())
                if 2 <= len(tok) <= 8 and upper_count >= 2:
                    out.add(tok.lower()); continue
                tl = tok.lower()
                if len(tl) > 3:
                    out.add(tl)
            return out
    
        def _is_meta_answer(text: str) -> bool:
            """Detect greetings/housekeeping answers where we should show no footer."""
            t = (text or "").lower()
            # short & meta-ish phrasing is enough to skip
            short = len(t) < 280
            meta_kw = (
                "hi", "hello", "hallo", "hey",
                "chat", "reden", "sprechen", "talk", "discuss",
                "fragen", "question", "questions",
                "booklet", "broschüre", "heft",
                "let's", "können wir", "kann ich", "kannst du", "gerne", "klar", "sure"
            )
            hits = sum(k in t for k in meta_kw)
            return short and hits >= 2
    
        def _score_hits_against_answer(answer_text: str, hits: list[dict]) -> list[tuple[float, dict]]:
            """
            Returns list of (score, hit_dict), sorted desc.
            Prefers embeddings; otherwise acronym-aware lexical cosine.
            """
            if not answer_text or not hits:
                return []
    
            # 1) Embeddings
            embedder = getattr(self.booklet_retriever, "embedder", None)
            if embedder is not None:
                try:
                    p_texts = [h.get("text", "") for h in hits]
                    P = embedder.encode(p_texts)
                    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
                    a = embedder.encode([answer_text])[0]
                    a = a / (np.linalg.norm(a) + 1e-12)
                    sims = P @ a
                    scored = [(float(sims[i]), {**hits[i], "_sim_mode": "embed"}) for i in range(len(hits))]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return scored
                except Exception:
                    pass  # fall back
    
            # 2) Lexical fallback (binary-cosine)
            import math
            aw = _tok_keep_acronyms(answer_text)
            scored = []
            for h in hits:
                hw = _tok_keep_acronyms(h.get("text", ""))
                overlap = len(aw & hw)
                denom = math.sqrt(max(len(aw), 1) * max(len(hw), 1))
                score = overlap / denom if denom else 0.0
                scored.append((score, {**h, "_sim_mode": "lex"}))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
    
        def _select_supporting_paras(answer_text: str, hits: list[dict], max_n: int = 5) -> list[str]:
            """
            Select 0..5 paragraphs only if there is meaningful similarity to the *answer*.
            - Embedding mode gates: top >= 0.28 AND (top - median) >= 0.08
            - Lexical mode gates:   top >= 0.14 AND (top - median) >= 0.05
            - Also require a minimal 'legal signal' in the answer (MAR/WpHG/etc.).
            """
            ranked = _score_hits_against_answer(answer_text, hits)
            if not ranked:
                return []
    
            # --- gate 1: legal signal in the answer (skip meta/very generic) ---
            LEGAL_TERMS = {
                "mar", "mifid", "mica", "esma", "bafin", "wphg",
                "prospekt", "ad-hoc", "adhoc", "insider", "marktmanipulation",
                "stimmrechte", "major", "holdings", "emittent", "ad-hoc-pflicht",
                "art", "§"  # allow citations like "Art. 17", "§ 33"
            }
            answer_tokens = {w.lower() for w in _tok_keep_acronyms(answer_text)}
            if not (answer_tokens & LEGAL_TERMS):
                return []  # no legal substance -> no footer
    
            # --- gate 2: absolute + relative similarity thresholds ---
            scores = [s for s, _ in ranked]
            if not scores:
                return []
            import numpy as _np
            top, med = scores[0], float(_np.median(scores))
            mode = ranked[0][1].get("_sim_mode", "lex")
            min_abs = 0.28 if mode == "embed" else 0.14
            min_gap = 0.08 if mode == "embed" else 0.05
            if top < min_abs or (top - med) < min_gap:
                return []  # weak/flat distribution -> no footer
    
            # --- take up to 5 distinct para_num entries ---
            selected, seen = [], set()
            for score, h in ranked:
                # apply per-item absolute floor as well
                threshold = 0.20 if mode == "embed" else 0.10
                if score < threshold:
                    continue
                pnum = h.get("para_num")
                if pnum is None or pnum in seen:
                    continue
                seen.add(pnum)
                selected.append(str(pnum))
                if len(selected) == max_n:
                    break
            return selected
        # ------------------------------------------------------------------------
        
      
        # 1) Simple keyword extraction (existing logic)
        keywords = self._extract_keywords(user_query)
    
        # 2) Retrieve booklet context (prefer ParagraphRetriever)
        hits = []
        try:
            if hasattr(self.booklet_retriever, "retrieve"):
                hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
            elif hasattr(self.booklet_retriever, "retrieve_best"):
                # Back-compat: ChapterRetriever branch
                chapter = self.booklet_retriever.retrieve_best(user_query)
                if chapter and isinstance(chapter, dict):
                    from mentor.rag.booklet_retriever import ParagraphRetriever  # your current path
                    chapter_num = chapter.get("chapter_num")
                    chapter_paras = [
                        p for p in (self.booklet_index.get("paragraphs") or [])
                        if p.get("chapter_num") == chapter_num
                    ]
                    if chapter_paras:
                        pr = ParagraphRetriever(chapter_paras)
                        hits = pr.retrieve(user_query, top_k=15) or []
                    else:
                        text = chapter.get("text", "")
                        paras = [t.strip() for t in text.split("\n\n") if t.strip()]
                        hits = [{"text": t} for t in paras[:15]]
            else:
                # Generic fallback (avoid passing unsupported kwargs)
                try:
                    hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
                except TypeError:
                    hits = self.booklet_retriever.retrieve(query=user_query, top_k=15) or []
        except Exception:
            hits = []
    
        # Normalize to list[str] for prompt
        booklet_chunks = [
            (h.get("text") if isinstance(h, dict) else str(h))
            for h in hits if h
        ]
    
        # 3) Optional web retrieval (kept off unless web_retriever is set)
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
    
        # 4) Build messages for LLM
        messages = self._build_prompt(
            user_query=user_query,
            booklet_chunks=booklet_chunks,
            web_snippets=web_snippets
        )
    
        # 5) Ask LLM
        result = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
        # Coerce to plain string for display
        reply_text = result if isinstance(result, str) else str(result)
        if _is_meta_answer(reply_text):
            return reply_text  # early exit: greetings/housekeeping -> no footer
    
        # 6) NEW: pick 0..5 supporting paragraph numbers based on the *answer*
        selected_para_nums = _select_supporting_paras(reply_text, hits, max_n=5)
    
        # Append footer only if we actually have meaningful support
        if selected_para_nums:
            footer = "\n\n---\n" + "_Key paragraphs: " + ", ".join(selected_para_nums) + "._"
            reply_text += footer
    
        return reply_text

    # -------- helpers (kept) --------------------
    def _extract_keywords(self, text):
        # TODO: later include legal keyword extraction
        # For now: split by spaces and take simple tokens
        return [w.strip() for w in text.split() if len(w) > 3]

    def _build_prompt(self, user_query, booklet_chunks, web_snippets):
        system = (
            "You are a helpful EU/German capital markets law tutor. "
            "Use the provided booklet excerpts and optional web snippets. "
            "If unsure, say what is known, and avoid fabricating structural references."
        )
        booklet_block = "\n\n".join(f"- {c}" for c in booklet_chunks[:15]) or "None"
        web_block = "\n\n".join(f"- {s}" for s in web_snippets[:4]) or "None"

        user_content = (
            f"USER QUERY:\n{user_query}\n\n"
            f"RELEVANT BOOKLET EXCERPTS:\n{booklet_block}\n\n"
            f"RELEVANT WEB SNIPPETS:\n{web_block}\n\n"
            "Please answer clearly and concisely."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
