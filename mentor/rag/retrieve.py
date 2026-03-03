from typing import List, Tuple

def retrieve_snippets(query: str, index, *, top_k: int = 8) -> Tuple[List[str], List[str]]:
  """
  Use filters + cosine similarity over prebuilt embeddings.
  Returns (snippets, source_lines). Placeholder for now.
  """
  raise NotImplementedError
