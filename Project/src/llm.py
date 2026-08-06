"""
llm.py
-------
Thin wrapper around the Anthropic API (Claude) used for the generative
half of the RAG pipeline: turning retrieved evidence + computed scores
into natural-language, explainable feedback, and for the conversational
Q&A interface.

If ANTHROPIC_API_KEY is not set, falls back to a deterministic template
generator so the rest of the system (scoring, retrieval, UI) is fully
demoable without any API key or network access.
"""
import os

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

MODEL = "claude-sonnet-4-6"


class LLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.enabled = bool(self.api_key) and _ANTHROPIC_AVAILABLE
        if self.enabled:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, system: str, user: str, max_tokens: int = 800) -> str:
        if self.enabled:
            try:
                resp = self.client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
            except Exception as e:
                return self._fallback(user, error=str(e))
        return self._fallback(user)

    def _fallback(self, user: str, error: str = None) -> str:
        note = f"[LLM offline{': ' + error if error else ''} — showing template-based response]\n\n"
        return note + user
