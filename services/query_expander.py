import json
from typing import List
from utils.helpers import get_logger

logger = get_logger("query_expander")


class QueryExpander:
    """
    Uses the Groq LLM to generate alternative phrasings of a user query.

    These variants are used to run parallel retrieval passes, improving
    recall for queries that may not exactly match indexed document text.
    Gracefully falls back to the original query if the LLM call fails.
    """

    def __init__(self):
        self._client = None
        self._model = None

    def _connect(self) -> None:
        """Lazily initializes the Groq client to avoid startup cost."""
        if self._client is not None:
            return
        from groq import Groq
        from utils.config import GROQ_API_KEY, GROQ_MODEL
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured — cannot expand queries.")
        self._client = Groq(api_key=GROQ_API_KEY)
        self._model = GROQ_MODEL
        logger.info("QueryExpander connected to Groq.")

    def expand(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generates semantically equivalent query variants using Groq JSON mode.

        Args:
            query:        The original user query.
            num_variants: Number of alternative phrasings to generate (default 2).

        Returns:
            A list containing the original query followed by the generated variants.
            Falls back to [query] if the LLM call fails or returns invalid output.
        """
        if not query.strip():
            return [query]

        try:
            self._connect()
        except Exception as e:
            logger.warning(f"QueryExpander could not connect to Groq: {e}. Using original query only.")
            return [query]

        system_prompt = (
            "You are a search query rewriting assistant. "
            "Given a user's search question, generate alternative phrasings that preserve the original intent "
            "but use different wording, synonyms, or sentence structure to improve document retrieval coverage. "
            f"Return exactly {num_variants} alternatives as a JSON object with key 'variants' "
            "containing a list of strings. Do NOT include the original query in your response."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}"},
                ],
                temperature=0.6,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            variants = data.get("variants", [])

            # Validate: must be a non-empty list of strings
            if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
                raise ValueError(f"Unexpected 'variants' format: {variants}")

            # Keep only the requested number of variants (trim if model returned more)
            variants = [v.strip() for v in variants if v.strip()][:num_variants]
            logger.info(f"Query expanded into {len(variants)} variant(s) for: '{query}'")

            # Return original first, then variants
            return [query] + variants

        except Exception as e:
            logger.warning(f"Query expansion failed ({e}). Falling back to original query.")
            return [query]
