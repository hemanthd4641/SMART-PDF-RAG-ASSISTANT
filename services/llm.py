from typing import List, Dict, Any
from groq import Groq
from utils.config import GROQ_API_KEY, GROQ_MODEL
from utils.helpers import get_logger

logger = get_logger("llm")

def generate_citations_block(
    retrieved_chunks: List[Dict[str, Any]],
    evidence_level: str = "Strong Evidence",
) -> str:
    """
    Generates a rich, deduplicated citation block with Evidence Indicator and page-level sources.

    Args:
        retrieved_chunks: List of context chunks with document_name and page_number.
        evidence_level:   Evidence level category ("Strong Evidence", "Moderate Evidence", "Insufficient Evidence").

    Returns:
        Formatted citation block as a markdown string.
    """
    badge_map = {
        "Strong Evidence": "🟢 Evidence: Strong",
        "Moderate Evidence": "🟡 Evidence: Moderate",
        "Insufficient Evidence": "🔴 Evidence: Insufficient",
    }
    evidence_badge = badge_map.get(evidence_level, f"🟢 Evidence: {evidence_level}")

    if not retrieved_chunks:
        return f"---\n**{evidence_badge}**"

    seen = set()
    ordered_citations = []

    for chunk in retrieved_chunks:
        doc_name = chunk.get("document_name", "unknown")
        page_num = chunk.get("page_number", 1)
        citation = (doc_name, page_num)

        if citation not in seen:
            seen.add(citation)
            ordered_citations.append(citation)

    lines = ["---", f"**{evidence_badge}**", ""]
    if ordered_citations:
        lines.append("**Sources:**")
        for doc_name, page_num in ordered_citations:
            lines.append(f"📄 **{doc_name}** — Page `{page_num}`")

    return "\n".join(lines)


class LLMService:
    """Interacts with the Groq API to generate responses using retrieved context and appending citations."""

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model_name = GROQ_MODEL
        self.client = None

    def connect(self) -> None:
        """Initializes the Groq client connection."""
        if self.client is not None:
            return

        if not self.api_key:
            logger.error("GROQ_API_KEY is missing from configurations.")
            raise ValueError("Groq API Key is not configured.")

        try:
            logger.info("Initializing Groq client...")
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client connected successfully (Model: {self.model_name}).")
        except Exception as e:
            logger.error(f"Failed to connect to Groq: {e}")
            raise RuntimeError(f"Groq client connection failure: {e}")

    def _build_messages(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None,
        is_comparison: bool = False,
        document_scope: List[str] = None,
    ) -> List[Dict[str, str]]:
        """Constructs system and user messages including history and labeled context blocks."""
        has_context = bool(context_chunks)
        doc_names_in_chunks = {c.get("document_name") for c in context_chunks if c.get("document_name")}
        query_lower = query.lower()
        comparison_keywords = ["compare", "comparison", "versus", "vs ", "difference", "differ", "contrast"]
        should_format_comparison = (
            is_comparison 
            or (len(doc_names_in_chunks) >= 2 and any(k in query_lower for k in comparison_keywords))
            or (is_comparison and len(doc_names_in_chunks) >= 2)
        )

        system_prompt = (
            "You are a friendly, knowledgeable, and conversational AI assistant. "
            "You help users understand and explore their uploaded documents through natural dialogue.\n\n"
            "PERSONALITY:\n"
            "- Be warm, clear, and professional — like a smart colleague, not a search engine.\n"
            "- Acknowledge what the user said before answering (e.g. 'Great question!', 'Sure!', 'Absolutely!').\n"
            "- For follow-up questions, reference your previous answer naturally (e.g. 'As I mentioned earlier...', "
            "'Building on that...', 'To add to what we discussed...').\n"
            "- Use conversational phrases and vary your sentence structure.\n"
            "- End responses with a natural follow-up invitation when appropriate "
            "(e.g. 'Would you like more details on any of these?', 'Feel free to ask about anything else!').\n\n"
            "ANSWERING FROM DOCUMENTS:\n"
            "1. When document context is provided, answer ONLY from that context — do not hallucinate.\n"
            "2. Give COMPLETE, DETAILED answers — never one-word replies.\n"
            "3. For lists (roles, products, steps), enumerate ALL items using bullet points.\n"
            "4. For roles/responsibilities, describe the full scope found in the context.\n"
            "5. Use markdown: **bold** key terms, - bullet lists, numbered steps where order matters.\n"
            "6. Synthesize information from multiple context blocks into one cohesive answer.\n"
            "7. NEVER truncate — if there are 10 items, list all 10.\n\n"
        )

        if should_format_comparison:
            system_prompt += (
                "DOCUMENT COMPARISON INSTRUCTIONS:\n"
                "The user is requesting a comparison across documents. You MUST structure your answer with a clear Markdown Comparison Table:\n"
                "| Topic / Feature | [Document A Name] | [Document B Name] |\n"
                "|---|---|---|\n"
                "| [Topic 1] | [Details from Document A] | [Details from Document B] |\n"
                "| [Topic 2] | [Details from Document A] | [Details from Document B] |\n\n"
                "Follow the Markdown table with a brief section explaining the main differences.\n"
                "Ground every single cell strictly on the provided context. If a detail is missing in a document, write 'Not specified'.\n\n"
            )

        system_prompt += (
            "WHEN INFORMATION IS NOT IN DOCUMENTS:\n"
            "- If the user asks something not found in the documents, respond conversationally: "
            "'I couldn't find that specific information in your uploaded documents. "
            "Could you clarify, or would you like me to look for something related?'\n"
            "- For casual chitchat (greetings, thanks, etc.) respond naturally and warmly.\n"
            "- Never say you 'cannot' do something — always offer an alternative.\n\n"
            "Do NOT generate citations or footnotes — those are added automatically."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            for turn in chat_history:
                if "user_question" in turn or "assistant_answer" in turn:
                    prior_q = turn.get("user_question", "").strip()
                    prior_a = turn.get("assistant_answer", "").strip()
                    if prior_q and prior_a:
                        messages.append({"role": "user", "content": prior_q})
                        messages.append({"role": "assistant", "content": prior_a})
                elif turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"].strip()})

        if has_context:
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                text = chunk.get("chunk_text", "").strip()
                doc_name = chunk.get("document_name", "unknown")
                page_num = chunk.get("page_number", 1)
                if text:
                    context_parts.append(
                        f"[Context Block {i+1} | {doc_name}, Page {page_num}]\n{text}"
                    )
            context_text = "\n\n".join(context_parts)
            user_message = f"Document Context:\n{context_text}\n\nUser: {query}"
        else:
            user_message = query

        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_response(
        self,
        user_question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        evidence_level: str = "Strong Evidence",
        is_comparison: bool = False,
    ) -> str:
        """
        Generates a conversational response grounded on retrieved document context.

        Supports multi-turn dialogue and structured document comparison tables.

        Args:
            user_question:        The user's current message.
            retrieved_chunks:     List of retrieved context chunks from the RAG pipeline.
            conversation_history: Optional list of recent turns.
            evidence_level:       Evidence evaluation category from EvidenceGate.
            is_comparison:        If True, enforces Markdown Comparison Table structure.

        Returns:
            Generated conversational response with evidence indicator and source citations appended.
        """
        self.connect()
        logger.info(f"Generating conversational response for: '{user_question}' (is_comparison={is_comparison})")

        has_context = bool(retrieved_chunks)
        messages = self._build_messages(
            query=user_question,
            context_chunks=retrieved_chunks,
            chat_history=conversation_history,
            is_comparison=is_comparison,
        )

        try:
            response = self._call_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=1500,
            )

            completion_text = response.choices[0].message.content.strip()
            logger.info("Successfully generated conversational response from Groq.")

            # Append citations only when real document chunks were used
            if has_context:
                citations_block = generate_citations_block(retrieved_chunks, evidence_level=evidence_level)
                if citations_block:
                    return f"{completion_text}\n\n{citations_block}"

            return completion_text

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise RuntimeError(f"Groq API generation failure: {e}")

    def _call_completion(self, messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 1500, response_format: Dict = None):
        """Calls Groq chat completion with automatic model fallback if model_name is not found."""
        fallback_models = [
            self.model_name,
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        seen = set()
        candidates = [m for m in fallback_models if m and not (m in seen or seen.add(m))]

        last_error = None
        for candidate_model in candidates:
            try:
                kwargs = {
                    "model": candidate_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                return self.client.chat.completions.create(**kwargs)
            except Exception as err:
                err_str = str(err)
                if "404" in err_str or "model_not_found" in err_str or "decommissioned" in err_str:
                    logger.warning(f"Groq model '{candidate_model}' unavailable, trying fallback candidate...")
                    last_error = err
                    continue
                raise err
        raise last_error

    def generate_summary(self, doc_text: str) -> Dict[str, str]:
        """
        Generates a summary and key topics for a document using Groq JSON Mode.
        """
        if not self.api_key:
            logger.warning("GROQ_API_KEY is missing. Returning demo summary.")
            return {
                "summary": "Demo mode: Groq API key is not configured.",
                "key_topics": "Demo, Document, Metadata"
            }
            
        self.connect()
        sample_text = doc_text[:12000] # Limit to first ~12000 characters to avoid token limit errors
        
        system_prompt = (
            "You are an expert document summarization assistant. Your task is to analyze the provided document text "
            "and extract a summary and key topics.\n"
            "You MUST respond in JSON format with exactly two keys: 'summary' and 'key_topics'.\n"
            "Format of response:\n"
            "{\n"
            "  \"summary\": \"Concise 2-3 sentence overview of the document (max 100 words).\",\n"
            "  \"key_topics\": \"Comma-separated list of 3-5 major tags or key topics (e.g. Finance, Contract, Policy).\"\n"
            "}"
        )
        
        try:
            response = self._call_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document Text:\n{sample_text}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            import json
            result_text = response.choices[0].message.content.strip()
            result_dict = json.loads(result_text)
            
            return {
                "summary": result_dict.get("summary", "No summary generated."),
                "key_topics": result_dict.get("key_topics", "General")
            }
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {
                "summary": "Failed to automatically generate summary due to API error.",
                "key_topics": "Error"
            }
