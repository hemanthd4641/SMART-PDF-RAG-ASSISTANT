from typing import List, Dict, Any
from groq import Groq
from utils.config import GROQ_API_KEY, GROQ_MODEL
from utils.helpers import get_logger

logger = get_logger("llm")

def generate_citations_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a rich, deduplicated citation block with document name and page number.
    Citations are rendered as styled markdown badges for visual clarity.

    Args:
        retrieved_chunks: List of context chunks with document_name and page_number.

    Returns:
        Formatted citation block as a markdown string.
    """
    if not retrieved_chunks:
        return ""

    seen = set()
    ordered_citations = []

    for chunk in retrieved_chunks:
        doc_name = chunk.get("document_name", "unknown")
        page_num = chunk.get("page_number", 1)
        citation = (doc_name, page_num)

        if citation not in seen:
            seen.add(citation)
            ordered_citations.append(citation)

    if not ordered_citations:
        return ""

    lines = ["---", "📎 **Sources**", ""]
    for i, (doc_name, page_num) in enumerate(ordered_citations, start=1):
        lines.append(f"{i}. 📄 **{doc_name}** — Page `{page_num}`")

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

    def generate_response(
        self,
        user_question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        """
        Generates a conversational response grounded on retrieved document context.

        Supports multi-turn dialogue by injecting recent conversation history as
        real Groq message turns so the model can reference prior exchanges naturally.

        Args:
            user_question:        The user's current message.
            retrieved_chunks:     List of retrieved context chunks from the RAG pipeline.
            conversation_history: Optional list of recent turns:
                                  [{"user_question": "...", "assistant_answer": "..."}]

        Returns:
            Generated conversational response with source citations appended
            (or a polite refusal if the answer cannot be found in the documents).
        """
        self.connect()
        logger.info(f"Generating conversational response for: '{user_question}'")

        has_context = bool(retrieved_chunks)

        # ── System prompt: conversational + document-grounded ──────────────────
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

            "WHEN INFORMATION IS NOT IN DOCUMENTS:\n"
            "- If the user asks something not found in the documents, respond conversationally: "
            "'I couldn't find that specific information in your uploaded documents. "
            "Could you clarify, or would you like me to look for something related?'\n"
            "- For casual chitchat (greetings, thanks, etc.) respond naturally and warmly.\n"
            "- Never say you 'cannot' do something — always offer an alternative.\n\n"

            "Do NOT generate citations or footnotes — those are added automatically."
        )

        # ── Build the messages list ────────────────────────────────────────────
        messages = [{"role": "system", "content": system_prompt}]

        # Inject recent conversation turns so the model has memory
        if conversation_history:
            for turn in conversation_history:
                prior_q = turn.get("user_question", "").strip()
                prior_a = turn.get("assistant_answer", "").strip()
                if prior_q and prior_a:
                    messages.append({"role": "user",      "content": prior_q})
                    messages.append({"role": "assistant", "content": prior_a})

        # Build the current user message with document context attached
        if has_context:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                text     = chunk.get("chunk_text", "").strip()
                doc_name = chunk.get("document_name", "unknown")
                page_num = chunk.get("page_number", 1)
                if text:
                    context_parts.append(
                        f"[Context Block {i+1} | {doc_name}, Page {page_num}]\n{text}"
                    )
            context_text = "\n\n".join(context_parts)
            user_message = (
                f"Document Context:\n{context_text}\n\n"
                f"User: {user_question}"
            )
        else:
            # No relevant chunks found — still respond conversationally
            user_message = user_question

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.4,   # Slightly warmer for conversational tone
                max_tokens=1500,   # Allow detailed answers
            )

            completion_text = response.choices[0].message.content.strip()
            logger.info("Successfully generated conversational response from Groq.")

            # Append citations only when real document chunks were used
            if has_context:
                citations_block = generate_citations_block(retrieved_chunks)
                if citations_block:
                    return f"{completion_text}\n\n{citations_block}"

            return completion_text

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise RuntimeError(f"Groq API generation failure: {e}")


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
            response = self.client.chat.completions.create(
                model=self.model_name,
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
