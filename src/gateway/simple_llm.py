# ============================================================
# src/gateway/simple_llm.py - Fast Local LLM (No Ollama needed)
#
# Learning Note:
#   This is a rule-based "LLM" that extracts answers directly
#   from retrieved context. It's NOT a real LLM but demonstrates
#   the full RAG pipeline working instantly with no waiting.
#
#   It works by:
#   1. Finding sentences in the context that match query keywords
#   2. Returning the most relevant sentences as a structured answer
#
#   This is actually how early QA systems worked before LLMs!
#   Real LLMs (GPT, Gemini, Llama) do the same but much better.
# ============================================================

import re
from typing import List


def extract_answer_from_context(question: str, context: str) -> str:
    """
    Extract a relevant answer from context using keyword matching.
    Fast, local, no API key or model download needed.
    """
    if not context or "No relevant information" in context:
        return "No relevant information found in the knowledge base for your question."

    question_lower = question.lower()

    # Extract keywords from question (remove common words)
    stop_words = {"what", "are", "is", "the", "a", "an", "how", "why", "when",
                  "where", "who", "which", "tell", "me", "about", "do", "does",
                  "can", "could", "would", "should", "of", "in", "on", "at",
                  "to", "for", "with", "by", "from", "list", "explain", "describe"}
    keywords = [w for w in re.findall(r'\w+', question_lower) if w not in stop_words and len(w) > 2]

    # Split context into sentences
    # Context format: [Chunk 1 | Source: ... | Relevance: ...]\ntext
    chunks = re.split(r'\[Chunk \d+.*?\]\n', context)
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        return context[:500]

    # Score each sentence by keyword overlap
    all_sentences = []
    for chunk in chunks:
        sentences = re.split(r'[.!\n]+', chunk)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            score = sum(1 for kw in keywords if kw in sent.lower())
            if score > 0:
                all_sentences.append((score, sent))

    # Sort by relevance and build answer
    all_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s[1] for s in all_sentences[:5]]

    if not top_sentences:
        # Fallback: return first 300 chars of first chunk
        return chunks[0][:300] + "..."

    answer = f"Based on the retrieved documents, here is the answer to '{question}':\n\n"
    answer += " ".join(top_sentences)

    return answer
