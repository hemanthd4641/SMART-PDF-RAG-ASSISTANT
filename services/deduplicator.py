from typing import List, Dict, Any
from utils.helpers import get_logger

logger = get_logger("deduplicator")


def _ngrams(text: str, n: int = 3) -> set:
    """
    Returns the set of character-level n-grams for a given text string.

    Args:
        text: Input text.
        n:    N-gram size (default 3 for trigrams).

    Returns:
        Set of n-gram strings.
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Computes Jaccard similarity between two texts using character-level n-grams.

    Jaccard = |A ∩ B| / |A ∪ B|

    Args:
        text_a: First text string.
        text_b: Second text string.
        n:      N-gram size.

    Returns:
        Similarity score in [0.0, 1.0]. Returns 0.0 if both sets are empty.
    """
    set_a = _ngrams(text_a, n)
    set_b = _ngrams(text_b, n)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
    ngram_size: int = 3,
) -> List[Dict[str, Any]]:
    """
    Removes near-duplicate chunks from a ranked list using Jaccard n-gram similarity.

    Iterates through chunks in ranked order. Each chunk is compared against all
    previously accepted chunks. If similarity to any accepted chunk meets or exceeds
    the threshold, it is considered a near-duplicate and is dropped.

    Args:
        chunks:               Ranked list of retrieved chunk dicts (highest rank first).
        similarity_threshold: Minimum Jaccard similarity to consider a chunk a duplicate.
                              Default 0.85 (85% n-gram overlap).
        ngram_size:           Character n-gram size for similarity computation (default 3).

    Returns:
        Deduplicated list preserving the original rank order of accepted chunks.
    """
    if not chunks:
        return chunks

    accepted: List[Dict[str, Any]] = []
    dropped = 0

    for chunk in chunks:
        candidate_text = chunk.get("chunk_text", "")
        is_duplicate = False

        for accepted_chunk in accepted:
            sim = jaccard_similarity(candidate_text, accepted_chunk.get("chunk_text", ""), n=ngram_size)
            if sim >= similarity_threshold:
                is_duplicate = True
                logger.debug(
                    f"Dropped near-duplicate chunk (similarity={sim:.3f} >= {similarity_threshold}): "
                    f"'{candidate_text[:60]}...'"
                )
                dropped += 1
                break

        if not is_duplicate:
            accepted.append(chunk)

    if dropped > 0:
        logger.info(f"Chunk deduplication: dropped {dropped} near-duplicate(s), retained {len(accepted)} chunk(s).")
    else:
        logger.info("Chunk deduplication: no near-duplicates found.")

    return accepted
