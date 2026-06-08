import sys
import os

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chunker import DocumentChunker

def test_basic_chunking():
    print("Running basic chunking tests...")
    
    # 1. Prepare dummy parsed document content
    # A text block containing paragraphs, sentences, and layout structures
    long_paragraph_1 = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence "
        "displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. "
        "The term 'artificial intelligence' had previously been used to describe machines that mimic and display human cognitive "
        "skills that are associated with the human mind, such as learning and problem-solving. This definition has since been rejected "
        "by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence "
        "can be spelled out."
    )
    long_paragraph_2 = (
        "As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, "
        "a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, "
        "having become a common technology. Modern machine learning capabilities include search and mathematical optimization, artificial neural networks, "
        "and methods based on statistics, probability and economics. AI also draws upon computer science, information engineering, mathematics, "
        "psychology, linguistics, philosophy, and many other fields."
    )
    
    parsed_doc = {
        "document_name": "AI_Introduction.txt",
        "pages": [
            {
                "page_number": 1,
                "text": f"{long_paragraph_1}\n\n{long_paragraph_2}"
            }
        ]
    }
    
    # Run Chunker with size=500 and overlap=100
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(parsed_doc)
    
    print(f"Generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "No chunks generated"
    
    # Verify outputs
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {c['chunk_id']}")
        print(f"Page: {c['page_number']}")
        print(f"Doc: {c['document_name']}")
        print(f"Length: {len(c['chunk_text'])} chars")
        print(f"Text Preview: {c['chunk_text'][:60]}...")
        
        # Assertions
        assert len(c["chunk_text"]) <= 500, f"Chunk {i+1} exceeds maximum size of 500 characters"
        assert c["document_name"] == "AI_Introduction.txt", "Document name mismatch"
        assert c["page_number"] == 1, "Page number mismatch"
        assert f"chunk_{i}" in c["chunk_id"], "Chunk index sequence suffix mismatch"
        
    # Check overlap property: check if the start of chunk 2 contains text from the end of chunk 1
    if len(chunks) > 1:
        c1_text = chunks[0]["chunk_text"]
        c2_text = chunks[1]["chunk_text"]
        
        # Check if they share overlapping segments
        # Find some common words or substrings near boundary
        print("\nChecking overlap properties between Chunk 1 and Chunk 2...")
        overlap_found = False
        # Let's check if the first 20 characters of chunk 2 are present near the end of chunk 1
        boundary_sample = c2_text[:30]
        if boundary_sample in c1_text:
            overlap_found = True
            print(f"Overlap verified: Substring '{boundary_sample}' is present in both chunks.")
        else:
            # Check for generic substring containment
            for length in range(30, 5, -1):
                sub = c2_text[:length]
                if sub in c1_text:
                    overlap_found = True
                    print(f"Overlap verified (length {length}): Substring '{sub}' is shared.")
                    break
        assert overlap_found, "Chunk overlap was not preserved correctly between adjacent splits"

def test_oversized_text_chunking():
    print("\nRunning oversized boundary tests...")
    
    # Edge case: A single word that is 700 characters long (exceeds size=500)
    giant_word = "A" * 700
    parsed_doc = {
        "document_name": "EdgeCase.txt",
        "pages": [
            {
                "page_number": 1,
                "text": giant_word
            }
        ]
    }
    
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(parsed_doc)
    
    print(f"Oversized word splits count: {len(chunks)}")
    assert len(chunks) >= 2, "Giant word should be split into multiple chunks"
    for c in chunks:
        assert len(c["chunk_text"]) <= 500, "Split chunk exceeds 500 size constraint"
    print("Oversized boundary tests passed!")

if __name__ == "__main__":
    test_basic_chunking()
    test_oversized_text_chunking()
    print("\nAll chunker tests passed successfully!")
