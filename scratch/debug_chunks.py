import sys
sys.path.insert(0, '.')
from services.parser import parse_document
from services.chunker import DocumentChunker

parsed = parse_document('data/test_company.pdf')
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(parsed)
print("Total chunks:", len(chunks))
for i, c in enumerate(chunks):
    print("--- CHUNK", i+1, "[", c["chunk_type"], "] Page", c["page_number"], "---")
    print(c["chunk_text"])
    print()
