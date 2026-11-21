"""
Part A - Task 5: Retrieval Improvement via Semantic Chunking & Metadata

This script demonstrates and evaluates an improvement to the baseline RAG retrieval.
Baseline: Stores each entire .txt file as a single monolithic document.
Improvement: Chunks documents by semantic sections (headers/paragraphs) and attaches
rich metadata (source filename, section heading, chunk index, character length).
"""

import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

os.environ["TQDM_DISABLE"] = "1"

DOCUMENTS_PATH = Path(__file__).parent.parent / "data" / "documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def chunk_document(filename: str, content: str) -> list[dict]:
    """Split a document into section-based chunks with metadata."""
    lines = content.strip().split("\n")
    chunks = []
    current_title = "Introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        # Check if line looks like a section heading (all uppercase or specific patterns)
        if stripped.isupper() and len(stripped) > 2 and not stripped.startswith("$"):
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append({
                        "section": current_title,
                        "text": chunk_text,
                    })
                current_lines = []
            current_title = stripped
        else:
            current_lines.append(line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append({
                "section": current_title,
                "text": chunk_text,
            })

    # Wrap into chunk records
    records = []
    for idx, c in enumerate(chunks):
        records.append({
            "id": f"{filename}#chunk_{idx}",
            "filename": filename,
            "section": c["section"],
            "text": c["text"],
            "length": len(c["text"]),
        })
    return records


def setup_collections():
    client = chromadb.Client()
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # 1. Baseline Collection (Entire files)
    baseline_col = client.create_collection("baseline", embedding_function=embedding_fn)
    files = []
    for f in sorted(DOCUMENTS_PATH.glob("*.txt")):
        content = f.read_text(encoding="utf-8")
        files.append((f.name, content))

    baseline_col.add(
        ids=[f[0] for f in files],
        documents=[DOCUMENT_PREFIX + f[1] for f in files],
        metadatas=[{"source": f[0], "granularity": "full_file"} for f in files],
    )

    # 2. Improved Chunked Collection
    chunked_col = client.create_collection("chunked", embedding_function=embedding_fn)
    all_chunks = []
    for fname, content in files:
        chunks = chunk_document(fname, content)
        all_chunks.extend(chunks)

    chunked_col.add(
        ids=[c["id"] for c in all_chunks],
        documents=[DOCUMENT_PREFIX + c["section"] + "\n" + c["text"] for c in all_chunks],
        metadatas=[{
            "source": c["filename"],
            "section": c["section"],
            "char_length": c["length"],
            "granularity": "chunk"
        } for c in all_chunks],
    )

    return baseline_col, chunked_col, len(files), len(all_chunks)


def compare_retrieval(baseline_col, chunked_col, query: str):
    print("=" * 80)
    print(f"QUERY: \"{query}\"")
    print("=" * 80)

    # Query Baseline
    base_res = baseline_col.query(query_texts=[QUERY_PREFIX + query], n_results=1)
    base_id = base_res["ids"][0][0]
    base_dist = base_res["distances"][0][0]
    base_doc = base_res["documents"][0][0]

    print(f"\n[BASELINE: Monolithic Document Retrieval]")
    print(f"  Matched ID:       {base_id}")
    print(f"  Vector Distance:  {base_dist:.4f}")
    print(f"  Payload Size:     {len(base_doc)} characters")
    print(f"  Preview (first 180 chars):\n    {base_doc[:180].replace(chr(10), ' ')}...")

    # Query Chunked
    chunk_res = chunked_col.query(query_texts=[QUERY_PREFIX + query], n_results=1)
    chunk_id = chunk_res["ids"][0][0]
    chunk_dist = chunk_res["distances"][0][0]
    chunk_meta = chunk_res["metadatas"][0][0]
    chunk_doc = chunk_res["documents"][0][0]

    print(f"\n[IMPROVEMENT: Section Chunking with Metadata]")
    print(f"  Matched ID:       {chunk_id}")
    print(f"  Source File:      {chunk_meta['source']}")
    print(f"  Section Header:   {chunk_meta['section']}")
    print(f"  Vector Distance:  {chunk_dist:.4f}")
    print(f"  Payload Size:     {len(chunk_doc)} characters")
    print(f"  Pinpointed Text:\n    {chunk_doc.strip().replace(chr(10), chr(10) + '    ')}")
    print()


def main():
    print("Initializing collections for comparison...")
    base_col, chunk_col, n_files, n_chunks = setup_collections()
    print(f"Indexed {n_files} monolithic files into baseline.")
    print(f"Indexed {n_chunks} semantic chunks into improved collection.\n")

    test_queries = [
        "Can I bring pets or animals?",
        "Are there power outlets for laptops?",
        "Do you have peanut allergies or use peanuts?",
        "What time do you close on Sunday?",
    ]

    for q in test_queries:
        compare_retrieval(base_col, chunk_col, q)


if __name__ == "__main__":
    main()
