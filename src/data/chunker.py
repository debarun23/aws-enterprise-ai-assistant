"""
src/data/chunker.py — Split raw docs into overlapping text chunks.
Each chunk becomes one retrieval unit in FAISS and one training context.
"""

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Split text into word-level overlapping chunks.
    Word-level (not char-level) avoids cutting mid-word.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def process_raw_docs(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_length: int = 100,
) -> list[dict]:
    """
    Read all per-service JSONL files, chunk each doc, save combined output.
    Each output record has: chunk_id, service, url, title, chunk_text.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    chunk_id = 0

    jsonl_files = list(raw_path.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} service files in {raw_dir}")

    for jsonl_file in sorted(jsonl_files):
        service = jsonl_file.stem
        doc_count = 0
        chunk_count = 0

        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                chunks = chunk_text(
                    doc["text"],
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                )

                for chunk in chunks:
                    # Skip chunks that are too short to be useful
                    if len(chunk.split()) < min_chunk_length:
                        continue

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "service": service,
                        "url": doc["url"],
                        "title": doc["title"],
                        "chunk_text": chunk,
                        "word_count": len(chunk.split()),
                    })
                    chunk_id += 1
                    chunk_count += 1

                doc_count += 1

        logger.info(
            f"  {service}: {doc_count} docs → {chunk_count} chunks"
        )

    # Save combined chunks
    output_file = proc_path / "chunks.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")
    return all_chunks