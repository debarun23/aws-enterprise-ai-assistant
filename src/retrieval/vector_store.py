"""
src/retrieval/vector_store.py
Builds and queries a FAISS index over AWS doc chunks.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any

logger = logging.getLogger(__name__)


class AWSVectorStore:
    """
    Wraps a FAISS flat index with chunk metadata.
    Supports build, save, load, and query operations.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.model = None
        self.index = None
        self.chunks: list[dict] = []
        self.dimension: int = 384  # all-MiniLM-L6-v2 output dim

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)

    def build(self, chunks_file: str) -> None:
        """
        Embed all chunks and build FAISS index.
        Uses IndexFlatIP (inner product) with normalized vectors = cosine similarity.
        """
        self._load_model()

        # Load chunks
        self.chunks = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        logger.info(f"Embedding {len(self.chunks)} chunks...")

        texts = [c["chunk_text"] for c in self.chunks]

        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required for cosine via inner product
        )

        self.dimension = embeddings.shape[1]
        logger.info(f"Embedding shape: {embeddings.shape}")

        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def save(self, index_dir: str) -> None:
        """Save FAISS index and chunk metadata to disk."""
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "chunks_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Vector store saved to {index_dir}")

    def load(self, index_dir: str) -> None:
        """Load FAISS index and chunk metadata from disk."""
        self._load_model()
        path = Path(index_dir)

        self.index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "chunks_meta.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def query(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve top_k most relevant chunks for a question.
        Returns list of dicts with chunk data + similarity score.
        """
        self._load_model()

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results