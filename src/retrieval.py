"""Vector, BM25, reciprocal-rank-fusion, and cross-encoder retrieval."""

import json
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from .config import (
    BM25_CANDIDATES,
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    RERANK_CANDIDATES,
    VECTOR_CANDIDATES,
)


def tokenize(text: str) -> list[str]:
    """A transparent baseline tokenizer; preserves article numbers like '50'."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_saved_chunks(path: Path = CHUNKS_PATH) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk cache not found: {path}\nRun `python -m src.ingest --rebuild` first."
        )
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
                docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
    if not docs:
        raise RuntimeError(f"No chunks were loaded from {path}.")
    return docs


def doc_key(doc: Document) -> str:
    """Stable identity for deduplicating vector/BM25 hits."""
    if doc.metadata.get("chunk_id"):
        return str(doc.metadata["chunk_id"])
    return "|".join(
        str(doc.metadata.get(field, "")) for field in ("source", "page", "start_index")
    ) + "|" + doc.page_content[:40]


class HybridRetriever:
    def __init__(self, load_reranker: bool = False):
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Chroma index not found: {CHROMA_DIR}\nRun `python -m src.ingest --rebuild` first."
            )

        print(f"Loading embeddings ({EMBEDDING_MODEL})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
        )

        print("Loading cached chunks and preparing BM25...")
        self.chunks = load_saved_chunks()
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.chunks])
        self._reranker = None  # Lazy load: vector/BM25 debugging doesn't need this model.
        if load_reranker:
            self._ensure_reranker()

    def vector_search(self, query: str, k: int = 5) -> list[Document]:
        if not query.strip():
            return []
        return self.vector_store.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int = BM25_CANDIDATES) -> list[Document]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        return [self.chunks[idx] for idx in top_indices]

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: Iterable[list[Document]], rrf_k: int = 60
    ) -> list[tuple[float, Document]]:
        """Fuse result lists using ranks, not incomparable vector/BM25 raw scores."""
        fused_scores: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}
        for results in ranked_lists:
            for rank, doc in enumerate(results, start=1):
                key = doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                doc_by_key[key] = doc
        return sorted(
            ((score, doc_by_key[key]) for key, score in fused_scores.items()),
            key=lambda item: item[0],
            reverse=True,
        )

    def _ensure_reranker(self) -> None:
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoder isn't installed. Run `pip install sentence-transformers`."
            ) from exc
        print(f"Loading reranker ({RERANKER_MODEL}, device={RERANKER_DEVICE})...")
        self._reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)

    def hybrid_search(
        self,
        query: str,
        final_k: int = 5,
        vector_k: int = VECTOR_CANDIDATES,
        bm25_k: int = BM25_CANDIDATES,
        rerank: bool = True,
        candidate_k: int = RERANK_CANDIDATES,
        debug: bool = False,
    ) -> list[Document]:
        """Retrieve with vector + BM25, fuse rankings, optionally rerank candidates, return top-k."""
        if not query.strip():
            return []

        vector_docs = self.vector_search(query, k=vector_k)
        bm25_docs = self.bm25_search(query, k=bm25_k)
        fused = self.reciprocal_rank_fusion([vector_docs, bm25_docs])
        candidates = [doc for _, doc in fused[:candidate_k]]

        if debug:
            print(f"Vector results: {len(vector_docs)} | BM25 results: {len(bm25_docs)}")
            print(f"Unique fused candidates: {len(fused)} | Sending to reranker: {len(candidates)}")

        if rerank and candidates:
            self._ensure_reranker()
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            reranked = sorted(
                zip(scores, candidates), key=lambda item: float(item[0]), reverse=True
            )
            results = [doc for _, doc in reranked[:final_k]]
            if debug:
                print("Top reranker scores:")
                for score, doc in reranked[: min(final_k, 5)]:
                    print(
                        f"  score={float(score):.4f} page={doc.metadata.get('page')} "
                        f"article={doc.metadata.get('article', '?')} "
                        f"chunk_id={doc.metadata.get('chunk_id')}"
                    )
            return results

        results = [doc for _, doc in fused[:final_k]]
        if debug:
            print("Top RRF results:")
            for score, doc in fused[: min(final_k, 5)]:
                print(
                    f"  rrf={score:.4f} page={doc.metadata.get('page')} "
                    f"article={doc.metadata.get('article', '?')} "
                    f"chunk_id={doc.metadata.get('chunk_id')}"
                )
        return results
