"""Inspect retrieved passages without loading the 8B generation model."""

import sys

from .config import FINAL_K
from .retrieval import HybridRetriever


def show_results(label, docs):
    print(f"\n{'=' * 18} {label} ({len(docs)} results) {'=' * 18}")
    for rank, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(
            f"\n#{rank} | page={meta.get('page')} | article={meta.get('article', '?')} "
            f"| chunk_id={meta.get('chunk_id')}"
        )
        print(doc.page_content[:650].replace("\n", " "))


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Question to retrieve: ").strip()
    if not query:
        raise SystemExit("Enter a non-empty question.")

    retriever = HybridRetriever(load_reranker=False)
    show_results("VECTOR ONLY", retriever.vector_search(query, k=FINAL_K))
    show_results("BM25 ONLY", retriever.bm25_search(query, k=FINAL_K))
    show_results(
        "HYBRID (RRF; no reranker)",
        retriever.hybrid_search(query, final_k=FINAL_K, rerank=False, debug=True),
    )
    # Uncomment to compare reranking after the first three retrieval modes work:
    # show_results("HYBRID + RERANKER", retriever.hybrid_search(query, final_k=FINAL_K, rerank=True))


if __name__ == "__main__":
    main()
