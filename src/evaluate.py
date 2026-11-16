"""Compare vector-only, hybrid RRF, and hybrid+reranker retrieval on labelled questions.

This starter measures article-level Recall@5, using `relevant_articles` labels in data/eval_questions.json.
For a production-quality CV metric, annotate gold evidence chunk IDs and evaluate chunk-level recall too.
"""

import json
from pathlib import Path

from .config import EVAL_PATH, FINAL_K
from .retrieval import HybridRetriever


def article_recall_at_k(docs, relevant_articles: set[str]) -> float:
    if not relevant_articles:
        return 0.0
    found = {
        str(doc.metadata["article"])
        for doc in docs
        if doc.metadata.get("article") is not None
    }
    return len(found & relevant_articles) / len(relevant_articles)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Evaluation set not found: {EVAL_PATH}")
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if not questions:
        raise RuntimeError("Evaluation JSON contains no questions.")

    retriever = HybridRetriever(load_reranker=False)
    per_mode = {"vector_only": [], "hybrid_rrf": [], "hybrid_reranked": []}

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        gold = {str(article) for article in item.get("relevant_articles", [])}
        if not gold:
            print(f"Skipping Q{number}: no relevant_articles labels")
            continue

        vector_docs = retriever.vector_search(question, k=FINAL_K)
        hybrid_docs = retriever.hybrid_search(question, final_k=FINAL_K, rerank=False)
        reranked_docs = retriever.hybrid_search(question, final_k=FINAL_K, rerank=True)

        scores = {
            "vector_only": article_recall_at_k(vector_docs, gold),
            "hybrid_rrf": article_recall_at_k(hybrid_docs, gold),
            "hybrid_reranked": article_recall_at_k(reranked_docs, gold),
        }
        for mode, score in scores.items():
            per_mode[mode].append(score)
        print(
            f"Q{number:02d} | gold articles={sorted(gold)} | "
            + " | ".join(f"{mode}={score:.0%}" for mode, score in scores.items())
        )

    print("\nArticle-level Recall@5 (macro-average across questions)")
    for mode, values in per_mode.items():
        print(f"  {mode:18s}: {mean(values):.1%}")
    print(
        "\nNote: these are article-label metrics on this small starter set, not a verified benchmark. "
        "Do not report a 72%→81% improvement unless your own labelled evaluation produces it."
    )


if __name__ == "__main__":
    main()
