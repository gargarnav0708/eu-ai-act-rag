"""Extract the PDF, split into article-aware recursive chunks, and build persistent Chroma."""

import argparse
import json
import re
import shutil
from pathlib import Path

import PyMuPDF as fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    PDF_PATH,
    STORAGE_DIR,
)

# Article headings are standalone lines. This avoids matching normal prose like "Article 5".
ARTICLE_HEADING_RE = re.compile(r"(?im)^[ \t]*Article[ \t]+(\d+[a-z]?)[ \t]*$")
ANNEX_HEADING_RE = re.compile(r"(?im)^[ \t]*ANNEX[ \t]+[IVXLCDM0-9]+\b")


def clean_page_text(text: str) -> str:
    """Light cleanup only; preserve legal wording, numbering, and meaningful line breaks."""
    text = text.replace("\u00ad", "")  # soft hyphen sometimes emitted from PDF line wraps
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[Document]:
    """Create one document per physical PDF page, preserving 1-based page references."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\nPut EU_AI_Act.pdf in the data/ folder or set EU_AI_ACT_PDF."
        )

    pages: list[Document] = []
    with fitz.open(pdf_path) as pdf:
        page_count = len(pdf)
        print(f"[1/4] Opened PDF: {pdf_path.name} ({page_count} pages)")
        for page_index, page in enumerate(pdf):
            text = clean_page_text(page.get_text("text", sort=True))
            if not text:
                print(f"  WARNING: page {page_index + 1} has no extractable text")
                continue
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_index + 1},
                )
            )

    if not pages:
        raise RuntimeError("No text could be extracted. Check whether the PDF is scanned or corrupted.")

    total_chars = sum(len(doc.page_content) for doc in pages)
    print(f"      Extracted {len(pages)} non-empty pages and {total_chars:,} characters")
    print(f"      First-page preview: {pages[0].page_content[:180]!r}")
    return pages


def split_pages_by_article(pages: list[Document]) -> list[Document]:
    """Split physical pages at Article boundaries before recursive chunking.

    This prevents one chunk from mixing (for example) the end of Article 4 with Article 5,
    making article-level retrieval diagnostics more meaningful. Annex content is kept separate
    from the numbered Articles; standalone cross-references inside Annexes are not treated as
    new Article headings.
    """
    sections: list[Document] = []
    current_article = ""
    in_annex = False

    for page_doc in pages:
        text = page_doc.page_content
        page_num = int(page_doc.metadata["page"])
        if ANNEX_HEADING_RE.search(text[:1000]):
            in_annex = True
            current_article = ""

        # Once the document reaches its Annexes, don't interpret references like "Article 49"
        # in an Annex heading as the start of a new Article.
        headings = [] if in_annex else list(ARTICLE_HEADING_RE.finditer(text))
        page_sections: list[tuple[str, str]] = []  # (section text, article label)

        if not headings:
            page_sections.append((text, current_article if not in_annex else ""))
        else:
            prefix = text[:headings[0].start()].strip()
            if prefix:
                page_sections.append((prefix, current_article))

            for i, match in enumerate(headings):
                start = match.start()
                end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
                section_text = text[start:end].strip()
                if section_text:
                    article = match.group(1)
                    page_sections.append((section_text, article))
                    current_article = article

        for section_index, (section_text, article) in enumerate(page_sections):
            metadata = {
                "source": page_doc.metadata["source"],
                "page": page_num,
                "segment_index": section_index,
            }
            if article:
                metadata["article"] = str(article)
            sections.append(Document(page_content=section_text, metadata=metadata))

    print(f"      Split PDF pages into {len(sections)} page/article sections")
    return sections


def save_chunks_jsonl(chunks: list[Document], destination: Path) -> None:
    """Save inspectable chunk text and metadata; BM25 uses this same source of truth."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for doc in chunks:
            file.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"      Saved chunk text/metadata to {destination}")


def build_index(rebuild: bool = False) -> None:
    pages = extract_pdf_pages(PDF_PATH)
    sections = split_pages_by_article(pages)

    # Split long sections recursively at paragraphs, lines, sentences, words, then characters.
    # Page and article metadata is inherited by each child chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(sections)
    for chunk in chunks:
        page_num = int(chunk.metadata["page"])
        segment_index = int(chunk.metadata.get("segment_index", 0))
        start_index = int(chunk.metadata.get("start_index", 0) or 0)
        chunk.metadata["chunk_id"] = f"p{page_num:03d}_seg{segment_index:02d}_s{start_index:06d}"

    if not chunks:
        raise RuntimeError("Chunking produced zero chunks; inspect the PDF extraction output.")
    print(f"[2/4] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"      Sample chunk ID: {chunks[0].metadata['chunk_id']}")
    print(f"      Sample chunk starts: {chunks[0].page_content[:150]!r}")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    has_existing_index = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    if has_existing_index and not rebuild:
        raise FileExistsError(
            f"A Chroma index already exists at {CHROMA_DIR}.\n"
            "To avoid accidental duplicate/stale data, rerun with --rebuild to replace it."
        )
    if rebuild and CHROMA_DIR.exists():
        print(f"      Removing old Chroma index: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    save_chunks_jsonl(chunks, CHUNKS_PATH)

    print(f"[3/4] Loading embedding model: {EMBEDDING_MODEL} (device={EMBEDDING_DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[4/4] Writing vectors to ChromaDB...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    vector_store.add_documents(
        documents=chunks,
        ids=[str(doc.metadata["chunk_id"]) for doc in chunks],
    )

    # Smoke test the persistent index before declaring success.
    test_results = vector_store.similarity_search("What is AI literacy?", k=2)
    if not test_results:
        raise RuntimeError("Chroma was written but returned zero results for the smoke test.")
    print("Index build complete.")
    for result in test_results:
        print(
            f"  smoke-test result: page={result.metadata.get('page')} "
            f"article={result.metadata.get('article', 'Annex/preamble')} "
            f"chunk_id={result.metadata.get('chunk_id')}"
        )
    print("Next: python -m src.inspect_retrieval \"What does Article 4 say about AI literacy?\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Replace an existing Chroma index instead of refusing to overwrite it.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
