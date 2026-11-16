# EU AI Act RAG Assistant

A modular, inspectable retrieval-augmented generation (RAG) pipeline for exploring Regulation (EU) 2024/1689, the EU Artificial Intelligence Act. It extracts the supplied regulation PDF, creates article-aware chunks, retrieves supporting passages with both semantic and keyword search, and can use a locally hosted language model to draft cited answers.

> **Important:** This is an educational information-retrieval project, not legal advice or a compliance determination. Generated answers can be incomplete or incorrect. Check the cited regulation text and consult a qualified legal professional for consequential decisions.

## What it does

- Extracts text from the PDF with PyMuPDF and tracks physical PDF pages.
- Splits text at standalone Article headings before recursive chunking, retaining article and page metadata.
- Embeds chunks with a Hugging Face sentence-transformer model and stores vectors in a persistent Chroma database.
- Combines dense vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- Optionally reranks fused candidates with a cross-encoder.
- Keeps retrieval diagnostics available independently of answer generation.
- Uses a grounded prompt that asks the language model to cite retrieved source labels and abstain when evidence is insufficient.
- Includes a small, manually labelled article-level retrieval evaluation set.

## How the pipeline works

```text
data/EU_AI_Act.pdf
       |
       v
PyMuPDF extraction -> page/article sections -> recursive chunks
                                               |             |
                                               v             v
                                      Chroma vectors     chunks.jsonl
                                               |             |
                                      dense search       BM25 search
                                               \             /
                                                RRF fusion
                                                    |
                                          optional reranking
                                                    |
                                      source-labelled passages
                                                    |
                                      optional Llama 3 answer
```

Ingestion and retrieval do not require the Llama model. Build the index and inspect retrieval first; the larger gated model is loaded only when you start the Q&A command.

## Requirements

- Python 3.11 or 3.12 recommended.
- Dependencies listed in `requirements.txt`.
- Internet access on first use to download embedding and reranker models from Hugging Face.
- A compatible NVIDIA CUDA environment for the included 4-bit Meta Llama 3 loader. A Mac or CPU-only machine can still run ingestion and retrieval by keeping the device settings on `cpu`.
- Access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face if you want to use answer generation.

The 4-bit `bitsandbytes` Llama path is specifically configured for CUDA; it is not an Apple Silicon/MPS loader. Install a PyTorch build compatible with your CUDA setup if the default `torch` package does not match your machine.

## Installation

Clone or download the project, then open a terminal in its root directory.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If Python 3.11 is unavailable, use Python 3.12. `.env` is local configuration and is excluded from version control; `.env.example` contains the project's default settings and a blank optional token variable.

## Build or rebuild the index

From the project root, run:

```bash
python -m src.ingest --rebuild
```

The command extracts the PDF, reports pages and chunks, writes chunk text and metadata, creates the Chroma vector index, and runs a small similarity-search smoke test. The first run downloads the embedding model.

Generated artifacts are kept under `storage/`:

- `storage/chunks.jsonl` — readable chunk text and metadata; also the source used to construct the BM25 index.
- `storage/chroma/` — persistent vector database.

Both generated paths are ignored by Git. `--rebuild` replaces the existing Chroma index and regenerates the chunk cache, so use it when you intentionally want to refresh the index.

## Inspect retrieval

Use the retrieval diagnostic command without loading the Llama model:

```bash
python -m src.inspect_retrieval "What does Article 4 say about AI literacy?"
```

The diagnostic compares vector-only, BM25-only, and hybrid RRF results and displays article, physical PDF page, chunk ID, and passage text. Use it to check whether the correct evidence is retrieved before investigating generated answers.

The interactive answer command is:

```bash
python -m src.qa
```

At the `Question>` prompt, enter a question about the regulation. Type `:debug` to toggle retrieval diagnostics or `exit` / `quit` to leave. The answer includes source labels and a list mapping those labels to the retrieved PDF page and, when available, Article number.

## Optional: enable answer generation

The answer-generation model is gated on Hugging Face:

1. Request access to [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and accept its terms.
2. On the machine that will run Q&A, authenticate with Hugging Face:

   ```bash
   hf auth login
   ```

3. Ensure that machine has a supported NVIDIA GPU, CUDA-enabled PyTorch, and a compatible `bitsandbytes` installation.
4. Run `python -m src.qa`.

The model download requires several gigabytes of disk space and substantial GPU memory. The project can use Hugging Face's local CLI credentials; do not put a real token in source code or commit it. The optional `HF_TOKEN` setting in `.env` is blank by default.

## Evaluate retrieval

After building the index, run:

```bash
python -m src.evaluate
```

The starter questions in `data/eval_questions.json` are labelled with relevant Article numbers. The script compares vector-only, hybrid RRF, and hybrid-plus-reranker retrieval using macro-averaged article-level Recall@5. This is a small starter set, not a validated benchmark: it does not identify gold evidence chunk IDs and should not be used to claim production retrieval quality or a specific performance improvement.

For a stronger evaluation, expand the question set, hold out questions during tuning, and label the exact evidence chunks needed to answer each question. Report the dataset, metric, and evaluation procedure alongside any results.

## Configuration

Edit `.env` and restart the command to apply settings. Environment variables already set in the shell take precedence over `.env`. Relative paths are resolved from the project root.

| Variable | Default | Purpose |
| --- | --- | --- |
| `EU_AI_ACT_PDF` | `data/EU_AI_Act.pdf` | Regulation PDF to ingest |
| `CHROMA_DIR` | `storage/chroma` | Persistent vector index directory |
| `CHUNKS_PATH` | `storage/chunks.jsonl` | Chunk cache used by BM25 |
| `EVAL_PATH` | `data/eval_questions.json` | Labelled retrieval questions |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for vector indexing and search |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional cross-encoder reranker |
| `LLM_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Gated model used for answer generation |
| `COLLECTION_NAME` | `eu_ai_act_v1` | Chroma collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Recursive chunk length and overlap, in characters |
| `VECTOR_CANDIDATES` | `20` | Number of initial vector-search candidates |
| `BM25_CANDIDATES` | `20` | Number of initial BM25 candidates |
| `RERANK_CANDIDATES` | `12` | Fused candidates sent to the reranker |
| `FINAL_K` | `5` | Number of passages returned to answer generation |
| `EMBEDDING_DEVICE` / `RERANKER_DEVICE` | `cpu` / `cpu` | Device used for embedding and reranking models |
| `HF_TOKEN` | *(blank)* | Optional Hugging Face token; CLI login is preferred |

Change retrieval settings and rebuild the index when changing settings that affect chunking or embeddings. A different embedding model must also be used consistently for indexing and querying.

## Project layout

```text
.
├── data/
│   ├── EU_AI_Act.pdf          # Source regulation
│   └── eval_questions.json    # Starter questions with Article labels
├── src/
│   ├── config.py              # Environment and path configuration
│   ├── ingest.py              # PDF extraction, chunking, and Chroma indexing
│   ├── retrieval.py           # Vector/BM25 retrieval, RRF, and reranking
│   ├── inspect_retrieval.py   # Retrieval diagnostics without Llama
│   ├── evaluate.py            # Article-level retrieval evaluation
│   ├── llm.py                 # CUDA 4-bit Llama loader
│   └── qa.py                  # Grounded prompt and interactive Q&A
├── storage/                   # Generated local index and chunk cache
├── .env.example               # Safe configuration template
├── requirements.txt
└── README.md
```

## Troubleshooting

- **PDF not found:** confirm `data/EU_AI_Act.pdf` exists or set `EU_AI_ACT_PDF` in `.env`.
- **No extractable text:** check the extraction preview and PDF integrity. The supplied PDF is text-based, so OCR should not normally be required.
- **Chroma index missing:** run `python -m src.ingest --rebuild` before retrieval, evaluation, or Q&A.
- **Existing index error during ingestion:** rerun with `--rebuild` only when you intend to replace the index.
- **Unexpected passages:** use `src.inspect_retrieval` to compare vector, BM25, and fused results before changing generation settings.
- **Hugging Face 401/403:** verify model access and run `hf auth login` on the same machine and account used to start the application.
- **CUDA unavailable:** ingestion and retrieval can run with CPU settings; the included 4-bit Llama loader requires CUDA.
- **CUDA out of memory:** reduce `RERANK_CANDIDATES` or `FINAL_K`, lower `max_new_tokens` in `src/llm.py`, or avoid keeping multiple models on the GPU at once.
- **Answer appears unsupported:** inspect its cited source passages. Grounded prompting and source labels help with verification but do not guarantee legal or factual correctness.
