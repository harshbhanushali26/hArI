# hArI — Document Intelligence System

> *Talk to your documents. Understand your data.*

hArI is an AI-powered document intelligence system that lets you upload PDFs, CSVs, and Excel files — then have a natural conversation with their contents. It uses a RAG (Retrieval-Augmented Generation) pipeline for PDF semantic search and LLM-driven pandas code execution for structured data analysis.

---

## Features

- **PDF Chat** — Semantic search over PDF content using ChromaDB + sentence-transformers
- **CSV / Excel Analysis** — Natural language queries converted to pandas operations via LLM
- **Mixed File Mode** — Upload both types together; hArI automatically routes each query to the right engine
- **Streaming Responses** — Token-by-token streaming with live cursor effect (ChatGPT-style)
- **Conversation Memory** — Maintains context across turns; summarizes older messages when the context window fills up
- **Persistent Vector Store** — PDF embeddings stored across sessions; no re-processing on reload
- **Modular Architecture** — Clean separation of UI, state, handlers, and core logic

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| RAG Pipeline | LangChain + ChromaDB |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| LLM | Groq API (`llama-4-scout-17b`) |
| Data Analysis | Pandas + NumPy |
| PDF Parsing | PyMuPDF |

---

## Project Structure

```
hArI/
├── app.py                  # Entry point — page config + render orchestration
├── config.py               # All config constants (models, chunking, RAG, memory, limits)
│
├── core/
│   ├── __init__.py         
│   ├── file_processor.py   # File loading, type detection, PDF/CSV/Excel parsing
│   ├── memory.py           # Conversation context + summarization logic
│   ├── query_intent.py     # Query classifier — routes to PDF (RAG) or CSV engine
│   ├── responser.py        # Prompt builder + streaming Groq response handler
│   └── utils.py            # Shared utilities (strip_thinking, Groq client helpers)
│
├── rag/
│   ├── __init__.py         
│   ├── embedder.py         # Splits docs into chunks, generates embeddings
│   ├── retriever.py        # Embeds user query, retrieves top-k chunks from ChromaDB
│   └── vector_store.py     # ChromaDB client, collection management, deduplication
│
├── ui/
│   ├── __init__.py         
│   ├── styles.py           # Global CSS injection (dark theme, purple accents)
│   ├── state.py            # Session state init + clear_chat()
│   ├── handlers.py         # Ingest, query, reset, remove file pipelines
│   └── components.py       # All Streamlit render functions
│
├── prompts/
│   └── system_prompt.md    # AI identity + mode-specific rules (PDF, CSV, General)
│
├── data/
│   └── chroma_store/       # Persistent ChromaDB vector storage
│
├── .env                    # API keys (not committed)
├── pyproject.toml
└── uv.lock
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/hArI.git
cd hArI

# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

---

## How It Works

### PDF Mode (RAG Pipeline)

```
Upload PDF
    │
    ▼
PyMuPDF extracts text
    │
    ▼
Text split into chunks (embedder.py)
    │
    ▼
Embeddings generated (all-MiniLM-L6-v2)
    │
    ▼
Stored in ChromaDB (vector_store.py)
    │
    ▼
User query → embed → retrieve top-k chunks (retriever.py)
    │
    ▼
Chunks + memory context injected into prompt (responser.py)
    │
    ▼
Groq streams answer token-by-token → rendered live in UI
```

### CSV / Excel Mode (Direct LLM)

```
Upload CSV / Excel
    │
    ▼
Pandas DataFrame created (file_processor.py)
    │
    ▼
Schema + sample rows extracted as metadata
    │
    ▼
User query + schema → Groq generates pandas code
    │
    ▼
Code executed safely → result passed back to LLM
    │
    ▼
Groq streams formatted answer → rendered live in UI
```

### Mixed Mode

When both file types are present, `query_intent.py` uses the LLM to classify whether each query is best answered by the PDF RAG engine or the CSV analysis engine — then routes accordingly.

---

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Description |
|---|---|
| `CHUNK_SIZE` | Token size per text chunk for PDF splitting |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | Number of chunks retrieved per query |
| `MEMORY_BUFFER_SIZE` | Max messages before summarization triggers |
| `MAX_FILE_SIZE_MB` | Upload size limit per file |
| `EMBEDDING_MODEL` | Sentence-transformer model name |
| `LLM_MODEL` | Groq model identifier |
| `ANALYSIS_MODEL` | Groq model used for pandas code generation |

---

## Known Limitations

- PDF support only (no `.docx`, `.txt` currently)
- CSV/Excel analysis depends on LLM-generated pandas code — complex queries may occasionally fail
- ChromaDB requires `hnsw:space: cosine` in collection metadata for correct similarity scoring

---

## Roadmap

- [x] Streaming LLM responses in UI
- [x] Modular ui/ folder architecture
- [ ] ChromaDB cosine similarity threshold filtering
- [ ] Add `.docx` and `.txt` support
- [ ] Multi-collection support (separate namespaces per session)
- [ ] Export chat history
- [ ] Confidence score display on retrieved chunks

---

## License

MIT License. See `LICENSE` for details.

---

*Built by [Harsh Bhanushali](https://github.com/harshbhanushali26)*