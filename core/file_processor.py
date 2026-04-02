"""
file_processor.py
-----------------
Handles all file ingestion for hArI.

Responsibilities:
    - Detect file type from extension (.pdf, .csv, .xlsx, .xls)
    - Load and extract content from each file type
    - Return a unified result dict consumed by the RAG pipeline and app layer

Supported file types:
    - PDF    → extracted page-by-page using PyMuPDF (fitz), returned as list[Document]
    - CSV    → loaded via pandas, returned as (DataFrame, metadata)
    - Excel  → loaded via pandas (first sheet), returned as (DataFrame, metadata)

Design decisions:
    - Custom Document class mirrors LangChain's structure without the dependency
    - PDF loaded from byte stream (fitz stream mode) — no tempfile required
    - CSV/Excel loaded directly from Streamlit BytesIO object — no tempfile required
    - Blank/image-only PDF pages are skipped silently
    - All errors raised explicitly — no silent empty returns

Public API:
    load_file(file)         -> dict   (main entry point)
    is_supported(file)      -> bool
    get_file_size_mb(file)  -> str

Internal:
    load_pdf_file(file)     -> list[Document]
    load_csv_file(file)     -> tuple[DataFrame, dict]
    load_excel_file(file)   -> tuple[DataFrame, dict]
    _get_file_extension()
    _clean_dataframe()
    _build_df_metadata()
"""

import fitz
import pandas as pd
from config import SUPPORTED_EXTENSIONS
from core.utils import get_file_extension


# ── Custom Document Class ─────────────────────────────────────────────────────

class Document:
    """Mimics LangChain Document structure — no LangChain dependency needed."""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(source={self.metadata.get('source')}, page={self.metadata.get('page')}, chars={len(self.page_content)})"

# ── File Type Detection ───────────────────────────────────────────────────────

def _get_file_extension(file) -> str:
    """Thin wrapper kept for internal use — delegates to utils.get_file_extension."""
    return get_file_extension(file)


def is_supported(file) -> bool:
    """Returns True if file extension is in SUPPORTED_EXTENSIONS."""
    return _get_file_extension(file) in SUPPORTED_EXTENSIONS

# ── Main Router ───────────────────────────────────────────────────────────────

def load_file(file) -> dict:
    """
    Master entry point. Accepts a Streamlit uploaded file object.
    Routes to correct loader based on extension.

    Returns unified dict:
    {
        "file_type" : "pdf" | "csv" | "excel",
        "filename"  : str,
        "documents" : list[Document] | None,    <- PDF only
        "dataframe" : pd.DataFrame | None,       <- CSV/Excel only
        "metadata"  : dict
    }
    """
    if not is_supported(file):
        raise ValueError(f"Unsupported file: {file.name}. Allowed: {SUPPORTED_EXTENSIONS}")

    ext = _get_file_extension(file)

    if ext == ".pdf":
        documents = load_pdf_file(file)
        return {
            "file_type" : "pdf",
            "filename"  : file.name,
            "documents" : documents,
            "dataframe" : None,
            "metadata"  : {
                "source"      : file.name,
                "file_type"   : "pdf",
                "total_pages" : documents[0].metadata["total_pages"] if documents else 0,
                "total_docs"  : len(documents),
                "file_size"   : get_file_size_mb(file)
            }
        }

    if ext == ".csv":
        df, meta = load_csv_file(file)
        return {
            "file_type" : "csv",
            "filename"  : file.name,
            "documents" : None,
            "dataframe" : df,
            "metadata"  : meta
        }

    if ext in [".xlsx", ".xls"]:
        df, meta = load_excel_file(file)
        return {
            "file_type" : "excel",
            "filename"  : file.name,
            "documents" : None,
            "dataframe" : df,
            "metadata"  : meta
        }

# ── PDF Loader ────────────────────────────────────────────────────────────────

def load_pdf_file(file) -> list[Document]:
    """
    Loads PDF directly from memory using PyMuPDF (fitz).
    No tempfile needed — reads from byte stream.
    Returns list of Document objects, one per page.
    Skips blank or image-only pages.
    """
    file.seek(0)
    file_bytes = file.read()
    documents = []

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            total_pages = doc.page_count
            for page_num, page in enumerate(doc):
                text = page.get_text("text")

                if not text.strip():
                    continue  # skip blank / image-only pages

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source"      : file.name,
                        "filename"    : file.name,
                        "page"        : page_num + 1,
                        "total_pages" : total_pages,
                        "file_type"   : "pdf"
                    }
                ))

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{file.name}': {e}")

    if not documents:
        raise ValueError(f"No extractable text in '{file.name}'. File may be scanned or image-based.")

    return documents

# ── CSV Loader ────────────────────────────────────────────────────────────────

def load_csv_file(file) -> tuple[pd.DataFrame, dict]:
    """
    Loads CSV directly from Streamlit file object.
    pandas read_csv() accepts BytesIO natively — no tempfile needed.
    Returns cleaned DataFrame and metadata dict.
    """
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV '{file.name}': {e}")

    df = _clean_dataframe(df)
    return df, _build_df_metadata(df, file.name, "csv", get_file_size_mb(file))

# ── Excel Loader ──────────────────────────────────────────────────────────────

def load_excel_file(file) -> tuple[pd.DataFrame, dict]:
    """
    Loads Excel directly from Streamlit file object.
    pandas read_excel() accepts BytesIO natively — no tempfile needed.
    Reads first sheet by default.
    Returns cleaned DataFrame and metadata dict.
    """
    file.seek(0)
    try:
        df = pd.read_excel(file, sheet_name=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel '{file.name}': {e}")

    df = _clean_dataframe(df)
    return df, _build_df_metadata(df, file.name, "excel", get_file_size_mb(file))

# ── Internal Helpers ──────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Strip whitespace from column names
    - Drop fully empty rows and columns
    - Reset index
    """
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)
    return df


def _build_df_metadata(df: pd.DataFrame, filename: str, file_type: str, file_size: str) -> dict:
    """
    Builds metadata dict from DataFrame.
    preview of first 3 rows used later in CSV system prompt for LLM context.
    """
    return {
        "source"    : filename,
        "file_type" : file_type,
        "file_size" : file_size,
        "rows"      : len(df),
        "columns"   : list(df.columns),
        "dtypes"    : df.dtypes.astype(str).to_dict(),
        "preview"   : df.head(3).to_dict(orient="records")
    }


def get_file_size_mb(file) -> str:
    """Returns formatted file size string e.g. '2.1 MB'"""
    return f"{file.size / 1024 / 1024:.1f} MB"