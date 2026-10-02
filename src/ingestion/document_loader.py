# ============================================================
# src/ingestion/document_loader.py - Document Loading & Chunking
#
# Learning Note:
#   This is Step 1 of any RAG system - getting text out of files.
#   Key decisions:
#     1. CHUNKING STRATEGY: How you split text matters enormously.
#        Too small = chunks lose context. Too large = irrelevant
#        text dilutes the answer. RecursiveCharacterTextSplitter
#        tries paragraph -> sentence -> word level splits in order.
#     2. CHUNK OVERLAP: We overlap chunks by ~200 chars so that
#        context around a split boundary is not lost.
#     3. METADATA: We always attach source filename and page number
#        so we can cite sources in answers.
# ============================================================

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None
from langchain_core.documents import Document

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every document chunk.
    This is what allows the system to say "Answer from: doc.pdf, page 3"
    """
    source_file: str
    chunk_index: int
    page_number: int = 0
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_type: str = ""
    total_chunks: int = 0


class DocumentLoader:
    """
    Loads documents from disk or GCS and splits them into chunks
    ready for embedding and indexing.

    Supported formats: PDF, DOCX, TXT, Markdown
    """

    # Map file extensions to LangChain loader classes
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        # Learning Note: chunk_size=512 tokens is a sweet spot for most
        # embedding models. chunk_overlap=128 ensures boundary context.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Order matters: try to split on paragraph first, then sentence,
            # then word, and only then arbitrary characters.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and return a list of LangChain Documents.
        Each Document has .page_content (text) and .metadata (dict).
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.LOADER_MAP:
            logger.warning("unsupported_file_type", file=file_path, ext=ext)
            return []

        logger.info("loading_document", file=file_path, type=ext)

        try:
            # Text and Markdown use UTF-8 without optional NLP downloads.
            if ext in {".txt", ".md"}:
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                loader = self.LOADER_MAP[ext](file_path)

            raw_docs = loader.load()
            logger.info("document_loaded", file=file_path, pages=len(raw_docs))
            return raw_docs

        except Exception as exc:
            logger.error("document_load_failed", file=file_path, error=str(exc))
            return []

    def chunk_documents(self, documents: List[Document], source_name: str) -> List[Document]:
        """
        Split raw documents into smaller chunks for embedding.

        Learning Note:
            After splitting, we enrich metadata on each chunk. This
            metadata travels with the chunk into the vector store and
            comes back during retrieval - this is how we know WHERE
            an answer came from.
        """
        chunks = self.splitter.split_documents(documents)

        # Enrich metadata on each chunk
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": source_name,
                "chunk_index": idx,
                "total_chunks": total,
                "document_id": str(uuid.uuid4()),
            })

        logger.info(
            "document_chunked",
            source=source_name,
            raw_pages=len(documents),
            chunks_created=total,
            avg_chunk_size=sum(len(c.page_content) for c in chunks) // max(total, 1),
        )
        return chunks

    def load_and_chunk(self, file_path: str) -> List[Document]:
        """
        Convenience method: load a file AND chunk it in one call.
        Returns list of chunks ready for embedding.
        """
        source_name = Path(file_path).name
        raw_docs = self.load_file(file_path)
        if not raw_docs:
            return []
        return self.chunk_documents(raw_docs, source_name)

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        Load all supported files from a directory.
        Useful for batch ingestion.
        """
        all_chunks = []
        directory = Path(dir_path)

        if not directory.exists():
            logger.error("directory_not_found", path=dir_path)
            return []

        supported_exts = set(self.LOADER_MAP.keys())
        files = [f for f in directory.iterdir() if f.suffix.lower() in supported_exts]

        logger.info("loading_directory", path=dir_path, files_found=len(files))

        for file_path in files:
            chunks = self.load_and_chunk(str(file_path))
            all_chunks.extend(chunks)

        logger.info("directory_load_complete", total_chunks=len(all_chunks))
        return all_chunks
