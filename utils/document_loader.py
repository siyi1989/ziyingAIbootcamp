import os
import json
import shutil
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_IMPORT_ERROR = None
QDRANT_STORE_ERROR = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from langchain_qdrant import QdrantVectorStore
except ImportError as exc:  # pragma: no cover - optional dependency
    QdrantClient = None
    models = None
    QdrantVectorStore = None
    QDRANT_IMPORT_ERROR = str(exc)


# ============================================================
# BASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data/raw_docs"
INDEX_DIR = BASE_DIR / "data/vector_store"

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".json": None,
}

# Free-tier limit is 100 embedding requests/minute — stay comfortably under it
BATCH_SIZE = 50
SECONDS_BETWEEN_BATCHES = 0


# ============================================================
# QDRANT ADAPTER
# ============================================================

class QdrantStoreAdapter:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def similarity_search_with_relevance_scores(self, query, k=12):
        results = self.vector_store.similarity_search_with_score(query, k=k)
        adapted = []

        for doc, score in results:
            similarity = max(0.0, min(1.0, 1.0 - score))
            adapted.append((doc, similarity))

        return adapted


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_data_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def list_documents():
    ensure_data_dirs()
    return sorted(
        p.name
        for p in DATA_DIR.glob("*")
        if p.suffix.lower() in LOADERS
    )


def has_documents():
    ensure_data_dirs()
    return any(
        p.suffix.lower() in LOADERS
        for p in DATA_DIR.glob("*")
    )


# ============================================================
# VECTOR STORE STATUS
# ============================================================

def has_vector_store():
    ensure_data_dirs()

    if use_qdrant():
        try:
            client = get_qdrant_client()
            cfg = get_qdrant_config()

            if client is None:
                return False

            collections = client.get_collections().collections

            return any(
                collection.name == cfg["collection_name"]
                for collection in collections
            )

        except Exception:
            return False

    return any(
        (INDEX_DIR / name).exists()
        for name in ["index.faiss", "index.pkl"]
    )


def clear_vector_store():
    ensure_data_dirs()

    if use_qdrant():
        try:
            client = get_qdrant_client()
            cfg = get_qdrant_config()

            if client is not None:
                collections = client.get_collections().collections
                collection_exists = any(
                    collection.name == cfg["collection_name"]
                    for collection in collections
                )

                if collection_exists:
                    client.delete_collection(
                        collection_name=cfg["collection_name"]
                    )

        except Exception:
            pass

    if INDEX_DIR.exists():
        for child in INDEX_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    return True


# ============================================================
# QDRANT STATUS
# ============================================================

def get_qdrant_status():
    cfg = get_qdrant_config()

    if not cfg["url"]:
        return {
            "enabled": False,
            "configured": False,
            "collection_exists": False,
            "store_usable": False,
            "error": "QDRANT_URL is not configured.",
        }

    if QdrantClient is None or QdrantVectorStore is None or models is None:
        return {
            "enabled": False,
            "configured": True,
            "collection_exists": False,
            "store_usable": False,
            "error": f"Qdrant dependencies are not installed: {QDRANT_IMPORT_ERROR}",
        }

    try:
        client = get_qdrant_client()

        if client is None:
           
