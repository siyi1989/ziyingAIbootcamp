import os
import json
import html
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
except ImportError as exc:
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

BATCH_SIZE = 50
SECONDS_BETWEEN_BATCHES = 0


# ============================================================
# QDRANT ADAPTER
# ============================================================

class QdrantStoreAdapter:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def similarity_search_with_relevance_scores(self, query, k=12):
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

        adapted = []

        for doc, score in results:
            similarity = max(
                0.0,
                min(1.0, 1.0 - score)
            )

            adapted.append(
                (doc, similarity)
            )

        return adapted


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_data_dirs():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


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
                shutil.rmtree(
                    child,
                    ignore_errors=True
                )
            else:
                child.unlink(
                    missing_ok=True
                )

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
            raise RuntimeError(
                "Failed to create Qdrant client."
            )

        collections = client.get_collections().collections

        collection_exists = any(
            collection.name == cfg["collection_name"]
            for collection in collections
        )

        store_usable = False
        store_error = None

        if collection_exists:
            try:
                embeddings = get_embeddings()

                QdrantVectorStore(
                    client=client,
                    collection_name=cfg["collection_name"],
                    embedding=embeddings,
                )

                store_usable = True

            except Exception as err:
                store_error = str(err)

        return {
            "enabled": True,
            "configured": True,
            "collection_exists": collection_exists,
            "store_usable": store_usable,
            "error": store_error,
        }

    except Exception as exc:
        return {
            "enabled": False,
            "configured": True,
            "collection_exists": False,
            "store_usable": False,
            "error": str(exc),
        }


def get_persistence_status():
    ensure_data_dirs()
    qdrant_status = get_qdrant_status()

    return {
        "documents": list_documents(),
        "has_index": has_vector_store(),
        "using_qdrant": qdrant_status["enabled"],
        "qdrant_configured": qdrant_status["configured"],
        "qdrant_collection_exists": qdrant_status["collection_exists"],
        "qdrant_store_usable": qdrant_status["store_usable"],
        "qdrant_error": qdrant_status["error"],
    }


# ============================================================
# FILE MANAGEMENT
# ============================================================

def save_uploaded_file(uploaded_file):
    ensure_data_dirs()

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in LOADERS:
        raise ValueError(
            "Unsupported file type. Please upload a JSON file."
        )

    dest = DATA_DIR / uploaded_file.name

    with open(dest, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    return dest


def delete_document(filename):
    path = DATA_DIR / filename

    if path.exists():
        path.unlink()

    clear_vector_store()

    return not path.exists()


# ============================================================
# JSON HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        value = json.dumps(
            value,
            ensure_ascii=False
        )

    return html.unescape(
        str(value)
    ).strip()


def get_first_available(row, keys):
    for key in keys:
        value = row.get(key)

        if value not in [None, ""]:
            return safe_text(value)

    return ""


def split_code_and_name(value):
    value = safe_text(value)

    if " - " in value:
        code, name = value.split(
            " - ",
            1
        )

        return code.strip(), name.strip()

    return value.strip(), value.strip()


def flatten_json_value(value, prefix=""):
    lines = []

    if isinstance(value, dict):
        for key, nested_value in value.items():
            new_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            lines.extend(
                flatten_json_value(
                    nested_value,
                    new_prefix
                )
            )

    elif isinstance(value, list):
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}[{idx}]"

            lines.extend(
                flatten_json_value(
                    item,
                    new_prefix
                )
            )

    else:
        lines.append(
            f"{prefix}: {safe_text(value)}"
        )

    return lines


def normalise_json_payload(payload):
    """
    Supports these JSON structures:

    1. List of records:
       [
         {
           "MINISTRY / STATUTORY BOARD": "...",
           "DEPARTMENT": "...",
           "SUB-BUSINESS UNIT": "..."
         }
       ]

    2. Dictionary containing a list:
       {
         "records": [...]
       }

    3. Single dictionary:
       {
         "MINISTRY / STATUTORY BOARD": "..."
       }
    """

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in [
            "records",
            "data",
            "items",
            "rows",
            "vendor_registry",
            "vendorRegistry",
        ]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

        return [payload]

    return []


def build_vendor_registry_document(row, source_name, row_number):
    """
    Converts one Vendor Registry JSON row into one LangChain Document.

    This keeps each customer code / department record intact so that
    retrieval does not split a single registry row across multiple chunks.
    """

    if not isinstance(row, dict):
        row = {
            "value": row
        }

    ministry = get_first_available(
        row,
        [
            "MINISTRY / STATUTORY BOARD",
            "ministry",
            "organisation",
            "organization",
            "statutory_board",
            "statutoryBoard",
        ]
    )

    department = get_first_available(
        row,
        [
            "DEPARTMENT",
            "department",
            "division",
            "business_unit",
            "businessUnit",
        ]
    )

    sub_business_unit = get_first_available(
        row,
        [
            "SUB-BUSINESS UNIT",
            "sub_business_unit",
            "subBusinessUnit",
            "customer_code",
            "customerCode",
            "vendor_code",
            "vendorCode",
        ]
    )

    description = get_first_available(
        row,
        [
            "description",
            "Description",
            "remarks",
            "Remarks",
            "keywords",
            "Keywords",
            "alias",
            "aliases",
        ]
    )

    ministry_code, ministry_name = split_code_and_name(
        ministry
    )

    department_code, department_name = split_code_and_name(
        department
    )

    sub_code, sub_name = split_code_and_name(
        sub_business_unit
    )

    full_registry_row = "\n".join(
        flatten_json_value(row)
    )

    page_content = f"""
Vendor Registry Record

Organisation / Ministry / Statutory Board:
{ministry}

Organisation Code:
{ministry_code}

Organisation Name:
{ministry_name}

Department / Division:
{department}

Department Code:
{department_code}

Department Name:
{department_name}

Customer Code / Sub-Business Unit:
{sub_business_unit}

Customer Code:
{sub_code}

Sub-Business Unit Name:
{sub_name}

Description / Keywords:
{description}

Search Keywords:
{ministry}
{ministry_code}
{ministry_name}
{department}
{department_code}
{department_name}
{sub_business_unit}
{sub_code}
{sub_name}
{description}

Full Registry Row:
{full_registry_row}
""".strip()

    metadata = {
        "source": source_name,
        "row_number": row_number,
        "document_type": "vendor_registry",
        "search_type": "vendor_registry",
        "ministry": ministry,
        "ministry_code": ministry_code,
        "ministry_name": ministry_name,
        "department": department,
        "department_code": department_code,
        "department_name": department_name,
        "sub_business_unit": sub_business_unit,
        "customer_code": sub_code,
        "sub_business_unit_name": sub_name,
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )


def load_json_documents(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    records = normalise_json_payload(payload)

    docs = []

    for idx, row in enumerate(records, start=1):
        docs.append(
            build_vendor_registry_document(
                row=row,
                source_name=path.name,
                row_number=idx
            )
        )

    return docs


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_all_documents():
    docs = []

    for path in DATA_DIR.glob("*"):
        suffix = path.suffix.lower()

        if suffix not in LOADERS:
            continue

        if suffix == ".json":
            loaded = load_json_documents(path)
            docs.extend(loaded)
            continue

        loader_cls = LOADERS.get(suffix)

        if not loader_cls:
            continue

        loader = loader_cls(str(path))
        loaded = loader.load()

        for d in loaded:
            d.metadata["source"] = path.name
            d.metadata["document_type"] = "uploaded_document"

        docs.extend(loaded)

    return docs


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Configure it before building or loading the vector store."
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )


# ============================================================
# STREAMLIT SECRETS
# ============================================================

def _get_streamlit_secret(name):
    try:
        import streamlit as st
    except Exception:
        return None

    if st is None:
        return None

    secrets = getattr(st, "secrets", None)

    if secrets is None:
        return None

    if hasattr(secrets, "get"):
        return secrets.get(name)

    return getattr(secrets, name, None)


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

def get_qdrant_config():
    return {
        "url": os.getenv("QDRANT_URL") or _get_streamlit_secret("QDRANT_URL") or "",
        "api_key": os.getenv("QDRANT_API_KEY") or _get_streamlit_secret("QDRANT_API_KEY") or "",
        "collection_name": (
            os.getenv("QDRANT_COLLECTION_NAME")
            or _get_streamlit_secret("QDRANT_COLLECTION_NAME")
            or "vendor-registry"
        ),
    }


def use_qdrant():
    cfg = get_qdrant_config()

    return (
        bool(cfg["url"])
        and QdrantClient is not None
        and QdrantVectorStore is not None
        and models is not None
    )


@lru_cache(maxsize=1)
def get_qdrant_client():
    if not use_qdrant():
        return None

    cfg = get_qdrant_config()

    return QdrantClient(
        url=cfg["url"],
        api_key=cfg["api_key"] or None
    )


def ensure_qdrant_collection(client, collection_name):
    if client is None:
        return False

    collections = client.get_collections().collections

    if any(
        collection.name == collection_name
        for collection in collections
    ):
        return True

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        ),
    )

    return True


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(progress_callback=None):
    """
    Build vector index from supported files in data/raw_docs.

    For Vendor Registry JSON:
    - Each JSON row becomes one Document.
    - JSON rows are not split because each row represents one billing record.

    Qdrant is used when configured.
    FAISS is used as fallback.
    """

    ensure_data_dirs()

    docs = load_all_documents()

    if not docs:
        return None

    # Clear existing vectors before rebuilding.
    # This prevents stale registry records from remaining searchable.
    clear_vector_store()

    json_docs = [
        doc
        for doc in docs
        if doc.metadata.get("document_type") == "vendor_registry"
    ]

    other_docs = [
        doc
        for doc in docs
        if doc.metadata.get("document_type") != "vendor_registry"
    ]

    chunks = []

    # Keep Vendor Registry rows intact.
    for doc in json_docs:
        doc.metadata["search_type"] = "vendor_registry"
        chunks.append(doc)

    # Split PDF / DOCX / TXT only if such documents remain.
    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )

        chunks.extend(
            splitter.split_documents(other_docs)
        )

    if not chunks:
        return None

    embeddings = get_embeddings()

    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    if use_qdrant():
        client = get_qdrant_client()
        cfg = get_qdrant_config()

        ensure_qdrant_collection(
            client,
            cfg["collection_name"]
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=cfg["collection_name"],
            embedding=embeddings,
        )

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            if progress_callback:
                progress_callback(
                    batch_num,
                    total_batches
                )

            batch_texts = [
                chunk.page_content
                for chunk in batch
            ]

            batch_metadatas = [
                dict(chunk.metadata)
                for chunk in batch
            ]

            vector_store.add_texts(
                batch_texts,
                metadatas=batch_metadatas
            )

            if i + BATCH_SIZE < len(chunks):
                time.sleep(
                    SECONDS_BETWEEN_BATCHES
                )

        return QdrantStoreAdapter(
            vector_store
        )

    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        if progress_callback:
            progress_callback(
                batch_num,
                total_batches
            )

        if vector_store is None:
            vector_store = FAISS.from_documents(
                batch,
                embeddings
            )
        else:
            vector_store.add_documents(
                batch
            )

        if i + BATCH_SIZE < len(chunks):
            time.sleep(
                SECONDS_BETWEEN_BATCHES
            )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(INDEX_DIR)
    )

    return vector_store


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    global QDRANT_STORE_ERROR

    ensure_data_dirs()

    if use_qdrant():
        try:
            client = get_qdrant_client()
            cfg = get_qdrant_config()

            if client is None:
                return None

            collections = client.get_collections().collections

            collection_exists = any(
                collection.name == cfg["collection_name"]
                for collection in collections
            )

            if not collection_exists:
                if has_documents():
                    return build_vector_store()

                return None

            embeddings = get_embeddings()

            vector_store = QdrantVectorStore(
                client=client,
                collection_name=cfg["collection_name"],
                embedding=embeddings,
            )

            QDRANT_STORE_ERROR = None

            return QdrantStoreAdapter(
                vector_store
            )

        except Exception as exc:
            QDRANT_STORE_ERROR = str(exc)
            return None

    if not has_vector_store():
        if has_documents():
            try:
                return build_vector_store()
            except Exception:
                return None

        return None

    try:
        embeddings = get_embeddings()
    except Exception:
        return None

    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception:
        return None
