import streamlit as st

from utils.auth import require_login
from utils.document_loader import (
    list_documents,
    save_uploaded_file,
    delete_document,
    build_vector_store,
    get_persistence_status,
)

require_login(allowed_roles=["Admin"])

# ============================================================
# PAGE HEADER
# ============================================================

st.title("📤 Admin — Manage Vendor Registry")

st.caption(
    "Upload Vendor Registry JSON files and rebuild the Qdrant index. "
    "Only Admins can access this page."
)

st.info(
    "Uploaded registry files are used to rebuild the vector database. "
    "The Chat Assistant retrieves information from Qdrant rather than "
    "reading the JSON file directly."
)

# ============================================================
# QDRANT STATUS
# ============================================================

status = get_persistence_status()

if status.get("using_qdrant"):

    if status.get("qdrant_store_usable"):

        st.success(
            "Qdrant Cloud is configured and the Vendor Registry collection "
            "is available for retrieval."
        )

        if status.get("qdrant_collection_exists"):

            st.caption(
                "Vendor Registry collection detected."
            )

    else:

        st.warning(
            "Qdrant Cloud is configured but the vector store "
            "cannot be initialised."
        )

        if status.get("qdrant_error"):
            st.error(f"Qdrant error: {status['qdrant_error']}")

    st.markdown(
        f"""
**Debug Information**

- Collection Exists: {status.get('qdrant_collection_exists')}
- Store Usable: {status.get('qdrant_store_usable')}
"""
    )

elif status.get("qdrant_configured"):

    st.warning(
        "Qdrant is configured but connection failed. "
        "Check credentials and network access."
    )

    if status.get("qdrant_error"):
        st.error(f"Qdrant error: {status['qdrant_error']}")

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Vendor Registry JSON",
    type=["json"],
    accept_multiple_files=True,
)

if uploaded_files:

    uploaded_count = 0

    for f in uploaded_files:

        save_uploaded_file(f)
        uploaded_count += 1

    st.success(
        f"Successfully uploaded {uploaded_count} JSON file(s). "
        "Click 'Rebuild Index' to update the vector database."
    )

# ============================================================
# CURRENT FILES
# ============================================================

st.divider()

st.subheader("📚 Current Registry Files")

docs = list_documents()

if docs:

    for idx, d in enumerate(docs):

        col1, col2 = st.columns([4, 1])

        col1.write(d)

        if col2.button(
            "🗑️ Delete",
            key=f"delete_{idx}_{d}"
        ):
            delete_document(d)
            st.rerun()

else:

    st.info(
        "No Vendor Registry files uploaded."
    )

# ============================================================
# REBUILD INDEX
# ============================================================

st.divider()

st.subheader("🔄 Vector Rebuild")

st.caption(
    "Rebuild the Vendor Registry collection after uploading "
    "a new registry file."
)

if st.button(
    "🔄 Rebuild Index",
    type="primary"
):

    progress_bar = st.progress(
        0,
        text="Preparing documents..."
    )

    def update_progress(
        batch_num,
        total_batches
    ):
        progress_bar.progress(
            batch_num / total_batches,
            text=f"Embedding batch {batch_num} of {total_batches}"
        )

    try:

        vs = build_vector_store(
            progress_callback=update_progress
        )

        progress_bar.empty()

        if vs is None:

            st.warning(
                "No registry files found for indexing."
            )

        else:

            st.success(
                "Vendor Registry vector index rebuilt successfully."
            )

            st.info(
                "The Chat Assistant will now use the latest Vendor Registry data."
            )

    except Exception as ex:

        progress_bar.empty()

        st.error(
            f"Index rebuild failed: {str(ex)}"
        )

# ============================================================
# INSTRUCTIONS
# ============================================================

st.divider()

with st.expander("ℹ️ Upload Instructions"):

    st.markdown(
        """
1. Export the latest Vendor Registry.
2. Save as JSON format.
3. Upload the JSON file.
4. Click **Rebuild Index**.
5. Wait for indexing to complete.
6. Test retrieval in the Chat Assistant page.

Recommended JSON structure:

```json
[
  {
    "organisation": "CAA",
    "department": "Finance Division",
    "customer_code": "CAA15",
    "sub_business_unit": "CAA1501"
  }
]
