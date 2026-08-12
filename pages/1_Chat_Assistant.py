import streamlit as st

from utils.security import sanitize_user_input
from utils.rag_engine import get_answer
from utils.feedback import save_feedback


# ============================================================
# PAGE SETUP
# ============================================================

st.title("💬 Vendors@Gov Billing Assistant")

st.caption(
    "Ask about vendor billing. The assistant will suggest the most relevant "
    "organisation, department, customer code, or sub-business unit based on "
    "the latest vendor registry"
)

st.write(
    "Use this Vendors@Gov Billing assistant to find the billing details e.g. Customer Code."
    "The response is generated from the uploaded vendor registry and should "
    "be verified against the vendor's billing context before use."
)


# ============================================================
# CONFIDENCE BADGES
# ============================================================

CONFIDENCE_BADGES = {
    "high": "🟢 High confidence",
    "medium": "🟡 Medium confidence — please review the suggested match",
    "low": "🔴 Low confidence — please verify manually",
    "none": "⚪ No vendor registry indexed",
}


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# HELPER: FORMAT METADATA
# ============================================================

def format_metadata(meta):
    if not meta:
        return ""

    parts = []

    if meta.get("timestamp"):
        parts.append(f"⏱ {meta.get('timestamp')}")

    if meta.get("backend"):
        parts.append(f"backend={meta.get('backend')}")

    if meta.get("collection"):
        parts.append(f"collection={meta.get('collection')}")

    if meta.get("retrieval_count") is not None:
        parts.append(f"retrievals={meta.get('retrieval_count')}")

    if meta.get("data_source"):
        parts.append(f"data source={meta.get('data_source')}")

    if meta.get("last_indexed"):
        parts.append(f"last indexed={meta.get('last_indexed')}")

    return ", ".join(parts)


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Example: I want to bill CAA Finance. Which customer code should I use?"
)


# ============================================================
# PROCESS USER INPUT
# ============================================================

if question:

    clean_question = sanitize_user_input(question)

    # --------------------------------------------------------
    # If input sanitisation modifies the question, do not query RAG
    # --------------------------------------------------------

    if clean_question != question:
        answer = clean_question
        sources = []
        confidence = "low"
        metadata = {
            "backend": "input_sanitisation",
            "retrieval_count": 0,
        }

    else:
        # ----------------------------------------------------
        # Query Qdrant through existing RAG engine
        #
        # Important:
        # The Chat Assistant should not read vendordata.json directly.
        # The Admin Upload page is responsible for uploading the registry,
        # rebuilding embeddings, and refreshing the Qdrant collection.
        # ----------------------------------------------------

        with st.spinner("Searching vendor registry..."):
            answer, sources, confidence, metadata = get_answer(clean_question)

    # --------------------------------------------------------
    # Save to session history
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "meta": metadata,
            "voted": None,
        }
    )

    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for i, entry in enumerate(st.session_state.chat_history):

    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant"):

        # ----------------------------------------------------
        # Confidence Badge
        # ----------------------------------------------------

        st.caption(
            CONFIDENCE_BADGES.get(
                entry.get("confidence"),
                "🔴 Low confidence — please verify manually"
            )
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(entry["answer"])

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_text = format_metadata(
            entry.get("meta")
        )

        if metadata_text:
            st.caption(metadata_text)

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if entry.get("sources"):
            st.caption(
                "Sources: " + ", ".join(entry["sources"])
            )

        # ----------------------------------------------------
        # Feedback
        # ----------------------------------------------------

        if entry["voted"] is None:

            col1, col2, col3 = st.columns([1, 1, 8])

            if col1.button("👍", key=f"up_{i}"):
                save_feedback(
                    entry["question"],
                    entry["answer"],
                    entry["sources"],
                    "up"
                )
                entry["voted"] = "up"
                st.rerun()

            if col2.button("👎", key=f"down_{i}"):
                save_feedback(
                    entry["question"],
                    entry["answer"],
                    entry["sources"],
                    "down"
                )
                entry["voted"] = "down"
                st.rerun()

        else:
            st.caption(
                f"Feedback recorded: {'👍' if entry['voted'] == 'up' else '👎'}"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("🕘 Question History")

    if st.session_state.chat_history:
        for i, entry in enumerate(
            reversed(st.session_state.chat_history),
            1
        ):
            st.markdown(
                f"**{i}.** {entry['question']}"
            )
    else:
        st.caption("No questions asked yet this session.")

    if st.button("Clear history"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("ℹ️ How to Ask")

    st.caption(
        "You may ask using organisation name, customer code, department name, "
        "or business description."
    )

    st.markdown(
        """
Examples:
- Which customer code should I use for CAA Finance?
- I am billing airport operations. What is the code I should use for billing?
- What is the likely billing department for procurement?
- Which sub-business unit should I use for HR-related billing?
"""
    )

    st.divider()

    st.subheader("📌 Note")

    st.caption(
        "The assistant uses the vendor registry indexed through the Admin Upload page. "
        "If the response looks outdated, ask the admin to upload the latest registry "
        "and rebuild the vector index."
    )
