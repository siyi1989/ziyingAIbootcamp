import streamlit as st
import graphviz

st.title("🔍 Methodology")

st.markdown(
    """
This solution uses Retrieval-Augmented Generation (RAG) to help users identify
the most appropriate organisation, department, customer code, or
sub-business unit for vendor billing.

The solution has two main processes:

1. **Vendor Registry Management** (Administrator)
2. **Vendor Billing Search & Recommendation** (User)
"""
)

# ============================================================
# ADMIN FLOW
# ============================================================

st.subheader("1️⃣ Vendor Registry Management")

g1 = graphviz.Digraph()
g1.attr(rankdir="LR")

g1.node("A", "Admin uploads\nVendor Registry JSON")
g1.node("B", "JSON saved to\ndata folder")
g1.node("C", "Admin clicks\n'Rebuild Index'")
g1.node("D", "Registry records\nparsed")
g1.node("E", "Searchable text\nconstructed")
g1.node("F", "Embeddings\ngenerated")
g1.node("G", "Stored in\nQdrant Collection")

g1.edges([
    "AB",
    "BC",
    "CD",
    "DE",
    "EF",
    "FG",
])

st.graphviz_chart(g1)

st.markdown(
    """
### Registry Upload

Administrators upload the latest Vendor Registry in JSON format.

### Registry Processing

Each registry record is processed into searchable content containing information such as:

- Organisation
- Department
- Customer Code
- Sub-Business Unit
- Keywords and descriptions

### Embedding Generation

The processed records are converted into vector embeddings.

### Vector Storage

Embeddings are stored in a Qdrant collection, enabling semantic search and persistence across application restarts.
"""
)

# ============================================================
# CHAT FLOW
# ============================================================

st.subheader("2️⃣ Vendor Billing Search & Recommendation")

g2 = graphviz.Digraph()
g2.attr(rankdir="LR")

g2.node("H", "User enters\nbilling query")
g2.node("I", "Input validation\nand screening")
g2.node("J", "Question embedded")
g2.node("K", "Qdrant semantic\nsearch")
g2.node("L", "Relevant registry\nrecords retrieved")
g2.node("M", "LLM analyses\nretrieved records")
g2.node("N", "Recommended\nbilling details")
g2.node("O", "Response shown\nto user")

g2.edges([
    "HI",
    "IJ",
    "JK",
    "KL",
    "LM",
    "MN",
    "NO",
])

st.graphviz_chart(g2)

st.markdown(
    """
### User Query

Users may search using:

- Organisation names
- Department names
- Customer codes
- Business functions
- Vendor descriptions
- Billing requirements

Example:

> Vendor says the work relates to Finance Division. Which customer code should be used?

### Retrieval

The user's query is converted into an embedding and compared against Vendor Registry records stored in Qdrant.

The most relevant records are retrieved based on semantic similarity rather than exact keyword matching.

### Context Grounding

Only the retrieved Vendor Registry records are provided to the Large Language Model (LLM).

The LLM is instructed to:

- Recommend likely matches
- Explain the rationale
- Present alternative matches where applicable
- Avoid generating customer codes not found in the registry

### Response Generation

The assistant returns:

- Recommended Organisation
- Recommended Department
- Customer Code (where available)
- Sub-Business Unit (where available)
- Alternative matches when confidence is lower

### Security Controls

- Input validation and sanitisation
- Prompt injection screening
- Grounded responses based on retrieved registry data
- Administrator-controlled registry updates
"""
)

# ============================================================
# TECH STACK
# ============================================================

st.subheader("3️⃣ Technical Architecture")

st.markdown(
    """
### Tech Stack

| Layer | Technology |
|---------|---------|
| User Interface | Streamlit |
| Authentication | Custom Role-Based Access |
| Orchestration | LangChain |
| Vector Database | Qdrant |
| Embeddings | OpenAI Embeddings |
| Large Language Model | GPT-4o-mini |
| Data Source | Vendor Registry JSON |
| Storage | Qdrant + Local Data Folder |
| Deployment | Docker / Streamlit Cloud |

### RAG Workflow Summary

1. Administrator uploads Vendor Registry.
2. Registry records are embedded and stored in Qdrant.
3. User submits a billing-related question.
4. Relevant registry records are retrieved from Qdrant.
5. The LLM generates recommendations using only retrieved records.
6. The user receives a suggested organisation, department, customer code, or sub-business unit.
"""
)
