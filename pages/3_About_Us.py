import streamlit as st

st.title("ℹ️ About This Project")

st.markdown(
    """
### Project Scope

The **Vendors@Gov Billing Assistant** is a Retrieval-Augmented Generation (RAG)
solution designed to help officers identify the most appropriate organisation,
department, customer code, or sub-business unit when processing vendor billing
transactions in Vendors@Gov.

The assistant searches information from the latest Vendor Registry uploaded by
administrators and uses semantic search to recommend the most relevant billing
details based on the user's query.

### Objectives

- Provide a fast and intuitive way for officers to identify the correct billing entity.
- Reduce manual searching through large Vendor Registry files.
- Improve consistency in the use of customer codes and billing departments.
- Assist users when vendors are unable to clearly identify the correct division or customer code.
- Provide a controlled process for administrators to maintain and update the Vendor Registry.

### Data Sources

- Vendor Registry data uploaded by Administrators in JSON format.
- Registry records are converted into embeddings and stored in a Qdrant vector database.
- The Chat Assistant retrieves relevant records from the indexed Vendor Registry and generates responses based on the retrieved information.

### Key Features

- 🔐 Role-based access control for Administrators
- 📤 Admin upload and management of Vendor Registry files
- 🔄 Rebuildable vector index using Qdrant
- 💬 Natural language search across vendor billing records
- 🎯 Suggestion of likely organisation, department, customer code, and sub-business unit
- 🕘 Session-based question history
- 👍 User feedback collection for response quality monitoring
- 🛡️ Input validation and prompt injection protection

### Intended Users

#### Administrator

Responsible for:

- Uploading the latest Vendor Registry
- Managing registry files
- Rebuilding the vector index
- Maintaining data quality and accuracy

#### User

Responsible for:

- Searching for billing information
- Identifying the appropriate organisation or department
- Obtaining customer codes and sub-business unit information
- Verifying billing details before submission in Vendors@Gov

### Important Note

The assistant provides recommendations based on the information available in the
uploaded Vendor Registry and semantic matching performed by the RAG engine.

Users remain responsible for validating the final billing details before
submitting transactions in Vendors@Gov, particularly where multiple possible
matches are returned.
"""
)
