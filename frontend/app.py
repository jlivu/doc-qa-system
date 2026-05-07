"""Streamlit frontend for the Document Q&A System.

Phase 3 — implement after the API endpoints are working.
Run locally with: streamlit run frontend/app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Q&A",
    page_icon=":page_facing_up:",
    layout="wide",
)

st.title("Intelligent Document Q&A")
st.caption("Upload government documents and ask questions in plain language.")

# ── Sidebar — document upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Ingest document"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"Ingested **{data['filename']}** — "
                    f"{data['pages']} pages, {data['chunks']} chunks"
                )
                st.session_state["document_id"] = data["document_id"]
                st.session_state["filename"] = data["filename"]
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")

    if "filename" in st.session_state:
        st.info(f"Active document: **{st.session_state['filename']}**")

# ── Main — question input ─────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about your document",
    placeholder="What was the total revenue in 2024?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching and generating answer..."):
        try:
            payload = {
                "question": question,
                "filters": (
                    {"document_id": st.session_state["document_id"]}
                    if "document_id" in st.session_state
                    else None
                ),
            }
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            if data.get("sources"):
                st.markdown("### Sources")
                for src in data["sources"]:
                    with st.expander(f"{src['filename']} — page {src['page']} (score: {src['score']:.2f})"):
                        st.write(src["text"])

        except requests.HTTPError as e:
            if e.response.status_code == 501:
                st.warning("The query pipeline is not yet implemented (Phase 2).")
            else:
                st.error(f"Query failed: {e}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
