"""Streamlit frontend for the Document Q&A System.

Professional, demo-ready UI with PDF upload, query interface with
conversation history sidebar, document list with delete, and source
citation display with highlights and confidence badges.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── API helpers ──────────────────────────────────────────────────────────────

def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=30)


def api_post_json(path: str, payload: dict):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=60)


def api_post_file(path: str, files: dict):
    return requests.post(f"{API_URL}{path}", files=files, timeout=120)


def api_delete(path: str):
    return requests.delete(f"{API_URL}{path}", timeout=30)


def refresh_documents():
    """Fetch the document list from the API and cache in session state."""
    try:
        resp = api_get("/documents")
        if resp.status_code == 200:
            st.session_state.documents = resp.json()["documents"]
        else:
            st.session_state.documents = []
    except requests.RequestException:
        st.session_state.documents = []


# ── Session state init ───────────────────────────────────────────────────────

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "needs_refresh" not in st.session_state:
    st.session_state.needs_refresh = True
if "polling_job_id" not in st.session_state:
    st.session_state.polling_job_id = None
if "polling_filename" not in st.session_state:
    st.session_state.polling_filename = None
if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None
if "confirm_delete_name" not in st.session_state:
    st.session_state.confirm_delete_name = None
if "confirm_clear_conversation" not in st.session_state:
    st.session_state.confirm_clear_conversation = False

# Refresh document list on first load or after ingest/delete
if st.session_state.needs_refresh:
    refresh_documents()
    st.session_state.needs_refresh = False


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Q&A — Vanuatu Gov",
    page_icon="📄",
    layout="wide",
)


# ── Sidebar — Document Library ───────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Document Library")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ingest") and uploaded:
        try:
            files = {"file": (uploaded.name, uploaded.read(), "application/pdf")}
            resp = api_post_file("/ingest", files=files)
            if resp.status_code == 202:
                job = resp.json()
                st.session_state.polling_job_id = job["job_id"]
                st.session_state.polling_filename = job["filename"]
            elif resp.status_code == 201:
                data = resp.json()
                st.success(f"{data['pages']} pages, {data['chunks']} chunks")
                refresh_documents()
            else:
                st.error(resp.json().get("detail", "Ingestion failed"))
        except requests.RequestException as e:
            st.error(f"Cannot reach the API: {e}")

    # Polling loop for async ingestion
    if st.session_state.polling_job_id:
        import time
        status_box = st.empty()
        job_id = st.session_state.polling_job_id
        fname = st.session_state.polling_filename
        status_box.info(f"Ingesting {fname}...")
        start = time.time()
        while time.time() - start < 300:
            try:
                resp = api_get(f"/jobs/{job_id}")
                if resp.status_code != 200:
                    break
                job = resp.json()
                if job["status"] == "completed":
                    status_box.success(
                        f"{job['pages']} pages, {job['chunks']} chunks"
                    )
                    refresh_documents()
                    st.session_state.polling_job_id = None
                    break
                if job["status"] == "failed":
                    status_box.error(
                        f"Ingestion failed: {job.get('error', 'Unknown error')}"
                    )
                    st.session_state.polling_job_id = None
                    break
                status_box.info(
                    f"Processing {fname}... (this may take a moment)"
                )
            except requests.RequestException:
                status_box.error("Lost connection to API during polling.")
                st.session_state.polling_job_id = None
                break
            time.sleep(2)
        else:
            status_box.error("Ingestion timed out after 5 minutes.")
            st.session_state.polling_job_id = None

    st.divider()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            cols = st.columns([4, 1, 1, 1])
            cols[0].write(doc["filename"])
            cols[1].write(f"{doc['pages']}p")
            cols[2].write(f"{doc['chunk_count']}c")
            if cols[3].button("🗑", key=f"del-{doc['document_id']}"):
                st.session_state.confirm_delete_id = doc["document_id"]
                st.session_state.confirm_delete_name = doc["filename"]

        # Delete confirmation dialog
        if st.session_state.confirm_delete_id is not None:
            st.warning(
                f"Are you sure you want to delete **{st.session_state.confirm_delete_name}**? "
                "This cannot be undone."
            )
            confirm_cols = st.columns(2)
            if confirm_cols[0].button("Confirm", key="confirm-delete-yes"):
                try:
                    resp = api_delete(f"/documents/{st.session_state.confirm_delete_id}")
                    if resp.status_code == 200:
                        refresh_documents()
                    else:
                        st.error("Delete failed")
                except requests.RequestException as e:
                    st.error(f"Cannot reach the API: {e}")
                st.session_state.confirm_delete_id = None
                st.session_state.confirm_delete_name = None
                st.rerun()
            if confirm_cols[1].button("Cancel", key="confirm-delete-no"):
                st.session_state.confirm_delete_id = None
                st.session_state.confirm_delete_name = None
                st.rerun()
    else:
        st.caption("No documents ingested yet.")

    # ── Sidebar — Conversation History ────────────────────────────────────

    st.divider()
    st.header("💬 Conversation")

    history = st.session_state.conversation_history
    for i in range(0, len(history), 2):
        if i < len(history):
            st.caption(f"Q: {history[i]['content'][:80]}...")
        if i + 1 < len(history):
            st.caption(f"A: {history[i + 1]['content'][:100]}...")

    if st.button("Clear conversation"):
        st.session_state.confirm_clear_conversation = True

    # Clear conversation confirmation dialog
    if st.session_state.confirm_clear_conversation:
        st.warning("Clear the entire conversation history?")
        clear_cols = st.columns(2)
        if clear_cols[0].button("Confirm", key="confirm-clear-yes"):
            st.session_state.conversation_history = []
            st.session_state.last_response = None
            st.session_state.confirm_clear_conversation = False
            st.rerun()
        if clear_cols[1].button("Cancel", key="confirm-clear-no"):
            st.session_state.confirm_clear_conversation = False
            st.rerun()


# ── Main — Query interface ───────────────────────────────────────────────────

st.title("🔍 Ask a Question")

question = st.text_input("Question", placeholder="What was the total revenue in 2024?")

doc_names = ["All documents"] + [d["filename"] for d in st.session_state.documents]
scope = st.selectbox("Scope", doc_names)

if st.button("Ask", type="primary") and question:
    payload: dict = {
        "question": question,
        "conversation_history": st.session_state.conversation_history,
    }
    if scope != "All documents":
        doc = next(
            (d for d in st.session_state.documents if d["filename"] == scope),
            None,
        )
        if doc:
            payload["filters"] = {"document_id": doc["document_id"]}

    with st.spinner("Thinking..."):
        try:
            resp = api_post_json("/query", payload)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.conversation_history = data["conversation_history"]
                st.session_state.last_response = data
            else:
                detail = resp.json().get("detail", "Query failed")
                st.error(detail)
        except requests.RequestException as e:
            st.error(f"Cannot reach the API: {e}")


# ── Main — Answer display ───────────────────────────────────────────────────

data = st.session_state.last_response
if data:
    st.markdown(data["answer"])

    conf = data.get("confidence", "low")
    if conf == "high":
        st.success("Confidence: ●●● High")
    elif conf == "medium":
        st.warning("Confidence: ●● Medium")
    else:
        st.error("Confidence: ● Low")

    if not data["found"]:
        st.info("No direct answer found. Try rephrasing your question.")
    else:
        st.subheader("Sources")
        for src in data["sources"]:
            label = f"📄 {src['filename']}, page {src['page']} (score: {src['score']})"
            with st.expander(label):
                if src.get("highlight"):
                    st.markdown(f"> **{src['highlight']}**")
                st.text(src["text"])
