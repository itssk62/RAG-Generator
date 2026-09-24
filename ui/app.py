"""Streamlit front end: manage knowledge bases, upload documents, ask grounded questions."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st

from components import render_answer

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ACTIVE = {"queued", "processing"}

st.set_page_config(page_title="RAG Generator", layout="wide")


def api(method: str, path: str, **kw) -> httpx.Response:
    return httpx.request(method, f"{API}{path}", timeout=60, **kw)


def error_text(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return detail if isinstance(detail, str) else json.dumps(detail)


def sse(path: str, payload: dict) -> Iterator[tuple[str, dict]]:
    event, data = None, []
    with httpx.stream("POST", f"{API}{path}", json=payload, timeout=httpx.Timeout(120, connect=10)) as r:
        if r.status_code != 200:
            r.read()
            yield "error", {"detail": error_text(r)}
            return
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and event:
                yield event, json.loads("\n".join(data) or "{}")
                event, data = None, []


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("RAG Generator")
    try:
        kbs = api("GET", "/kbs").json()
        formats = api("GET", "/config/formats").json()
        health = api("GET", "/health").json()
    except httpx.HTTPError as e:
        st.error(f"API not reachable at {API}: {type(e).__name__}")
        st.stop()
    if health["llm"]["missing_keys"]:
        st.warning(f"LLM `{health['llm']['model']}` has no key set ({', '.join(health['llm']['missing_keys'])}); answers will fail until it is added to .env.")

    names = {kb["id"]: kb["name"] for kb in kbs}
    kb_id = st.selectbox("Knowledge base", list(names), format_func=names.get, index=0 if kbs else None, placeholder="Create one below")

    with st.form("create_kb", clear_on_submit=True):
        st.subheader("New knowledge base")
        name = st.text_input("Name")
        description = st.text_input("Description")
        if st.form_submit_button("Create") and name.strip():
            r = api("POST", "/kbs", json={"name": name, "description": description})
            if r.status_code == 201:
                st.rerun()
            st.error(error_text(r))

    if kb_id:
        with st.expander("Danger zone"):
            if st.checkbox(f"Yes, delete '{names[kb_id]}' and all its documents") and st.button("Delete knowledge base"):
                api("DELETE", f"/kbs/{kb_id}")
                st.session_state.pop(f"chat_{kb_id}", None)
                st.rerun()

if not kb_id:
    st.info("Create a knowledge base in the sidebar to get started.")
    st.stop()

ask_tab, docs_tab = st.tabs(["Ask", "Documents"])

# ---------------------------------------------------------------- documents
with docs_tab:
    exts = [e.lstrip(".") for e in formats["extensions"]]
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(f"Upload ({', '.join(exts)}; max {formats['max_upload_mb']} MB each)", type=exts, accept_multiple_files=True)
        if st.form_submit_button("Upload and index") and files:
            r = api("POST", f"/kbs/{kb_id}/documents", files=[("files", (f.name, f.getvalue())) for f in files])
            if r.status_code == 202:
                st.success(f"Queued {len(r.json())} document(s) for indexing.")
            else:
                st.error(error_text(r))

    @st.fragment(run_every=2)
    def documents() -> None:
        docs = api("GET", f"/kbs/{kb_id}/documents").json()
        if not docs:
            st.caption("No documents yet.")
            return
        for d in docs:
            cols = st.columns([4, 2, 1, 1, 1])
            cols[0].markdown(f"**{d['filename']}**  \n{d['size_bytes'] // 1024} KB")
            color = {"ready": "green", "failed": "red"}.get(d["status"], "blue")
            cols[1].markdown(f":{color}[{d['status']}]" + (f"  \n{d['chunk_count']} chunks" if d["status"] == "ready" else ""))
            if d["error"]:
                cols[1].caption(d["error"])
            if cols[3].button("Reindex", key=f"re_{d['id']}", disabled=d["status"] in ACTIVE):
                api("POST", f"/kbs/{kb_id}/documents/{d['id']}/reindex")
            if cols[4].button("Delete", key=f"del_{d['id']}"):
                api("DELETE", f"/kbs/{kb_id}/documents/{d['id']}")
                st.rerun(scope="fragment")

    documents()

# ---------------------------------------------------------------- ask
with ask_tab:
    history = st.session_state.setdefault(f"chat_{kb_id}", [])
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["answer"], turn["final"])

    question = st.chat_input(f"Ask {names[kb_id]} a question")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            box = st.empty()
            text, final = "", None
            for event, data in sse(f"/kbs/{kb_id}/query/stream", {"question": question}):
                if event == "token":
                    text += data["text"]
                    box.markdown(text + " \u258c")
                elif event == "done":
                    final = data
                elif event == "error":
                    box.error(data["detail"])
            if final is not None:
                box.empty()
                with box.container():
                    render_answer(final["answer"], final)
                history.append({"question": question, "answer": final["answer"], "final": final})
