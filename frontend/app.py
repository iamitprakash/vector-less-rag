import streamlit as st
import requests
import json

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Enterprise Vectorless RAG", layout="wide")

st.title("🏛 Enterprise Vectorless RAG")
st.markdown("Full-featured RAG utilizing Local Ollama, SQLite Persistence, and Asynchronous Reasoning.")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Document Management
with st.sidebar:
    st.header("📂 Document Management")
    
    # Upload
    uploaded_file = st.file_uploader("Upload PDF to Server", type="pdf")
    if uploaded_file:
        if st.button("Processing & Store"):
            with st.spinner("Extracting and storing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{BASE_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("Document stored successfully!")
                else:
                    st.error(f"Failed to upload: {response.text}")

    st.divider()
    
    # List Documents
    st.subheader("Stored Documents")
    if st.button("Refresh List"):
        docs_res = requests.get(f"{BASE_URL}/documents")
        if docs_res.status_code == 200:
            st.session_state.docs = docs_res.json()
        else:
            st.error("Could not fetch documents.")
            
    if "docs" in st.session_state:
        for doc in st.session_state.docs:
            st.text(f"📄 {doc['filename']} (ID: {doc['id']})")
    
    st.divider()
    st.info("Backend: FastAPI | Engine: Ollama (Async)")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the enterprise library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Querying Enterprise API...", expanded=True) as status:
            st.write("🛰 Sending request to FastAPI...")
            payload = {"query": prompt}
            
            try:
                response = requests.post(f"{BASE_URL}/query", json=payload, stream=True)
                
                if response.status_code == 200:
                    status.update(label="Retrieval successful, generating answer...", state="running")
                    
                    # Streaming display
                    def stream_viewer():
                        full_content = ""
                        sources_raw = None
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if "SOURCES_METADATA:" in chunk:
                                parts = chunk.split("SOURCES_METADATA:")
                                yield parts[0]
                                sources_raw = parts[1]
                                break
                            yield chunk
                        
                        if sources_raw:
                            st.session_state.last_sources = json.loads(sources_raw)["sources"]
                    
                    answer = st.write_stream(stream_viewer())
                    status.update(label="Complete!", state="complete", expanded=False)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    if "last_sources" in st.session_state:
                        with st.expander("Enterprise Sources"):
                            for s in st.session_state.last_sources:
                                st.markdown(f"- **{s['source']}** (Page {s['page']})")
                else:
                    status.update(label="API Error", state="error", expanded=False)
                    st.error(f"Error: {response.text}")
            except Exception as e:
                status.update(label="Connection Error", state="error", expanded=False)
                st.error(f"Could not connect to backend: {e}")
