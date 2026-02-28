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
            response = requests.post(f"{BASE_URL}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                status.update(label="Response received!", state="complete", expanded=False)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                if sources:
                    with st.expander("Enterprise Sources"):
                        for s in sources:
                            st.markdown(f"- **{s['source']}** (Page {s['page']})")
            else:
                status.update(label="API Error", state="error", expanded=False)
                error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
