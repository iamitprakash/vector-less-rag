import uuid
import streamlit as st
import requests
import json

# Session ID for Phase 4
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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
    
    # Phase 5: Comparison Mode
    st.subheader("⚙️ Agent Settings")
    comparison_mode = st.toggle("📊 Comparison Mode (Synthesis)", value=False, help="Analyze multiple documents side-by-side.")
    
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
        # Phase 4: Thought Trace Container
        thought_container = st.expander("🔬 Engine X-Ray (Thought Trace)", expanded=True)
        thought_log = thought_container.empty()
        thoughts = []

        with st.status("Thinking...", expanded=False) as status:
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id,
                "comparison_mode": comparison_mode # Phase 5
            }
            
            try:
                response = requests.post(f"{BASE_URL}/query", json=payload, stream=True)
                
                if response.status_code == 200:
                    status.update(label="Processing...", state="running")
                    
                    # Streaming display
                    def stream_viewer():
                        full_content = ""
                        sources_raw = None
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if "THOUGHT:" in chunk:
                                thought_line = chunk.replace("THOUGHT:", "").strip()
                                thoughts.append(f"- {thought_line}")
                                thought_log.markdown("\n".join(thoughts))
                                continue
                                
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
                    # thought_container.update(expanded=False) # Removed: invalid command
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    if "last_sources" in st.session_state:
                        with st.expander("Enterprise Sources"):
                            for s in st.session_state.last_sources:
                                col1, col2 = st.columns([3, 1])
                                col1.markdown(f"- **{s['source']}** (Page {s['page']})")
                                if col2.button("👁️ Show Preview", key=f"preview_{s['source']}_{s['page']}"):
                                    # Fetch page image (Need doc_id, find it from session state or query backend)
                                    # For simplicity, we search session state docs
                                    doc_id = None
                                    if "docs" in st.session_state:
                                        for d in st.session_state.docs:
                                            if d['filename'] == s['source']:
                                                doc_id = d['id']
                                                break
                                    
                                    if doc_id:
                                        img_res = requests.get(f"{BASE_URL}/page-image/{doc_id}/{s['page']}")
                                        if img_res.status_code == 200:
                                            img_b64 = img_res.json()["image_base64"]
                                            st.image(base64.b64decode(img_b64), caption=f"{s['source']} - Page {s['page']}")
                                        else:
                                            st.error("Could not load preview.")
                                    else:
                                        st.error("Document reference missing.")
                else:
                    status.update(label="API Error", state="error", expanded=False)
                    st.error(f"Error: {response.text}")
            except Exception as e:
                status.update(label="Connection Error", state="error", expanded=False)
                st.error(f"Could not connect to backend: {e}")
