# 🏛 Enterprise Vectorless RAG

An enterprise-grade, local-first RAG (Retrieval-Augmented Generation) system that uses LLM reasoning instead of vector embeddings. Inspired by `roe-ai/vectorless`.

## 🚀 Key Features

- **Vectorless Reasoning**: Uses local Ollama models (`llama3.1`) to navigate document libraries via "Smart Selection" and "Page Detection" rather than mathematical vector similarity.
- **Enterprise-Ready Architecture**:
  - **Backend**: FastAPI for structured API access.
  - **Persistence**: SQLite (SQLAlchemy) for permanent document storage.
  - **Asynchronous**: Parallel page scanning for faster retrieval.
- **Local-Only**: Zero-data leakage. Everything stays between your PDFs and your local Ollama instance.
- **Stateless frontend**: Clean Streamlit UI that interacts with the API layer.

## 🏗 Project Structure

```text
vector-less/
├── backend/          # FastAPI logic & Database models
├── frontend/         # Streamlit UI
├── data/             # SQLite storage (git-ignored)
└── venv/             # Python environment
```

## 🛠 Setup & Launch

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally with `llama3.1`.

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the System

1. **Initialize Database** (First time only):

   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 -c "from backend.database import init_db; init_db()"
   ```

2. **Start Backend & Frontend**:

   ```bash
   # In one terminal:
   uvicorn backend.main:app --host 0.0.0.0 --port 8000

   # In another:
   streamlit run frontend/app.py
   ```

## 📖 How it Works

1. **Upload**: PDFs are extracted page-by-page and stored in the SQLite database.
2. **Select**: When questioned, the LLM first looks at all document titles to select those most likely to contain the answer.
3. **Detect**: The LLM asynchronously reads the selected document pages to find the exact relevant sections.
4. **Generate**: A final answer is synthesised using the detected pages as context, complete with citations.
