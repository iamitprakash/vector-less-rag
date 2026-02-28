from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
from pypdf import PdfReader
from .database import SessionLocal, init_db, get_db, Document, Page, Chunk, ChatMessage
from .models import DocumentSchema, QueryRequest, QueryResponse
from .engine import AsyncVectorlessEngine

app = FastAPI(title="Enterprise Vectorless RAG API")

# Initialize Engine
engine = AsyncVectorlessEngine()

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/upload", response_model=DocumentSchema)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file temporarily
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract text
        reader = PdfReader(temp_path)
        
        # Check if doc already exists
        db_doc = db.query(Document).filter(Document.filename == file.filename).first()
        if db_doc:
            # Delete old pages
            db.query(Page).filter(Page.document_id == db_doc.id).delete()
        else:
            db_doc = Document(filename=file.filename)
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
        
        # Add new pages
        pages_content = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            pages_content.append(content)
            db_page = Page(
                document_id=db_doc.id,
                page_number=i+1,
                content=content
            )
            db.add(db_page)
        
        # Phase 3: Metadata + Summary
        db_doc.summary = await engine.summarize_document(file.filename, pages_content)
        db_doc.metadata_json = await engine.extract_metadata(file.filename, pages_content)
        
        # Phase 4: Semantic Chunking
        chunks_data = await engine.chunk_document(file.filename, [p for p in db.query(Page).filter(Page.document_id == db_doc.id).all()])
        for c in chunks_data:
            db_chunk = Chunk(
                document_id=db_doc.id,
                content=c["content"],
                page_range=c["page_range"]
            )
            db.add(db_chunk)
        
        db.commit()
        db.refresh(db_doc)
        return db_doc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents", response_model=list[DocumentSchema])
async def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

from fastapi.responses import StreamingResponse
import json

@app.post("/query")
async def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    session_id = request.session_id or "default"
    
    # Phase 4: Retrieve Chat History
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    
    # Step 0: Check Cache (Only if no history/refinement needed for speed)
    if not history:
        cached_result = await engine.get_cached_query(db, request.query)
        if cached_result:
            return QueryResponse(
                answer=cached_result["answer"],
                sources=cached_result["sources"]
            )

    async def stream_generator():
        # Phase 4: Query Refinement
        yield "THOUGHT: Analyzing conversation history and refining query...\n"
        refined_query = await engine.refine_query(request.query, history)
        if refined_query != request.query:
            yield f"THOUGHT: Refined query to: '{refined_query}'\n"
        
        # Phase 3 Step 1: Query Expansion
        yield "THOUGHT: Expanding query for better retrieval coverage...\n"
        query_variations = await engine.expand_query(refined_query)
        
        # Step 2: Select documents
        yield "THOUGHT: Selecting most relevant documents...\n"
        all_selected_docs = []
        for q in query_variations:
            docs = await engine.select_documents(db, q)
            all_selected_docs.extend(docs)
        
        doc_ids = {doc.id for doc in all_selected_docs}
        selected_docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
        
        if not selected_docs:
            yield "I couldn't find any relevant documents for your request."
            return

        # Phase 4: Step 3 - Detect relevant pages/chunks
        yield f"THOUGHT: Scanning {len(selected_docs)} documents for relevant sections...\n"
        relevant_pages = await engine.detect_relevant_pages(refined_query, selected_docs)
        
        if not relevant_pages:
            yield "I couldn't find any specific sections that answer your question."
            return

        # Step 4: Generate Streaming Answer
        yield f"THOUGHT: Generating answer from {len(relevant_pages)} relevant sources...\n"
        combined_answer = ""
        stream = await engine.generate_answer(refined_query, relevant_pages, stream=True)
        
        async for chunk in stream:
            combined_answer += chunk
            yield chunk
            
        # After stream ends, yield sources and persist to DB
        sources_json = json.dumps({"sources": [{"source": p["source"], "page": p["page"]} for p in relevant_pages]})
        yield f"\n\nSOURCES_METADATA:{sources_json}"
        
        # Persist to Chat History
        user_msg = ChatMessage(session_id=session_id, role="user", content=request.query)
        ast_msg = ChatMessage(session_id=session_id, role="assistant", content=combined_answer)
        db.add(user_msg)
        db.add(ast_msg)
        db.commit()
        
        # Cache the full result
        await engine.cache_query(db, request.query, combined_answer, relevant_pages)

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
