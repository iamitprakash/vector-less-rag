from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
from pypdf import PdfReader
from .database import init_db, get_db, Document, Page
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
    # Step 0: Check Cache
    cached_result = await engine.get_cached_query(db, request.query)
    if cached_result:
        return QueryResponse(
            answer=cached_result["answer"],
            sources=cached_result["sources"]
        )

    # Phase 3 Step 1: Query Expansion
    query_variations = await engine.expand_query(request.query)
    
    # Step 2: Select documents (using all variations)
    all_selected_docs = []
    for q in query_variations:
        docs = await engine.select_documents(db, q)
        all_selected_docs.extend(docs)
    
    # Deduplicate docs
    doc_ids = {doc.id for doc in all_selected_docs}
    selected_docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    
    if not selected_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    # Step 3: Detect relevant pages (using original query for precision)
    relevant_pages = await engine.detect_relevant_pages(request.query, selected_docs)
    
    # Step 4: Generate Streaming Answer
    async def stream_generator():
        combined_answer = ""
        stream = await engine.generate_answer(request.query, relevant_pages, stream=True)
        
        async for chunk in stream:
            combined_answer += chunk
            yield chunk
            
        # After stream ends, yield sources for the frontend to parse
        sources_json = json.dumps({"sources": [{"source": p["source"], "page": p["page"]} for p in relevant_pages]})
        yield f"\n\nSOURCES_METADATA:{sources_json}"
        
        # Cache the full result
        await engine.cache_query(db, request.query, combined_answer, relevant_pages)

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
