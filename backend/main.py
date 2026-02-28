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
        
        # Generate summary
        db_doc.summary = await engine.summarize_document(file.filename, pages_content)
        
        db.commit()
        db.refresh(db_doc)
        return db_doc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents", response_model=list[DocumentSchema])
async def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    # Step 1: Select documents (if not specified)
    if not request.document_ids:
        selected_docs = await engine.select_documents(db, request.query)
    else:
        selected_docs = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
    
    if not selected_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    # Step 2: Detect relevant pages
    relevant_pages = await engine.detect_relevant_pages(request.query, selected_docs)
    
    # Step 3: Generate answer
    answer = await engine.generate_answer(request.query, relevant_pages)
    
    return QueryResponse(
        answer=answer,
        sources=[{"source": p["source"], "page": p["page"]} for p in relevant_pages]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
