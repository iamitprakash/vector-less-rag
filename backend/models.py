import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PageBase(BaseModel):
    page_number: int
    content: str

class PageCreate(PageBase):
    pass

class PageSchema(PageBase):
    id: int
    document_id: int

    class Config:
        from_attributes = True

class DocumentBase(BaseModel):
    filename: str
    description: Optional[str] = None
    summary: Optional[str] = None
    metadata_json: Optional[str] = None

class DocumentCreate(DocumentBase):
    pass

class PageSchema(BaseModel):
    id: int
    document_id: int
    page_number: int
    content: str

    class Config:
        from_attributes = True

class ChunkSchema(BaseModel):
    id: int
    document_id: int
    content: str
    page_range: str

    class Config:
        from_attributes = True

class ChatMessageSchema(BaseModel):
    role: str
    content: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class DocumentSchema(DocumentBase):
    id: int
    pages: List[PageSchema] = []

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    session_id: Optional[str] = None # Added for Phase 4
    comparison_mode: bool = False # Added for Phase 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
