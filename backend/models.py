from pydantic import BaseModel
from typing import List, Optional

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

class DocumentCreate(DocumentBase):
    pass

class DocumentSchema(DocumentBase):
    id: int
    pages: List[PageSchema] = []

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
