from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

DATABASE_URL = "sqlite:///./data/vectorless.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True) # Store JSON of entities, dates, etc.
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    page_number = Column(Integer)
    content = Column(Text)
    
    document = relationship("Document", back_populates="pages")

class Chunk(Base):
    """Semantic chunks for better cross-page retrieval."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    page_range = Column(String) # e.g., "1-2"
    
    document = relationship("Document", back_populates="chunks")

class QueryCache(Base):
    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True)
    answer = Column(Text)
    sources = Column(Text) # JSON string of sources
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # UUID for the session
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
