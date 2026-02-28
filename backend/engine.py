import asyncio
import ollama
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .database import Document, Page

class AsyncVectorlessEngine:
    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model

    async def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Async wrapper for ollama.chat with optional system prompt"""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(model=self.model, messages=messages)
        )
        return response['message']['content'].strip()

    async def select_documents(self, db: Session, query: str) -> List[Document]:
        """Step 1: LLM selects relevant documents."""
        documents = db.query(Document).all()
        if not documents:
            return []
        
        # If only one document exists, we might want to just pick it, 
        # but let's let the LLM decide for consistency.
        
        doc_list = "\n".join([f"ID: {doc.id} - {doc.filename}" for doc in documents])
        system_prompt = "You are a specialized retrieval assistant. Your ONLY job is to return a comma-separated list of document IDs. DO NOT provide any explanation or conversational text."
        prompt = f"""
        Documents available:
        {doc_list}
        
        Query: "{query}"
        
        Which IDs are relevant? (Return only IDs or "None")
        """
        
        content = await self.chat(prompt, system=system_prompt)
        if content.lower() == "none":
            return []
        
        # Robust parsing: extract all integers from the response
        import re
        found_ids = [int(match) for match in re.findall(r'\b\d+\b', content)]
        
        if not found_ids:
            return []
            
        return db.query(Document).filter(Document.id.in_(found_ids)).all()

    async def check_page_relevance(self, query: str, doc_name: str, page_num: int, page_content: str) -> Optional[Dict[str, Any]]:
        """Checks if a single page is relevant."""
        prompt = f"""
        Document: {doc_name} (Page {page_num})
        Content Snippet: {page_content[:1500]}
        
        Question: {query}
        
        Is this page relevant to answering the question? Answer with 'YES' or 'NO'.
        """
        
        content = await self.chat(prompt)
        if "YES" in content.upper():
            return {
                "source": doc_name,
                "page": page_num,
                "content": page_content
            }
        return None

    async def detect_relevant_pages(self, query: str, selected_docs: List[Document]) -> List[Dict[str, Any]]:
        """Step 2: Scans pages in parallel."""
        tasks = []
        for doc in selected_docs:
            for page in doc.pages:
                tasks.append(self.check_page_relevance(query, doc.filename, page.page_number, page.content))
        
        # Limit concurrency to avoid overloading local Ollama
        semaphore = asyncio.Semaphore(5)
        
        async def sem_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[sem_task(t) for t in tasks])
        return [r for r in results if r is not None]

    async def generate_answer(self, query: str, context_pages: List[Dict[str, Any]]) -> str:
        """Step 3: Generate final answer."""
        if not context_pages:
            return "I couldn't find any relevant information in the uploaded documents."
        
        context_text = ""
        for p in context_pages:
            context_text += f"\n--- Source: {p['source']} (Page {p['page']}) ---\n{p['content']}\n"
        
        prompt = f"""
        You are an enterprise AI assistant. Use the following context to answer the user's question with citations.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Answer:
        """
        
        return await self.chat(prompt)
