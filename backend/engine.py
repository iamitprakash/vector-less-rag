import asyncio
import ollama
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .database import Document, Page

class AsyncVectorlessEngine:
    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model

    async def summarize_document(self, filename: str, pages_content: List[str]) -> str:
        """Generates a brief summary of the document based on its first few pages."""
        # Use first 3 pages for context (or fewer if doc is short)
        context = "\n\n".join(pages_content[:3])
        prompt = f"""
        Document Filename: {filename}
        
        Content (first few pages):
        {context[:4000]}
        
        Generate a 2-sentence summary of what this document is about. 
        Focus on the main topic and purpose.
        """
        return await self.chat(prompt)

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
        
        doc_list = "\n".join([f"ID: {doc.id} | Name: {doc.filename} | Summary: {doc.summary or 'No summary'}" for doc in documents])
        system_prompt = "You are a specialized retrieval assistant. Your ONLY job is to return a comma-separated list of document IDs. DO NOT provide any explanation or conversational text."
        prompt = f"""
        Select the most relevant document IDs for the query.
        
        Documents:
        {doc_list}
        
        Query: "{query}"
        
        Return only IDs or "None":
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
        """Step 2: Scans pages in batches to reduce LLM calls."""
        all_relevant_pages = []
        semaphore = asyncio.Semaphore(2) # Stricter limit for batching to prevent timeouts
        
        async def process_batch(doc_name: str, batch: List[Page]):
            batch_text = ""
            for p in batch:
                batch_text += f"\n--- PAGE {p.page_number} ---\n{p.content[:1000]}\n"
            
            system_prompt = "You are a page detection assistant. Identify which specific page numbers contain relevant information for the query."
            prompt = f"""
            Document: {doc_name}
            Batch Content:
            {batch_text}
            
            Query: "{query}"
            
            Which page numbers in this batch are relevant? 
            Return a comma-separated list of page numbers or "None". 
            DO NOT explain.
            """
            
            async with semaphore:
                content = await self.chat(prompt, system=system_prompt)
            
            if content.lower() == "none" or not content:
                return []
            
            import re
            found_pages = [int(match) for match in re.findall(r'\b\d+\b', content)]
            
            relevant = []
            for p_num in found_pages:
                page_obj = next((p for p in batch if p.page_number == p_num), None)
                if page_obj:
                    relevant.append({
                        "source": doc_name,
                        "page": page_obj.page_number,
                        "content": page_obj.content
                    })
            return relevant

        tasks = []
        batch_size = 3
        for doc in selected_docs:
            for i in range(0, len(doc.pages), batch_size):
                batch = doc.pages[i:i + batch_size]
                tasks.append(process_batch(doc.filename, batch))
        
        results = await asyncio.gather(*tasks)
        for r_list in results:
            all_relevant_pages.extend(r_list)
            
        return all_relevant_pages

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
