import asyncio
import ollama
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .database import Document, Page, Chunk, ChatMessage

class AsyncVectorlessEngine:
    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model

    async def summarize_document(self, filename: str, pages_content: List[str]) -> str:
        """Generates a brief summary of the document based on its first few pages."""
        try:
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
        except:
            return "Summary generation failed."

    async def extract_metadata(self, filename: str, pages_content: List[str]) -> str:
        """Extracts key entities, dates, and topics for faster filtering."""
        try:
            context = "\n\n".join(pages_content[:3]) # Use first 3 pages
            system_prompt = """
            You are a metadata extraction assistant. 
            Identify key entities (names, company, projects, tech stack) and any date ranges mentioned.
            Return JSON: {"entities": ["A", "B"], "dates": ["2024", "Jan 2023"], "topics": ["C", "D"]}
            """
            prompt = f"Document: {filename}\nContent:\n{context[:4000]}"
            return await self.chat(prompt, system=system_prompt, json_mode=True)
        except:
            return "{}"

    async def chunk_document(self, filename: str, pages: List[Page]) -> List[Dict[str, Any]]:
        """Phase 4: Semantic chunking with overlap."""
        chunks = []
        full_text = ""
        page_map = [] # To track which character belongs to which page
        
        current_pos = 0
        for p in pages:
            full_text += p.content + "\n"
            page_map.append((current_pos, current_pos + len(p.content) + 1, p.page_number))
            current_pos += len(p.content) + 1
            
        # Chunking: ~1500 chars with 300 char overlap
        chunk_size = 1500
        overlap = 300
        
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            content = full_text[start:end]
            
            # Find pages involved in this chunk
            pages_involved = set()
            for p_start, p_end, p_num in page_map:
                if not (end <= p_start or start >= p_end):
                    pages_involved.add(p_num)
            
            p_min = min(pages_involved) if pages_involved else 0
            p_max = max(pages_involved) if pages_involved else 0
            page_range = f"{p_min}-{p_max}"
            
            chunks.append({
                "content": content,
                "page_range": page_range
            })
            
            if end >= len(full_text):
                break
            start += (chunk_size - overlap)
            
        return chunks

    async def refine_query(self, query: str, history: List[ChatMessage]) -> str:
        """Phase 4: Resolves pronouns and context using chat history."""
        if not history:
            return query
            
        history_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in history[-5:]]) # Last 5 messages
        
        system_prompt = """
        You are a query refinement assistant. 
        Based on the chat history, rewrite the user's latest query to be a standalone search query.
        Resolve pronouns (he, she, it, they, that project, then) to their actual entities.
        IF the query is already standalone, return it as is.
        Return JSON: {"standalone_query": "..."}
        """
        prompt = f"History:\n{history_text}\n\nLatest Query: {query}"
        
        try:
            content = await self.chat(prompt, system=system_prompt, json_mode=True)
            import json
            return json.loads(content).get("standalone_query", query)
        except:
            return query

    async def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False, stream: bool = False) -> Any:
        """Async wrapper for ollama.chat with optional system prompt, JSON mode, and streaming"""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        
        options = {"format": "json"} if json_mode else {}
        
        if stream:
            # We need to run the iterator in a thread but consume it as an async generator
            def sync_stream():
                return ollama.chat(model=self.model, messages=messages, stream=True, **options)
            
            loop = asyncio.get_event_loop()
            sync_gen = await loop.run_in_executor(None, sync_stream)
            
            async def async_gen():
                for chunk in sync_gen:
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield chunk['message']['content']
            return async_gen()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(model=self.model, messages=messages, **options)
        )
        return response['message']['content'].strip()

    async def expand_query(self, query: str) -> List[str]:
        """Generates 2-3 variations of the query to broaden search coverage."""
        system_prompt = """
        You are a search query expansion assistant. 
        Generate 2-3 alternative search queries based on the user's input.
        Focus on synonyms and related technical concepts.
        Return JSON: {"variations": ["query 1", "query 2", ...]}
        """
        prompt = f"Original Query: {query}"
        
        try:
            content = await self.chat(prompt, system=system_prompt, json_mode=True)
            import json
            data = json.loads(content)
            variations = data.get("variations", [])
            # Include original query
            if query not in variations:
                variations.append(query)
            return variations[:4] # Limit to 4 total
        except:
            return [query]

    async def select_documents(self, db: Session, query: str) -> List[Document]:
        """Step 1: LLM selects relevant documents."""
        documents = db.query(Document).all()
        if not documents:
            return []
        
        # If only one document exists, we might want to just pick it, 
        # but let's let the LLM decide for consistency.
        
        doc_list = "\n".join([f"ID: {doc.id} | Name: {doc.filename} | Summary: {doc.summary or 'No summary'}" for doc in documents])
        system_prompt = """
        You are a specialized retrieval assistant. 
        Your ONLY job is to return a JSON object containing a list of relevant document IDs.
        Format: {"relevant_ids": [1, 2, ...]}
        If none are relevant, return {"relevant_ids": []}
        """
        prompt = f"""
        Select relevant document IDs for the query.
        
        Documents:
        {doc_list}
        
        Query: "{query}"
        """
        
        content = await self.chat(prompt, system=system_prompt, json_mode=True)
        
        try:
            import json
            data = json.loads(content)
            found_ids = data.get("relevant_ids", [])
            if not found_ids:
                return []
            return db.query(Document).filter(Document.id.in_(found_ids)).all()
        except:
            # Fallback to robust regex if JSON fails or structure is wrong
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
        """Step 2: Hierarchical scanning to reduce LLM calls for large docs."""
        all_relevant_pages = []
        semaphore = asyncio.Semaphore(2)
        
        async def fast_pass(doc_name: str, batch: List[Page]):
            """Pass 1: Quick check over a larger batch (10 pages)"""
            batch_text = ""
            for p in batch:
                batch_text += f"\n[P{p.page_number}]: {p.content[:500]}\n"
            
            system_prompt = """
            You are a fast-pass retrieval assistant. Identify page numbers that are relevant to the query.
            If the query is broad (e.g. "summarize", "about this", "depth"), pick pages that look like introductions, tables of contents, or overviews.
            Return JSON: {"potential_pages": [1, 5, 10, ...]}
            If none, return {"potential_pages": []}
            """
            prompt = f"Query: {query}\n\nBatch Content:\n{batch_text}"
            
            async with semaphore:
                content = await self.chat(prompt, system=system_prompt, json_mode=True)
            
            try:
                import json
                return json.loads(content).get("potential_pages", [])
            except:
                return []

        async def deep_scan(doc_name: str, batch: List[Page]):
            """Pass 2: Detailed check on promising pages."""
            batch_text = ""
            for p in batch:
                batch_text += f"\n--- PAGE {p.page_number} ---\n{p.content[:1500]}\n"
            
            system_prompt = """
            Verify if these specific pages contain the answer OR provide a good overview for a summary request.
            Return JSON: {"relevant_pages": [1, 2, ...]}
            """
            prompt = f"Query: {query}\n\nPromising Pages:\n{batch_text}"
            
            async with semaphore:
                content = await self.chat(prompt, system=system_prompt, json_mode=True)
            
            try:
                import json
                relevant_nums = json.loads(content).get("relevant_pages", [])
                return [p for p in batch if p.page_number in relevant_nums]
            except:
                return []

        for doc in selected_docs:
            # Step 2a: Fast Pass (10 pages per batch)
            potential_page_nums = []
            fast_tasks = []
            batch_size_fast = 10
            # Sort pages to ensure order
            sorted_pages = sorted(doc.pages, key=lambda x: x.page_number)
            for i in range(0, len(sorted_pages), batch_size_fast):
                batch = sorted_pages[i:i + batch_size_fast]
                fast_tasks.append(fast_pass(doc.filename, batch))
            
            results = await asyncio.gather(*fast_tasks)
            for r in results:
                potential_page_nums.extend(r)
            
            # Phase 5 Fallback: If no pages found but query is broad, take first 5 pages
            is_broad = any(word in query.lower() for word in ["summarize", "about", "depth", "overview", "who is", "what is"])
            if not potential_page_nums and is_broad:
                potential_page_nums = [p.page_number for p in sorted_pages[:5]]
            
            if not potential_page_nums:
                continue
                
            # Step 2b: Deep Scan (3 promising pages per batch)
            promising_pages = [p for p in sorted_pages if p.page_number in potential_page_nums]
            deep_tasks = []
            batch_size_deep = 3
            for i in range(0, len(promising_pages), batch_size_deep):
                batch = promising_pages[i:i + batch_size_deep]
                deep_tasks.append(deep_scan(doc.filename, batch))
            
            deep_results = await asyncio.gather(*deep_tasks)
            for r_list in deep_results:
                for p_obj in r_list:
                    all_relevant_pages.append({
                        "source": doc.filename,
                        "page": p_obj.page_number,
                        "content": p_obj.content,
                        "doc_summary": doc.summary # Pass summary for extra context
                    })
            
        return all_relevant_pages

    async def generate_answer(self, query: str, context_pages: List[Dict[str, Any]], stream: bool = False) -> Any:
        """Step 3: Generate final answer (item support for streaming)."""
        if not context_pages:
            error_msg = "I couldn't find any relevant information in the uploaded documents."
            if stream:
                async def empty_gen(): yield error_msg
                return empty_gen()
            return error_msg
        
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
        
        return await self.chat(prompt, stream=stream)

    async def synthesize_comparison(self, query: str, context_pages: List[Dict[str, Any]], stream: bool = False) -> Any:
        """Phase 5: Generates a comparative synthesis across multiple documents."""
        if not context_pages:
            return "I couldn't find enough information to perform a comparison."
            
        context_text = ""
        for p in context_pages:
            context_text += f"\n--- Source: {p['source']} (Page {p['page']}) ---\n{p['content']}\n"
            
        system_prompt = """
        You are a Document Synthesis Expert. 
        Your goal is to compare and contrast information across multiple documents.
        1. Create a clear Markdown table for side-by-side comparisons where possible.
        2. Summarize key differences and similarities.
        3. ALWAYS cite your sources using [DocName, Page X] format.
        4. Be objective and professional.
        """
        
        prompt = f"""
        User Query: {query}
        
        Context from various documents:
        {context_text}
        
        Provide a comprehensive synthesis and comparison based on the query.
        """
        return await self.chat(prompt, system=system_prompt, stream=stream)

    async def get_cached_query(self, db: Session, query: str) -> Optional[Dict[str, Any]]:
        """Checks if a normalized version of the query exists in cache."""
        from .database import QueryCache
        import json
        
        normalized_query = query.strip().lower()
        cached = db.query(QueryCache).filter(QueryCache.query == normalized_query).first()
        
        if cached:
            return {
                "answer": cached.answer,
                "sources": json.loads(cached.sources)
            }
        return None

    async def cache_query(self, db: Session, query: str, answer: str, sources: List[Dict[str, Any]]):
        """Stores a query result in the persistent cache."""
        from .database import QueryCache
        import json
        
        normalized_query = query.strip().lower()
        new_cache = QueryCache(
            query=normalized_query,
            answer=answer,
            sources=json.dumps(sources)
        )
        db.add(new_cache)
        db.commit()
