from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import httpx
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import json
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# Import RAG components only when needed
sentence_transformers = None
faiss = None
PyPDF2 = None
docx = None
tiktoken = None

app = FastAPI(
    title="AiNeuralChat - Real RAG System"
)

# Serve frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def lazy_import_rag():
    """Lazy import of RAG dependencies"""
    global sentence_transformers, faiss, PyPDF2, docx, tiktoken
    
    if sentence_transformers is None:
        print("🔄 Loading RAG components...")
        import sentence_transformers as st
        import faiss as f
        import PyPDF2 as pdf
        from docx import Document
        import tiktoken as tk
        
        sentence_transformers = st
        faiss = f
        PyPDF2 = pdf
        docx = Document
        tiktoken = tk
        print("✅ RAG components loaded!")

load_dotenv()

app = FastAPI(title="AiNeuralChat - Real RAG System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 3000

# Global variables
embedding_model = None
vector_store = None
documents: Dict[str, Dict] = {}
chunks: Dict[str, Dict] = {}

class RAGSystem:
    def __init__(self):
        self.initialized = False
        
    def initialize(self):
        """Initialize RAG system components"""
        if self.initialized:
            return
            
        lazy_import_rag()
        
        print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL}")
        global embedding_model, vector_store
        
        embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS
        vector_store = faiss.IndexFlatIP(embedding_dim)
        vector_store = faiss.IndexIDMap(vector_store)
        
        self.initialized = True
        print(f"✅ RAG system initialized! Embedding dim: {embedding_dim}")
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    def extract_text_from_file(self, content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        try:
            if filename.lower().endswith('.txt'):
                return content.decode('utf-8', errors='ignore')
            
            elif filename.lower().endswith('.pdf'):
                import io
                pdf_file = io.BytesIO(content)
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
            elif filename.lower().endswith(('.doc', '.docx')): 
                import io
                doc_file = io.BytesIO(content)
                doc = docx(doc_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
                
            else:
                return content.decode('utf-8', errors='ignore')
                
        except Exception as e:
            print(f"❌ Error extracting text from {filename}: {e}")
            return f"Error extracting text from {filename}: {str(e)}"
    
    def add_document(self, doc_id: str, filename: str, content: bytes) -> Dict:
        """Add document to RAG system"""
        if not self.initialized:
            self.initialize()
            
        print(f"📄 Processing document: {filename}")
        
        # Extract text
        text = self.extract_text_from_file(content, filename)
        
        if not text.strip():
            raise ValueError("Could not extract text from document")
        
        # Create chunks
        text_chunks = self.chunk_text(text)
        print(f"📦 Created {len(text_chunks)} chunks")
        
        # Generate embeddings
        chunk_embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
        faiss.normalize_L2(chunk_embeddings)
        
        # Store document
        documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "content": text,
            "chunk_count": len(text_chunks),
            "uploaded_at": datetime.now().isoformat(),
            "size": len(content)
        }
        
        # Add chunks to vector store
        chunk_ids = []
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Store chunk metadata
            chunks[chunk_id] = {
                "id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "text": chunk_text,
                "chunk_index": i
            }
            
            # Add to FAISS index
            vector_store.add_with_ids(
                embedding.reshape(1, -1), 
                np.array([hash(chunk_id) % (2**31)])
            )
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": len(text_chunks)
        }
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        if not self.initialized or vector_store.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in vector store
        scores, indices = vector_store.search(query_embedding, min(top_k, vector_store.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            # Find matching chunk
            for chunk_id, chunk_data in chunks.items():
                if hash(chunk_id) % (2**31) == idx:
                    results.append({
                        "chunk_id": chunk_id,
                        "doc_id": chunk_data["doc_id"],
                        "filename": chunk_data["filename"],
                        "text": chunk_data["text"],
                        "similarity_score": float(score),
                        "chunk_index": chunk_data["chunk_index"]
                    })
                    break
        
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

# Initialize RAG system
rag_system = RAGSystem()

# Groq AI Integration
async def chat_with_groq(message: str, context_chunks: List[Dict] = None) -> str:
    """Chat with Groq AI using RAG context"""
    
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key or not groq_key.startswith("gsk_"):
        return handle_without_api(message, context_chunks)
    
    try:
        # Build context from RAG results
        context_text = ""
        if context_chunks:
            context_text = "\n\nRelevant document context:\n"
            for chunk in context_chunks:
                context_text += f"\nFrom '{chunk['filename']}':\n{chunk['text']}\n"
                context_text += f"(Similarity: {chunk['similarity_score']:.3f})\n"
        
        # Create the full message
        full_message = message
        if context_text:
            full_message = f"{message}\n{context_text}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are AiNeuralChat, a helpful AI assistant with access to document knowledge. When provided with document context, use it to give accurate and detailed answers. Always cite the source documents when referencing them."
                        },
                        {"role": "user", "content": full_message}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return handle_without_api(message, context_chunks)
                
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return handle_without_api(message, context_chunks)

def handle_without_api(message: str, context_chunks: List[Dict] = None) -> str:
    """Handle chat without API key"""
    if context_chunks:
        response = "Based on your uploaded documents:\n\n"
        for chunk in context_chunks:
            response += f"From '{chunk['filename']}':\n{chunk['text'][:200]}...\n\n"
        response += "💡 Add your FREE Groq API key to get AI-powered analysis!"
        return response
    
    return f"You asked: '{message}'\n\n🤖 Add your FREE Groq API key to enable full AI chat!"

# API Routes
# @app.get("/")
# async def root():
#     groq_status = "✅ Connected" if os.getenv("GROQ_API_KEY", "").startswith("gsk_") else "❌ Not configured"
#     return {
#         "message": "AiNeuralChat - Real RAG System",
#         "groq_status": groq_status,
#         "documents": len(documents),
#         "chunks": len(chunks),
#         "rag_initialized": rag_system.initialized,
#         "docs": "/docs"
#     }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "groq_ai": bool(os.getenv("GROQ_API_KEY", "").startswith("gsk_")),
        "rag_system": "active" if rag_system.initialized else "ready",
        "documents": len(documents),
        "chunks": len(chunks)
    }

# Documents
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        doc_id = str(uuid.uuid4())
        
        # Process with RAG system
        result = rag_system.add_document(doc_id, file.filename, content)
        
        return {
            "id": doc_id,
            "filename": file.filename,
            "message": f"✅ '{file.filename}' processed! Created {result['chunks_created']} searchable chunks.",
            "chunks_created": result["chunks_created"]
        }
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

@app.get("/api/documents/")
async def list_documents():
    return [
        {
            "id": doc["id"],
            "filename": doc["filename"],
            "uploaded_at": doc["uploaded_at"],
            "chunk_count": doc["chunk_count"],
            "size": doc["size"]
        }
        for doc in documents.values()
    ]

# RAG Chat
@app.post("/api/chat/send")
async def send_message(message_data: dict):
    content = message_data.get("content", "")
    
    print(f"💬 Processing query: {content[:100]}...")
    
    # Perform RAG search
    relevant_chunks = rag_system.search_similar_chunks(content, top_k=3)
    print(f"🔍 Found {len(relevant_chunks)} relevant chunks")
    
    # Get AI response with RAG context
    ai_response = await chat_with_groq(content, relevant_chunks)
    
    # Create response
    response_msg = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat(),
        "rag_sources": [
            {
                "filename": chunk["filename"],
                "similarity": chunk["similarity_score"],
                "chunk_preview": chunk["text"][:100] + "..."
            }
            for chunk in relevant_chunks
        ] if relevant_chunks else None
    }
    
    return {
        "conversation_id": str(uuid.uuid4()),
        "message": response_msg
    }

@app.get("/api/rag/stats")
async def rag_stats():
    return {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "rag_initialized": rag_system.initialized,
        "embedding_model": EMBEDDING_MODEL
    }

if __name__ == "__main__":
    print("🚀 AiNeuralChat - Real RAG System")
    print("🧠 Advanced document understanding with vector embeddings")
    print("⚡ Fast startup with lazy loading")
    print("")
    
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key.startswith("gsk_"):
        print("✅ Groq AI: CONNECTED")
    else:
        print("❌ Groq AI: NOT CONFIGURED")
    
    print("")
    print("🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
