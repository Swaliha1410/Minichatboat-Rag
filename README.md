<img width="1366" height="641" alt="AiNeuralChat" src="https://github.com/user-attachments/assets/01f39c0d-7ea0-4370-a7cf-191cbac8bff5" /># AiNeuralChat - RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional AI chatbot with **Real RAG (Retrieval-Augmented Generation)** capabilities. Upload documents (PDF, DOCX, TXT) and chat with your data using advanced vector embeddings and semantic search.

![AiNeuralChat Interface]('<img width="1366" height="641" alt="AiNeuralChat" src="https://github.com/user-attachments/assets/2d1235e0-6b36-446b-9ab2-a76050ccfb7e" />
')
 
## ✨ Features

### 🧠 **Real RAG Implementation**
- **Vector Embeddings** using Sentence Transformers
- **FAISS Vector Store** for efficient similarity search
- **Semantic Search** - understands context, not just keywords
- **Smart Chunking** with overlap for better context retention

### 📄 **Document Processing**
- ✅ PDF files (PyPDF2)
- ✅ Word documents (DOCX)
- ✅ Text files (TXT)
- ✅ Advanced text extraction and processing

### 🤖 **AI Integration**
- **Groq AI** integration (FREE and FAST!)
- Context-aware responses using RAG
- Citation of source documents
- Conversational interface

### 🎨 **Professional UI**
- Clean, modern ChatGPT-like interface
- Custom AiNeuralChat logo
- Real-time typing indicators
- Conversation history sidebar
- Responsive design

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning)
- Groq API key (FREE - get it at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Swaliha1410/Minichatboat-Rag.git
cd Minichatboat-Rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```bash
# Copy from example
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
JWT_SECRET_KEY=your_secret_key_here
```

4. **Run the application**

**Option 1: Using the launcher (Windows)**
```bash
launch-aineuralchat.bat
```

**Option 2: Manual start**
```bash
# Start backend
python groq_app.py

# Open aineuralchat.html in your browser
```

5. **Access the application**
- Frontend: `aineuralchat.html` (opens automatically)
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📖 How It Works

### RAG Pipeline

```
📄 Document Upload
    ↓
🔍 Text Extraction (PDF/DOCX/TXT)
    ↓
✂️ Smart Chunking (500 chars with 50 char overlap)
    ↓
🧠 Generate Embeddings (Sentence Transformers)
    ↓
💾 Store in FAISS Vector Database
    ↓
❓ User Query
    ↓
🔎 Vector Similarity Search
    ↓
📚 Retrieve Relevant Chunks
    ↓
🤖 AI Response with Context (Groq)
```

### Technology Stack

**Backend:**
- **FastAPI** - Modern, fast web framework
- **Sentence Transformers** - State-of-the-art embeddings
- **FAISS** - Efficient vector similarity search
- **PyPDF2** - PDF text extraction
- **python-docx** - Word document processing
- **Groq AI** - Fast LLM inference

**Frontend:**
- Pure HTML/CSS/JavaScript
- No framework dependencies
- Modern, responsive design

## 🎯 Usage

### 1. Upload Documents

Click the upload button (📎) and select your documents:
- Supported formats: PDF, DOCX, TXT
- Multiple documents can be uploaded
- Each document is processed into searchable chunks

### 2. Ask Questions

Type your questions in the chat input:
```
"What is the main topic of the uploaded document?"
"Summarize the key points from my files"
"Find information about [specific topic]"
```

### 3. Get AI Responses

The AI will:
- Search through your documents using semantic similarity
- Find the most relevant chunks
- Provide context-aware answers
- Cite the source documents

## 📁 Project Structure

```
Minichatboat-Rag/
├── aineuralchat.html          # Frontend interface
├── groq_app.py                # Backend with RAG system
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your environment variables (not in git)
├── launch-aineuralchat.bat    # Windows launcher script
├── README.md                  # This file
├── uploads/                   # Uploaded files storage
└── vector_stores/             # Vector embeddings storage
```

## 🔧 Configuration

### RAG Parameters

Edit `groq_app.py` to customize:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE = 500                       # Characters per chunk
CHUNK_OVERLAP = 50                     # Overlap between chunks
MAX_TOKENS = 3000                      # Max context tokens
```

### Groq AI Settings

```python
"model": "llama-3.1-8b-instant"       # Groq model
"max_tokens": 1000                     # Max response tokens
"temperature": 0.7                     # Response creativity
```

## 🛠️ API Endpoints

### Health Check
```bash
GET /api/health
```

### Upload Document
```bash
POST /api/documents/upload
Content-Type: multipart/form-data
Body: file
```

### Send Message
```bash
POST /api/chat/send
Content-Type: application/json
Body: {"content": "Your question here"}
```

### List Documents
```bash
GET /api/documents/
```

### RAG Statistics
```bash
GET /api/rag/stats
```

## 🔍 RAG System Details

### Vector Embeddings

- **Model:** `all-MiniLM-L6-v2`
- **Dimension:** 384
- **Speed:** Fast inference (~5ms per query)
- **Quality:** High semantic understanding

### Vector Store

- **Backend:** FAISS (Facebook AI Similarity Search)
- **Index Type:** IndexFlatIP (Inner Product)
- **Normalization:** L2 normalized vectors
- **Search:** Cosine similarity

### Chunking Strategy

- **Size:** 500 characters per chunk
- **Overlap:** 50 characters
- **Boundary:** Word-aware splitting
- **Context:** Maintains semantic coherence

## 📊 Performance

- **Startup Time:** ~2-3 seconds
- **Document Processing:** ~1-2 seconds per page
- **Query Response:** ~200-500ms
- **Embedding Generation:** ~50-100ms per chunk
- **Vector Search:** <10ms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast and free LLM API
- **Sentence Transformers** for excellent embedding models
- **FAISS** for efficient vector search
- **FastAPI** for the amazing web framework

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Dependencies Installation Issues
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with specific versions
pip install -r requirements.txt --no-cache-dir
```

### Groq API Key Issues
- Ensure your API key starts with `gsk_`  
- Get a free key at: https://console.groq.com
- Add it to `.env` file correctly

## 📧 Contact

- GitHub: [@Swaliha1410](https://github.com/Swaliha1410)
- Project Link: [https://github.com/Swaliha1410/Minichatboat-Rag](https://github.com/Swaliha1410/Minichatboat-Rag)

## ⭐ Star this repository if you find it helpful!

---

**Built with ❤️ using FastAPI, Sentence Transformers, and Groq AI**
