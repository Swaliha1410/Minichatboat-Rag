@echo off
echo ========================================
echo      AiNeuralChat - RAG System
echo ========================================
echo.
echo Starting AiNeuralChat with Real RAG...
echo - Vector embeddings with Sentence Transformers
echo - Semantic search with FAISS
echo - PDF/DOCX document processing
echo - Advanced chunking strategy
echo.
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend will open automatically...
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
start "" "aineuralchat.html"
python groq_app.py

pause