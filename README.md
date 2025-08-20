# 📄 NovaMind RAG Chat — Document Question Answering with Streamlit

A simple **Streamlit application** that lets you upload **PDF, DOCX, PPTX, and TXT** files and then ask natural language questions.  
The assistant retrieves relevant chunks from your documents and answers using an LLM through [OpenRouter](https://openrouter.ai).  

![Demo](https://github.com/your-username/novamind-rag-chat/assets/demo-screenshot.png)

---

## ✨ Features
- 📑 Upload multiple files (PDF, Word, PowerPoint, Text)
- 🔍 Text extraction:
  - PDF → per page
  - DOCX → paragraphs & tables
  - PPTX → slide text & notes
  - TXT → full text
- ✂️ Smart chunking with adjustable size & overlap
- 🔎 Retrieval:
  - **Embeddings** via OpenRouter (default)
  - Fallback to **TF-IDF** (local, no API calls)
- 🤖 Question answering grounded in uploaded files
- ⚡ Streaming responses for fast interactivity
- 🎨 Clean Streamlit UI with sidebar settings

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/novamind-rag-chat.git
cd novamind-rag-chat
