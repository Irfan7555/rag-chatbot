# Enterprise RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot application with a clean, modern interface styled for rac (Saudi Telecom Company). This application enables users to upload documents (PDF, DOCX), create knowledge bases, and interact with the content through a conversational AI interface.

![ RAG Chatbot](https://raw.githubusercontent.com/syedazharmbnr1/rag-chatbot/refs/heads/main/image.png?token=GHSAT0AAAAAAC64DJZX3IG6FAFD5ISXOJ3K2AGJZQA)

## Features

- **Document Processing**: Upload and process PDF and DOCX files into searchable knowledge bases
- **Multiple Knowledge Bases**: Create, manage, and select multiple knowledge bases for different use cases
- **RAG Chat Interface**: Interactive chat with AI that retrieves relevant information from your documents
- **Source Citations**: Automatically provides source references with page numbers and relevance scores
- **Direct Chat Mode**: Option to chat directly with OpenAI models without using knowledge bases
- **Chat History**: Persistent storage of conversation history
- **Responsive UI**: Clean, modern interface with rac branding
- **Suggested Prompts**: Sample questions to help users get started

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Configure your OpenAI API key in `settings.yml`:
   ```yaml
   openai_key: "your-openai-api-key-here"
   ```

5. Create necessary directories:
   ```
   mkdir -p db logs FAISS_Index
   ```

## Usage

1. Start the application:
   ```
   streamlit run main.py
   ```

2. Upload Documents:
   - Click on "Upload Documents" in the main interface
   - Drag and drop PDF or DOCX files
   - The system will automatically process files into knowledge bases

3. Chat with your documents:
   - Type a question in the chat input
   - The system will retrieve relevant information from your documents
   - View source citations to see where information came from

4. Manage Knowledge Bases:
   - Select different knowledge bases from the sidebar
   - Use multiple knowledge bases simultaneously for comprehensive answers
   - Switch to Direct Chat Mode when you don't need to reference documents

## Project Structure

```
rag-chatbot/
├── assets/                  # Static assets
│   ├── fonts/               # rac custom fonts
│   └── style.css            # Main stylesheet
├── db/                      # Database directory
│   └── chat_history.db      # SQLite database
├── FAISS_Index/             # Vector stores for knowledge bases
├── logs/                    # Application logs
├── utils/                   # Utility modules
│   ├── __init__.py          # Package initialization
│   ├── chat.py              # Chat functionality
│   ├── database.py          # Database operations
│   ├── document_processing.py # Document processing
│   └── icons.py             # SVG icons
├── main.py                  # Main application
├── settings.yml             # Configuration settings
└── README.md                # Project documentation
```

## Configuration

The application can be configured through the `settings.yml` file:

```yaml
# API Keys
openai_key: "sk-"  # Replace with your actual OpenAI API key

# Models
chat_model: "gpt-4o-mini"
embeddings_model: "text-embedding-3-small"

# Database
database:
  type: "sqlite"
  path: "db/chat_history.db"

# Chunking Settings
default_chunk_size: 1000
default_chunk_overlap: 200

# Retrieval Settings
retrieval:
  top_k: 4
  search_type: "mmr"
  fetch_k: 8
```

## Requirements

The main dependencies for this project include:

- streamlit
- langchain
- langchain-openai
- faiss-cpu
- pyPDF
- docx2txt
- PyYAML
- SQLite3 (included in Python standard library)

For a complete list, see `requirements.txt`.

## License

[Data Panther]

---

Created by Datapanther
