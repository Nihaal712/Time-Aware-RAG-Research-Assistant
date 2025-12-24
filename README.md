<<<<<<< HEAD
# Time-Aware-RAG-Research-Assistant
Time-aware Retrieval-Augmented Generation system that reduces outdated and hallucinated research answers using temporal filtering, FAISS-based retrieval, and arXiv fallback.
=======
<<<<<<< HEAD
# Time-Aware-RAG-Research-Assistant
Time-aware Retrieval-Augmented Generation system that reduces outdated and hallucinated research answers using temporal filtering, FAISS-based retrieval, and arXiv fallback.
=======
# Time-Aware RAG Research Assistant

A professional-grade time-aware RAG (Retrieval-Augmented Generation) research assistant chatbot that helps users explore academic research without outdated or hallucinated answers.

## Features

- **Time-Aware Retrieval**: Filter research papers by publication periods
- **Local Document Processing**: Upload and process PDF research papers
- **arXiv Integration**: Automatic fallback to arXiv when local results are insufficient
- **Conversational Memory**: Persistent chat history across sessions
- **Temporal Comparisons**: Compare research developments across different time periods
- **Source Citations**: Clear attribution for all generated responses

## Setup

### Prerequisites

- Python 3.8 or higher
- Groq API key (sign up at https://groq.com)

### Installation

1. Clone the repository and navigate to the project directory

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

5. Run the application:
```bash
python app.py
```

The application will be available at http://localhost:7861

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
- `HUGGINGFACE_MODEL`: Embedding model name (default: all-MiniLM-L6-v2)
- `FAISS_INDEX_PATH`: Path to FAISS index storage (default: faiss_index)
- `CHAT_HISTORY_FILE`: Chat history file path (default: chat_history.json)
- `LOG_LEVEL`: Logging level (default: INFO)

## Usage

1. **Upload Documents**: Use the file upload interface to add PDF research papers
2. **Add URLs**: Enter URLs to research papers for automatic processing
3. **Set Time Filter**: Choose from "All time", "2025 only", "2024+", "2023+", or "Pre-2023"
4. **Ask Questions**: Type research questions in the chat interface
5. **View Sources**: Expand the sources accordion to see retrieved documents

## Time Filters

- **All time**: No temporal filtering
- **2025 only**: Only papers from 2025
- **2024+**: Papers from 2024 onwards
- **2023+**: Papers from 2023 onwards  
- **Pre-2023**: Papers published before 2023

## Architecture

The system consists of several modular components:

- `app.py`: Main application and Gradio interface
- `ingestion.py`: Document processing and year detection
- `retrieval.py`: Time-aware local document retrieval
- `fallback.py`: arXiv integration and fallback logic
- `generation.py`: Response generation with citations
- `memory.py`: Conversation memory management
- `utils.py`: Shared utility functions

## Deployment

### Local Deployment

1. Follow the installation steps above
2. Run the application:
```bash
python app.py
```
3. Access the interface at http://localhost:7860

### HuggingFace Spaces Deployment

1. Create a new Space on HuggingFace
2. Upload the following files:
   - `app.py`
   - `ingestion.py`
   - `retrieval.py`
   - `fallback.py`
   - `generation.py`
   - `memory.py`
   - `utils.py`
   - `requirements.txt`

3. Add your Groq API key as a secret in Space settings:
   - Go to Settings → Secrets
   - Add `GROQ_API_KEY` with your API key value

4. The Space will automatically build and deploy

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GROQ_API_KEY=""
ENV FAISS_INDEX_PATH="/app/faiss_index"
ENV CHAT_HISTORY_FILE="/app/chat_history.json"

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t rag-assistant .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here rag-assistant
```

## Testing

### Sample Testing Workflow

1. **Prepare Test Documents**:
   - Download 2-3 research papers in PDF format
   - Save them to a test directory

2. **Start the Application**:
```bash
python app.py
```

3. **Test Document Upload**:
   - Click "Upload PDF" and select a test paper
   - Verify the upload status shows success
   - Check that document count increases

4. **Test URL Processing**:
   - Enter a URL to a research paper (e.g., arXiv paper link)
   - Click "Add URL"
   - Verify the URL is processed successfully

5. **Test Time Filtering**:
   - Ask a question: "What are the latest developments in machine learning?"
   - Try different time filters: "2024+", "2023+", "Pre-2023"
   - Verify results change based on time filter

6. **Test Temporal Comparison**:
   - Ask: "How has deep learning evolved since 2020?"
   - Verify the response compares findings across time periods
   - Check that sources are properly cited

7. **Test Conversation Memory**:
   - Ask a question and get a response
   - Ask a follow-up: "Tell me more about that"
   - Verify the system understands context from previous messages
   - Close and restart the app
   - Verify chat history is restored

8. **Test arXiv Fallback**:
   - Ask a question about a niche topic not in your documents
   - Verify the system automatically searches arXiv
   - Check that arXiv sources are labeled as "[arXiv]"

### Running Unit Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_generation.py -v

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

### Troubleshooting

**Issue**: "Invalid or missing Groq API key"
- Solution: Ensure `GROQ_API_KEY` is set in `.env` file or environment variables

**Issue**: "No documents in vector store"
- Solution: Upload at least one PDF or URL before asking questions

**Issue**: "FAISS index corruption"
- Solution: Delete the `faiss_index` directory and restart the application

**Issue**: "arXiv search timeout"
- Solution: Check internet connection and try again. arXiv API may be temporarily unavailable

**Issue**: "PDF extraction fails"
- Solution: Ensure PDF is not corrupted. Try with a different PDF file

## Performance Considerations

- **Document Processing**: Large PDFs (>50MB) may take time to process
- **Vector Store Size**: FAISS performance degrades with very large collections (>10,000 documents)
- **Embedding Generation**: First query may be slow as embeddings are generated
- **arXiv Search**: Fallback search adds 2-5 seconds to query time

## Architecture Details

### Document Processing Pipeline
1. Extract text from PDF or URL
2. Detect publication year using statistical consensus
3. Split into chunks (1000 tokens with 200 token overlap)
4. Generate embeddings using HuggingFace model
5. Store in FAISS with metadata

### Query Processing Pipeline
1. Retrieve documents from FAISS with time filtering
2. Calculate relevance score
3. If insufficient, trigger arXiv fallback
4. Merge results with proper source labeling
5. Generate response using Groq LLM
6. Add citations and save to conversation memory

### Memory Management
- Short-term: LangChain ConversationBufferMemory for active session
- Long-term: JSON file persistence for chat history
- Automatic save after each interaction
- Automatic load on application startup

## API Integration

### Groq API
- Model: llama3-8b-8192
- Temperature: 0.7
- Max tokens: 1024
- Rate limits: Check Groq documentation

### HuggingFace Embeddings
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Normalized embeddings enabled

### arXiv API
- No authentication required
- Rate limit: ~3 requests per second
- Results limited to 5 papers per query

## Development Status

Core functionality is fully implemented and tested. 
>>>>>>> fb30943 (Initial commit: Time-Aware RAG Research Assistant)
>>>>>>> dd0e086 (Initial commit: Time-Aware RAG Research Assistant)
