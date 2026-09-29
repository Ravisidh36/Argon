# Argon

**Argon** is a local-first CLI AI agent for coding, document analysis, RAG, file-system automation, calculations, and web search. It is designed to behave like an actual command-line agent: the LLM selects tools, Argon executes them, real tool results are preserved, and the agent can continue multi-step tasks until completion.

## Highlights

- Local CLI AI agent
- Multi-round agent/tool execution
- Multiple tool calls in a single model response
- Native OpenAI-compatible tool-call handling
- JSON-text fallback parsing for local models such as Qwen
- Tool-name validation and alias normalization
- File creation, reading, appending, listing, and deletion
- Local document indexing and RAG retrieval
- **Docling** document parsing
- **BGE-M3** embeddings
- **FAISS** vector search
- **Cross-encoder reranking**
- Model-specific FAISS indexes and embedding compatibility checks
- Calculator tool
- Web-search tool
- Ollama / Qwen local inference
- Gemini integration

## Architecture

```text
                         User / CLI
                             │
                             ▼
                        AgentLoop
                             │
                             ▼
                    LLM / EndpointModel
                             │
                     Tool selection
                             │
                             ▼
                      ToolExecutor
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
      File tools         RAG tools         Other tools
   write/read/list/   index/retrieve      calculate
    append/delete                         web_search
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                       Tool results
                             │
                             ▼
                  More tool rounds if needed
                             │
                             ▼
                       Final response
```

## Agent Loop

Argon's agent loop supports both independent and dependent multi-step workflows.

For independent operations, the model can request multiple tools in one response and Argon executes all returned calls.

For dependent operations, tool results are preserved in the endpoint conversation so the model can request the next operation using the actual previous result.

Example:

```text
User request
    ↓
Qwen selects write_file
    ↓
Argon executes it
    ↓
Tool result is recorded
    ↓
Qwen requests read_file
    ↓
Argon executes it
    ↓
Qwen verifies result
    ↓
Next required tool
    ↓
Final answer
```

The loop has a configurable safety limit of 8 rounds to prevent infinite execution.

## Tool System

Tool definitions are centralized in `tools/__init__.py`. The currently registered tools are:

| Tool | Purpose |
|---|---|
| `write_file` | Create or overwrite a file |
| `read_file` | Read a file |
| `list_files` | List files and directories |
| `append_file` | Append content to a file |
| `dlt_file` | Delete a file |
| `index_document` | Index a local document for RAG |
| `retrieve_document` | Retrieve relevant information from indexed documents |
| `calculate` | Perform calculations |
| `web_search` | Search the web for external/current information |

Tool execution is centralized in `agent/executor.py`.

### Tool-call robustness

The endpoint model supports:

- Native OpenAI-compatible `tool_calls`
- A single JSON tool call returned as text
- Multiple JSON tool calls returned as separate JSON objects
- JSON arrays of tool calls
- Markdown-fenced JSON
- Common malformed Qwen arguments
- Common tool aliases such as `create_file` → `write_file`

Unknown tool names are rejected rather than passed directly into the executor.

## File-System Automation

Argon can perform real filesystem operations rather than merely describing what the user should type.

Example:

```text
Create data/documents/example.txt containing hello and then verify its contents.
```

The intended workflow is:

```text
write_file → read_file → final confirmation
```

For multi-file tasks:

```text
write_file(a.txt)
write_file(b.txt)
read_file(a.txt)
read_file(b.txt)
    ↓
final response
```

The terminal should expose the useful final result rather than raw tool-call JSON.

## RAG Pipeline

Argon's document pipeline combines structured parsing, semantic retrieval, and reranking:

```text
PDF / Document
      │
      ▼
   Docling
      │
      ▼
Structured / cleaned text
      │
      ▼
    Chunking
      │
      ▼
   BGE-M3
  embeddings
      │
      ▼
    FAISS
      │
      ▼
Candidate retrieval
      │
      ▼
Cross-encoder reranking
      │
      ▼
Relevant chunks
      │
      ▼
      LLM
      │
      ▼
Grounded answer
```

### Document parsing

**Docling** is used to convert and structure document content before indexing.

### Embeddings

The embedding layer uses the configured Hugging Face embedding model and defaults to:

```text
BAAI/bge-m3
```

Embeddings are normalized before indexing.

### FAISS

FAISS provides local vector search. Indexes are isolated by embedding model so vectors from incompatible embedding spaces are not accidentally mixed. Embedding metadata is stored alongside the index to detect model mismatches.

### Reranking

The initial vector search retrieves a candidate pool. A cross-encoder then scores the query and candidate text together to improve relevance ordering.

Current reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This gives Argon a two-stage retrieval architecture:

```text
Bi-encoder retrieval
        ↓
Candidate pool
        ↓
Cross-encoder reranking
        ↓
Top relevant chunks
```

## Document Grounding

For questions about indexed local documents, Argon is designed to answer using retrieved document evidence.

Example:

```text
What holidays are mentioned in 3.pdf?
```

The intended workflow is:

```text
retrieve_document
      ↓
relevant chunks
      ↓
reranking
      ↓
LLM synthesis
```

The system should not invent dates or facts that are absent from the retrieved material.

RAG quality depends on the entire pipeline:

```text
Parsing → Chunking → Embeddings → Retrieval → Reranking → Generation
```

## Local LLM with Ollama

Argon supports an OpenAI-compatible endpoint and can run against a local Ollama model.

Current development setup:

```text
Model:
qwen2.5-coder:3b

Typical Ollama endpoint:
http://localhost:11434/v1/chat/completions
```

The exact endpoint and model are configurable through environment variables.

The endpoint client sends tool definitions to the model and converts the model response into normalized internal tool calls before handing them to `ToolExecutor`.

## Gemini

Argon also contains a Gemini model integration. The configured Gemini chat model defaults to:

```text
gemini-2.5-flash
```

The agent loop is designed to work through a common model interface rather than embedding provider-specific logic in the orchestration layer.

## Project Structure

```text
Argon/
├── agent/
│   ├── executor.py          # Central tool registry and execution
│   └── loop.py              # Multi-round agent loop
│
├── models/
│   ├── endpoint_llm.py      # Ollama / OpenAI-compatible endpoint client
│   └── gemini.py            # Gemini integration
│
├── retrieval/
│   ├── embeddings.py        # Embedding configuration
│   ├── index_document.py    # Document indexing
│   ├── retrieve.py          # Document retrieval
│   ├── rerank.py            # Cross-encoder reranking
│   └── vector_store.py      # FAISS storage/loading
│
├── tools/
│   ├── __init__.py          # Tool definitions
│   ├── file_access.py       # File-system tools
│   ├── calculator.py        # Calculator
│   └── web_search.py        # Web search
│
├── data/
│   └── documents/           # Local documents and generated files
│
├── vector_store/            # FAISS indexes and document registry
├── config.py                # Configuration
├── main.py                  # CLI entry point
└── README.md
```

## Configuration

Configuration is loaded through environment variables.

Typical settings:

```text
MODEL_PROVIDER=endpoint
LLM_ENDPOINT=http://localhost:11434/v1/chat/completions
MODEL_NAME=qwen2.5-coder:3b
EMBEDDING_MODEL=BAAI/bge-m3
```

For Gemini:

```text
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_key
CHAT_MODEL=gemini-2.5-flash
```

The application also supports configuring the Hugging Face cache location through `HF_HOME`.

## Installation

Create a virtual environment and install dependencies.

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the required environment variables in `.env`.

For local inference, make sure Ollama is running and the configured model is installed.

## Running Argon

From the project root:

```bat
python main.py
```

Argon then accepts requests directly in the terminal.

Exit with:

```text
/quit
```

## Example Requests

### File operation

```text
Create data/documents/example.txt containing hello.
```

### File workflow

```text
Create data/documents/input.txt containing 10, create a C++ program that reads it using ifstream, then read both files and verify them.
```

### RAG

```text
What holidays are mentioned in 3.pdf?
```

### Calculator

```text
Calculate (42 * 18) + 100.
```

### Web search

```text
Search the web for the current weather in Jaipur.
```

### Multi-tool workflow

```text
Create the required files, verify them, calculate the requested values, retrieve information from indexed documents, search the web for current information, create a report, and read the report to verify it.
```

## Design Principles

### Execute actions, don't describe them

When the user asks Argon to create, modify, or delete a file, the corresponding tool should perform the operation.

### Trust actual tool results

Argon should not claim that a file was created, read, verified, indexed, calculated, or searched unless a corresponding tool result confirms the operation.

### Preserve tool state

Tool calls and their results are preserved so dependent workflows can continue across multiple rounds.

### Ground document answers

Retrieved document chunks are treated as evidence for local-document questions. The final answer should stay within the retrieved evidence.

### Keep the CLI clean

Raw tool JSON, tool schemas, internal instructions, and intermediate retrieval data should not be printed as the final user-facing answer.

## Current Limitations

Argon is an actively developed personal AI-agent project. Small local models can still make mistakes, including incorrect tool selection, malformed arguments, incomplete task planning, or unnecessary extra inference rounds.

RAG quality depends on document parsing, chunking, embedding, retrieval, reranking, and generation. Correct retrieval does not by itself guarantee a correct final answer.

Local LLM inference speed depends on the selected model, hardware, prompt size, tool-schema size, and number of inference rounds.

## Development Status

Argon is a work-in-progress project focused on building a practical local CLI agent that combines:

```text
Local LLM
    +
Tool Calling
    +
Multi-step Execution
    +
Docling
    +
BGE-M3
    +
FAISS
    +
Cross-Encoder Reranking
    +
File-System Automation
    +
Calculator
    +
Web Search
    =
Argon
```

## License

See the repository for the current license and contribution information.
