# Argon

**Argon** is a local-first CLI AI agent for coding, document analysis, RAG, file-system automation, calculations, and web search. It is designed to behave like an actual command-line agent: the LLM selects tools, Argon executes them, real tool results are returned to the model, and the model can continue the task through multiple tool rounds.

The current endpoint setup is designed around an **OpenAI-compatible API**, with **Ollama/Qwen** supported locally. Argon also contains a Gemini integration for API-based inference.

## Highlights

- Local CLI AI agent
- Multi-round tool execution
- Native assistant/tool message handling for the endpoint model
- Support for multiple tool calls in a single model response
- File creation, reading, appending, listing, and deletion
- Local document indexing and RAG retrieval
- **Docling** document parsing
- **BGE-M3** embeddings
- **FAISS** vector search
- **Cross-encoder reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Model-specific vector-store isolation and embedding compatibility checks
- Calculator tool
- Web-search tool
- Ollama / Qwen local inference
- Gemini model integration
- Tool-name normalization for common malformed/invented tool names
- Protection against claiming an operation succeeded without an actual tool result

## Architecture

```text
                         ┌─────────────────────┐
                         │      User / CLI      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      AgentLoop      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LLM / Endpoint    │
                         │   Tool Selection    │
                         └──────────┬──────────┘
                                    │
                         tool calls │
                                    ▼
                         ┌─────────────────────┐
                         │    ToolExecutor     │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       File-system tools       RAG tools             Other tools
       write/read/list/        index/retrieve        calculator
       append/delete                                 web_search
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Tool results     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ LLM continues task  │
                         │  (next tool round)  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Final response    │
                         └─────────────────────┘
```

## Agent Loop

Argon's agent loop is intentionally multi-step. A complex request does not have to finish after one tool call.

For example:

```text
User request
    ↓
Qwen selects write_file
    ↓
Argon executes write_file
    ↓
Actual tool result is recorded
    ↓
Qwen sees the result
    ↓
Qwen selects read_file
    ↓
Argon executes read_file
    ↓
Qwen verifies the file
    ↓
Qwen selects calculate / retrieve_document / web_search / ...
    ↓
Task complete
    ↓
Short final response
```

The current loop allows up to **8 tool rounds** to prevent an endless agent loop. Every tool call returned in a round is executed before the next model round. citeturn32file0

The endpoint model preserves tool execution using the assistant/tool message protocol rather than turning tool results into fake user messages. This is particularly important for local models such as Qwen running through Ollama. citeturn33file0

## Tool System

Tool definitions are centralized in `tools/__init__.py` and exposed to the model using provider-agnostic function schemas. The current registered tools are: citeturn34file0

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

### Tool execution reliability

The endpoint client validates tool names against the registered tool set. It also normalizes common model-generated aliases such as `create_file` → `write_file`, `readFile` → `read_file`, `calculator` → `calculate`, and `search_documents` → `retrieve_document`. If a tool name cannot be mapped to a registered tool, it is not executed. citeturn33file0

The endpoint client also handles both native OpenAI-compatible `tool_calls` and models that output tool calls as JSON text. Multiple JSON tool calls can be parsed from one response. citeturn33file0

## File-System Automation

The file tools allow Argon to perform actual file operations instead of merely printing code or commands for the user to execute.

Example:

```text
Create data/documents/test.txt containing hello, then read it and verify the contents.
```

The intended flow is:

```text
write_file → read_file → final confirmation
```

For a larger coding task, Argon can perform workflows such as:

```text
write input file
      ↓
write C++ program
      ↓
read input file
      ↓
read C++ program
      ↓
verify
      ↓
calculate
      ↓
write report
      ↓
read report
```

The model is instructed to rely on actual tool results and not claim that a file was created, read, verified, or searched unless the corresponding operation returned a result. citeturn32file0

## RAG Pipeline

Argon's document retrieval pipeline combines structured document parsing, semantic embeddings, FAISS retrieval, and cross-encoder reranking.

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
   Embeddings
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

### BGE-M3

The embedding layer uses `HuggingFaceEmbeddings` and the configured embedding model, which defaults to `BAAI/bge-m3`. Embeddings are normalized before indexing so FAISS L2 ranking corresponds to cosine-similarity ranking. citeturn37file0

### FAISS

Vector indexes are stored separately for each embedding model. This prevents vectors from different embedding spaces from being accidentally mixed. The vector-store implementation also stores embedding-model metadata and checks it when loading an index.

### Reranking

Initial semantic retrieval produces a candidate pool. The candidates are then reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates the query and candidate text together, allowing Argon to reorder the retrieved chunks based on query-specific relevance rather than relying only on embedding similarity. citeturn36file0

## Document Grounding

For questions about indexed documents, Argon is intended to answer from retrieved document content rather than inventing information.

Example:

```text
What holidays are mentioned in 3.pdf?
```

The retrieval pipeline should provide relevant chunks from `3.pdf`, after which the LLM generates an answer from those chunks.

A retrieval system can still fail if the document is parsed poorly, the relevant information is not retrieved, or the LLM misinterprets the retrieved context. RAG therefore depends on the complete pipeline:

```text
Parsing → chunking → embedding → retrieval → reranking → generation
```

## Local LLM with Ollama

Argon supports an OpenAI-compatible local endpoint. The current configuration defaults to:

```text
Endpoint:
http://localhost:8000/v1/chat/completions

Model:
qwen2.5-coder:3b
```

Both values can be overridden through environment variables. The model provider defaults to the local endpoint provider. Gemini is available as an alternative provider. citeturn35file0

For Ollama, the model must be installed and running locally. A typical Ollama-compatible endpoint is:

```text
http://localhost:11434/v1/chat/completions
```

The endpoint client sends tool schemas to the model and parses the returned tool calls before passing them to Argon's `ToolExecutor`. citeturn33file0

## Gemini

Argon also contains a Gemini integration. The configured Gemini model defaults to:

```text
gemini-2.5-flash
```

when the Gemini provider is selected. citeturn35file0

## Project Structure

```text
Argon/
├── agent/
│   ├── executor.py          # Central tool registry and execution
│   └── loop.py              # Multi-round agent loop
│
├── models/
│   ├── endpoint_llm.py      # OpenAI-compatible endpoint client
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
│   ├── file_access.py       # File-system operations
│   ├── calculator.py        # Calculator
│   └── web_search.py        # Web search
│
├── data/
│   └── documents/           # Local documents and generated files
│
├── vector_store/            # Local FAISS indexes and document registries
├── config.py                # Model/provider configuration
├── main.py                  # CLI entry point
└── README.md
```

## Configuration

Configuration is loaded through environment variables using `python-dotenv`.

Important settings include:

```text
MODEL_PROVIDER=endpoint
LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
MODEL_NAME=qwen2.5-coder:3b
EMBEDDING_MODEL=BAAI/bge-m3
```

For Gemini:

```text
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_key
CHAT_MODEL=gemini-2.5-flash
```

The repository also configures a Hugging Face cache location through `HF_HOME` when one is not already provided. citeturn35file0

## Installation

Create and activate a virtual environment:

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bat
pip install -r requirements.txt
```

Configure the required environment variables in `.env`.

For local inference, make sure Ollama is installed and the configured model is available.

## Running Argon

From the project root:

```bat
python main.py
```

Argon then accepts requests directly from the terminal.

Exit with:

```text
/quit
```

## Example Requests

### Create and verify a file

```text
Create data/documents/example.txt containing hello, then read the file and verify its contents.
```

### Create a C++ program using file input

```text
Create data/documents/input.txt containing 10 20 30, then create data/documents/program.cpp that reads the numbers using ifstream instead of hardcoding them, and verify both files.
```

### RAG query

```text
What holidays are mentioned in 3.pdf?
```

### Calculation

```text
Calculate (42 * 18) + 100.
```

### Web search

```text
Search the web for the current weather in Jaipur.
```

### Multi-tool workflow

```text
Create the required files, read them to verify their contents, calculate the requested values, retrieve information from indexed documents, search the web for current information, create a report containing the results, and read the report to verify it.
```

## Design Principles

### 1. Execute actions instead of describing them

When the user asks Argon to create or modify a file, the corresponding file tool should perform the operation. The final response should summarize the completed operation instead of dumping tool-call JSON.

### 2. Trust tool results

The model should not claim that an operation succeeded merely because it intended to call a tool. Actual tool results are the source of truth for execution status.

### 3. Preserve tool-call state

Tool calls and their results are preserved using the assistant/tool protocol so the model can perform dependent operations across multiple rounds. citeturn33file0

### 4. Ground document answers

Retrieved chunks are evidence for document questions. The generation step should not invent facts that are absent from the retrieved material.

### 5. Keep the CLI usable

Intermediate tool-call JSON, internal reasoning, schemas, and raw retrieval data should not be presented as the user's final answer.

## Current Limitations

Argon is still an actively developed agent and small local models can make mistakes. In particular:

- Small models may select the wrong tool.
- Models may produce malformed JSON arguments.
- A local model may need multiple inference rounds for complex tasks.
- Tool-call generation can be slower for long prompts or large tool schemas.
- RAG quality depends on parsing, chunking, embeddings, retrieval, and reranking.
- The model can still misunderstand retrieved evidence even when retrieval itself is correct.
- Web-search results are external information and should be distinguished from local-document results.

The endpoint client includes argument normalization, tool-name validation, JSON tool-call parsing, native tool-message handling, and a request timeout to make these failures more manageable. citeturn33file0

## Development Status

Argon is a work-in-progress personal AI-agent project focused on building a practical local CLI agent with reliable tool execution and grounded document retrieval.

The current architecture emphasizes:

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
Web Search
   =
Argon
```

## License

See the repository for the current license and contribution information.
