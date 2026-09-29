# Argon

Argon is a local-first CLI AI agent for coding, document analysis, retrieval, and file-system workflows. It combines a self-hosted LLM with tool calling, document ingestion, RAG retrieval, reranking, and practical file-system tools so the agent can execute multi-step tasks instead of only generating text.

## What Argon Does

- **CLI agent:** interact with Argon directly from the terminal.
- **Tool calling:** the model can select registered tools and Argon executes them through a central tool registry.
- **Multi-step execution:** the agent can continue across multiple tool rounds, allowing workflows such as `write → read → verify → calculate → report`.
- **File-system operations:** create, read, append, list, and delete files inside the configured workspace.
- **Document RAG:** index local documents and retrieve relevant chunks for questions about their contents.
- **Semantic retrieval + reranking:** FAISS provides fast candidate retrieval, followed by a reranking stage for more precise results.
- **Document parsing:** Docling is used for structured document extraction before indexing.
- **Embeddings:** BGE-M3 is supported for document embeddings and model-specific FAISS indexes.
- **Web search:** retrieve current external information when the agent determines that local documents are insufficient.
- **Calculator:** execute arithmetic expressions through a dedicated tool.
- **Local/self-hosted inference:** Argon can communicate with an OpenAI-compatible endpoint such as Ollama without requiring a hosted LLM API key.

## Architecture

```text
User
  │
  ▼
CLI (main.py)
  │
  ▼
AgentLoop
  │
  ├── LLM / Tool Selection
  │       │
  │       ├── write_file
  │       ├── read_file
  │       ├── list_files
  │       ├── append_file
  │       ├── dlt_file
  │       ├── calculate
  │       ├── index_document
  │       ├── retrieve_document
  │       └── web_search
  │
  ▼
ToolExecutor
  │
  ├── File tools
  ├── RAG / retrieval pipeline
  ├── Calculator
  └── Web search
  │
  ▼
Tool results
  │
  ▼
AgentLoop continues until the task is complete
  │
  ▼
Concise final response
```

## RAG Pipeline

Argon's document workflow is designed around retrieval quality rather than simply dumping an entire document into the LLM.

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
 BGE-M3 embeddings
      │
      ▼
   FAISS index
      │
      ▼
Candidate retrieval
      │
      ▼
   Reranking
      │
      ▼
Relevant chunks
      │
      ▼
LLM-generated answer
```

Indexes are kept model-specific so embeddings from different models are not accidentally mixed. The project also stores embedding metadata alongside the FAISS index to detect model mismatches.

## Tool System

Argon uses a provider-agnostic tool-definition layer. The registered tools currently include:

| Tool | Purpose |
|---|---|
| `write_file` | Create or overwrite a file |
| `read_file` | Read a file |
| `list_files` | List files/directories |
| `append_file` | Append content to a file |
| `dlt_file` | Delete a file |
| `index_document` | Index a local document |
| `retrieve_document` | Retrieve relevant document information |
| `calculate` | Perform calculations |
| `web_search` | Search the web for external/current information |

Tool execution is centralized in `agent/executor.py`, while `agent/loop.py` controls iterative execution and continuation.

## Multi-Step Agent Execution

Argon is designed to behave more like a CLI agent than a one-shot chatbot. For example, a request such as:

```text
Create an input file, create a C++ program that reads it, read both files to verify them, calculate some values, search the indexed documents, search the web, create a report, and verify the report.
```

can require several tool calls. The agent loop executes returned tool calls, feeds verified tool results back to the model, and continues until the model no longer requests another operation or the configured safety limit is reached.

The implementation also normalizes common model-generated tool aliases to the exact registered tool names where appropriate.

## Local LLM / Ollama

Argon can use an OpenAI-compatible local endpoint such as Ollama's `/v1/chat/completions` interface.

Example model configuration:

```text
qwen2.5-coder:3b
```

The endpoint and model are configured through the project's configuration/environment settings.

Example Ollama endpoint:

```text
http://localhost:11434/v1/chat/completions
```

Make sure the selected model is available in Ollama before starting Argon.

## Project Structure

```text
Argon/
├── agent/
│   ├── executor.py          # Tool registry and execution
│   ├── loop.py              # Iterative agent/tool loop
│   └── context_builder.py   # Retrieval/context prompt construction
├── models/
│   ├── endpoint_llm.py      # OpenAI-compatible endpoint client
│   └── gemini.py            # Gemini model integration
├── retrieval/
│   ├── embeddings.py        # Embedding configuration
│   ├── index_document.py    # Document indexing
│   ├── retrieve.py          # Retrieval interface
│   ├── rerank.py            # Reranking
│   └── vector_store.py      # FAISS storage/loading
├── tools/
│   ├── file_access.py       # File-system tools
│   ├── calculator.py        # Calculator tool
│   └── web_search.py        # Web-search tool
├── data/
│   └── documents/           # Local documents/files
├── vector_store/             # Local FAISS indexes and registries
├── config.py                # Application configuration
├── main.py                  # CLI entry point
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies used by the project.

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the required environment variables according to the selected LLM provider and enabled tools.

For a local Ollama setup, make sure Ollama is running and the configured model is installed.

## Running Argon

Start the CLI from the project root:

```bat
python main.py
```

Then enter requests directly in the terminal. Use:

```text
/quit
```

to exit the CLI.

## Example Requests

### File workflow

```text
Create data/documents/example.txt containing hello, then read the file and verify its contents.
```

### Coding workflow

```text
Create a C++ program that reads numbers from data/documents/input.txt instead of hardcoding them, then verify the program using read_file.
```

### RAG workflow

```text
What holidays are mentioned in 3.pdf?
```

Argon retrieves relevant indexed chunks and uses those chunks as the basis for the answer.

### Calculation

```text
Calculate (42 * 18) + 100 and explain the result briefly.
```

### Combined workflow

```text
Create the required files, verify them, perform calculations, retrieve information from indexed documents, search the web for current information, create a report, and verify the report.
```

## Design Goals

Argon is being developed around four practical goals:

1. **Grounded answers** — document questions should be answered from retrieved document content rather than invented information.
2. **Reliable tool execution** — filesystem and other actions should be performed by tools instead of merely described in text.
3. **Multi-step autonomy** — complex requests should be decomposed into a sequence of tool operations and verification steps.
4. **Local-first execution** — the core agent can run against a self-hosted model and local vector stores.

## Current Limitations

Small local models can still make incorrect tool selections, produce malformed arguments, or require several inference rounds for complex workflows. Argon therefore validates tool names, normalizes common argument formats, limits iterative execution, and relies on actual tool results rather than trusting the model's claims about completed operations.

The quality of document answers also depends on document parsing, chunking, embedding quality, retrieval, and reranking. A successful retrieval does not automatically guarantee that the underlying document contains the answer, so the final model should not invent unsupported facts.

## License

See the repository for the project's current license and contribution information.
