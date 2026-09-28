from retrieval import index_document, retrieve
from tools import file_access, calculator, web_search

def get_tool_definitions():
    """
    Returns the standard JSON-schema definitions for all tools.
    These are provider-agnostic and work with OpenAI, Ollama, vLLM, and Qwen.
    """
    return [
        {
            "type": "function",
            "function": index_document.index_document_tool
        },
        {
            "type": "function",
            "function": retrieve.retrieve_document_tool
        },
        {
            "type": "function",
            "function": file_access.files_list
        },
        {
            "type": "function",
            "function": file_access.file_read
        },
        {
            "type": "function",
            "function": file_access.file_write
        },
        {
            "type": "function",
            "function": file_access.file_append
        },
        {
            "type": "function",
            "function": file_access.file_delete
        },
        {
            "type": "function",
            "function": calculator.calculate_tool
        },
        {
            "type": "function",
            "function": web_search.web_search_tool
        },
    ]
