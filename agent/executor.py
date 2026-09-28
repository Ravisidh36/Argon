from retrieval import index_document, retrieve
from tools import file_access, calculator, web_search


tool_registry = {
    "index_document": index_document.index_document,
    "retrieve_document": retrieve.retrieve_document,
    "list_files": file_access.list_files,
    "read_file": file_access.read_file,
    "write_file": file_access.write_file,
    "append_file": file_access.append_file,
    "dlt_file": file_access.dlt_file,
    "calculate": calculator.calculate,
    "web_search": web_search.web_search,
}


class ToolExecutor:

    def execute(self, function_call):

        try:
            tool = tool_registry[function_call.name]
            return {
                "success": True,
                "result": tool(**function_call.args)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }