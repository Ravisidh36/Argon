from agent.executor import ToolExecutor
from agent.context_builder import ContextBuilder


class AgentLoop:
    """CLI-style agent loop: execute tools locally and only use the LLM when needed."""

    LLM_RESULT_TOOLS = {
        "retrieve_document",
        "read_file",
        "list_files",
        "calculate",
        "web_search",
    }

    ACTION_TOOLS = {
        "write_file",
        "append_file",
        "dlt_file",
        "index_document",
    }

    MAX_TOOL_ROUNDS = 3

    def __init__(self, model):
        self.model = model
        self.executor = ToolExecutor()
        self.context_builder = ContextBuilder()

    def start(self, tools=None, system_instruction=None):
        self.model.start_chat(
            tools=tools,
            system_instruction=system_instruction,
        )

    def _response(self, text):
        return self.model.create_response(text)

    def run(self, query):
        current_query = query

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.model.generate(current_query)

            if not response.function_calls:
                return response

            results = []

            # Execute every tool call returned in this model response.
            for function_call in response.function_calls:
                tool_result = self.executor.execute(function_call)

                if not tool_result.get("success", False):
                    return self._response(
                        f"Tool '{function_call.name}' failed: "
                        f"{tool_result.get('error', 'Unknown tool error')}"
                    )

                results.append({
                    "name": function_call.name,
                    "result": tool_result.get("result", ""),
                })

            context_results = [
                item for item in results
                if item["name"] in self.LLM_RESULT_TOOLS
            ]

            action_results = [
                item for item in results
                if item["name"] in self.ACTION_TOOLS
            ]

            # Retrieval/read/search results need one synthesis call.
            # If actions happened in the same round, include their result too.
            if context_results:
                result_text = "\n\n".join(
                    f"[{item['name']}]\n{item['result']}"
                    for item in results
                    if item["result"]
                )

                prompt = self.context_builder.build(query, result_text)
                return self.model.generate_no_tools(prompt)

            # Pure filesystem/index actions are already complete.
            # Do NOT make another LLM call just to say "done".
            if action_results:
                messages = [
                    str(item["result"]).strip()
                    for item in action_results
                    if str(item["result"]).strip()
                ]
                return self._response(
                    "\n".join(messages) or "Operation completed successfully."
                )

            # Unknown tool: return its result directly rather than generating
            # another expensive model response.
            messages = [
                str(item["result"]).strip()
                for item in results
                if str(item["result"]).strip()
            ]
            return self._response(
                "\n".join(messages) or "Operation completed successfully."
            )

        return self._response("I stopped after reaching the tool-operation limit.")
