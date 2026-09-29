from agent.executor import ToolExecutor
from agent.context_builder import ContextBuilder


class AgentLoop:

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

    MAX_TOOL_ROUNDS = 5

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

        # =====================================================
        # TOOL LOOP
        # =====================================================

        current_query = query

        for _ in range(self.MAX_TOOL_ROUNDS):

            response = self.model.generate(current_query)

            # -------------------------------------------------
            # No tool call
            # -------------------------------------------------

            if not response.function_calls:
                return response

            results = []

            # -------------------------------------------------
            # Execute ALL tool calls returned by the model
            # -------------------------------------------------

            for function_call in response.function_calls:

                tool_name = function_call.name

                tool_result = self.executor.execute(
                    function_call
                )

                if not tool_result.get("success", False):

                    error = tool_result.get(
                        "error",
                        "Unknown tool error"
                    )

                    return self._response(
                        f"Tool '{tool_name}' failed: {error}"
                    )

                results.append({
                    "name": tool_name,
                    "result": tool_result.get(
                        "result",
                        ""
                    ),
                })

            # =================================================
            # ACTION TOOLS
            # =================================================

            action_results = [
                r for r in results
                if r["name"] in self.ACTION_TOOLS
            ]

            # =================================================
            # INFORMATION TOOLS
            # =================================================

            context_results = [
                r for r in results
                if r["name"] in self.LLM_RESULT_TOOLS
            ]

            # -------------------------------------------------
            # Retrieval / read operations need LLM synthesis
            # -------------------------------------------------

            if context_results:

                result_text = "\n\n".join(
                    f"[{item['name']}]\n{item['result']}"
                    for item in context_results
                )

                prompt = self.context_builder.build(
                    query,
                    result_text
                )

                return self.model.generate_no_tools(
                    prompt
                )

            # -------------------------------------------------
            # Action tools
            #
            # The operation already happened.
            # Tell the model what happened and let it decide
            # whether another tool is necessary.
            # -------------------------------------------------

            if action_results:

                result_text = "\n".join(
                    str(item["result"])
                    for item in action_results
                    if item["result"]
                )

                # Ask the model whether the user's request
                # still requires another operation.
                current_query = (
                    f"The following tool operations were completed:\n\n"
                    f"{result_text}\n\n"
                    f"Original user request:\n{query}\n\n"
                    "Continue completing the user's request. "
                    "If more tool operations are required, call "
                    "the appropriate tool. "
                    "If the request is completely finished, "
                    "give a short final response."
                )

        # -----------------------------------------------------
        # Safety limit
        # -----------------------------------------------------

        return self._response(
            "I stopped after reaching the maximum number "
            "of tool operations."
        )