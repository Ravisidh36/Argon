from agent.executor import ToolExecutor
from agent.context_builder import ContextBuilder


class AgentLoop:
    """CLI-style agent loop with iterative multi-tool execution."""

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

    MAX_TOOL_ROUNDS = 8

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

    @staticmethod
    def _format_results(results):
        parts = []
        for item in results:
            result = str(item.get("result", "")).strip()
            if result:
                parts.append(f"[{item['name']}]\n{result}")
        return "\n\n".join(parts)

    def run(self, query):
        """
        Keep executing until the model stops requesting tools.

        A single model response may contain several independent tool calls.
        If the task is multi-step, the tool results are fed back to the model
        so it can request the next dependent operations. Only the final turn
        is returned to the CLI user.
        """
        current_query = query
        all_results = []

        for round_number in range(self.MAX_TOOL_ROUNDS):
            response = self.model.generate(current_query)

            # The model has decided that no more tools are needed.
            if not response.function_calls:
                return response

            round_results = []
            failed = False

            # Execute every tool call returned in this round.
            for function_call in response.function_calls:
                tool_result = self.executor.execute(function_call)

                if not tool_result.get("success", False):
                    failed = True
                    error = tool_result.get("error", "Unknown tool error")
                    round_results.append({
                        "name": function_call.name,
                        "result": f"ERROR: {error}",
                    })
                    continue

                round_results.append({
                    "name": function_call.name,
                    "result": tool_result.get("result", ""),
                })

            all_results.extend(round_results)

            if failed:
                # Give the model the actual failure so it can recover, rather
                # than exposing an internal traceback to the user.
                current_query = (
                    f"Continue the user's original task. This was tool round "
                    f"{round_number + 1}. Some requested operations failed. "
                    "Inspect the tool results below, correct the problem using "
                    "the available tools, and continue until the original task "
                    "is complete. Do not invent tool names or results.\n\n"
                    f"Original request:\n{query}\n\n"
                    f"Tool results:\n{self._format_results(round_results)}"
                )
                continue

            # Feed the completed operations back into the model. This is what
            # allows dependent workflows such as write -> read -> verify ->
            # write report -> read report to continue across rounds.
            current_query = (
                "Continue executing the user's original request. You are in "
                f"tool round {round_number + 1}. The following tools have just "
                "finished successfully. Determine what remaining operations "
                "are required. If more tools are needed, call them now. If the "
                "entire request is complete, provide only a concise final answer. "
                "Do not repeat completed operations unless verification or the "
                "user's request requires it. Use only registered tools and do not "
                "print tool-call JSON.\n\n"
                f"Original request:\n{query}\n\n"
                f"Results from this round:\n{self._format_results(round_results)}"
            )

        # The model kept requesting tools beyond the safety limit. Give a
        # concise status instead of hanging forever.
        return self._response(
            f"I stopped after {self.MAX_TOOL_ROUNDS} tool rounds to prevent an "
            "endless execution loop."
        )
