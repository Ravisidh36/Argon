from agent.executor import ToolExecutor


class AgentLoop:
    """CLI-style agent loop with iterative multi-tool execution."""

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

    def start(self, tools=None, system_instruction=None):
        self.model.start_chat(
            tools=tools,
            system_instruction=system_instruction,
        )

    def _response(self, text):
        return self.model.create_response(text)

    @staticmethod
    def _tool_summary(results):
        parts = []
        for item in results:
            name = item["name"]
            if item.get("success", False):
                parts.append(f"{name}: {item.get('result', '')}")
            else:
                parts.append(f"{name}: ERROR: {item.get('error', 'Unknown error')}")
        return "\n".join(parts)

    def run(self, query):
        """
        Execute a complete multi-step task.

        Tool results are returned to the model using the native assistant/tool
        message protocol. This is important for Ollama/Qwen: the model sees
        exactly which calls it made and which results came back, so it can
        continue dependent workflows instead of hallucinating that they ran.
        """
        continuation = query

        for round_number in range(self.MAX_TOOL_ROUNDS):
            response = self.model.generate(continuation)

            # No more tool calls: the model has produced the final response.
            if not response.function_calls:
                return response

            results = []

            # Execute every call returned in this round.
            for function_call in response.function_calls:
                result = self.executor.execute(function_call)
                results.append({
                    "name": function_call.name,
                    **result,
                })

            # IMPORTANT: preserve the actual assistant tool-call message and
            # tool results in the endpoint conversation. Do not turn them into
            # a fake user prompt.
            if hasattr(self.model, "add_tool_results"):
                self.model.add_tool_results(response, results)

            # Give the model a short continuation instruction. The actual tool
            # outputs are already in the conversation as tool messages.
            failed = any(not item.get("success", False) for item in results)
            if failed:
                continuation = (
                    "Continue the original task. One or more tool calls failed. "
                    "Inspect the tool error messages, correct the problem with "
                    "the registered tools, and continue. Do not claim an operation "
                    "succeeded unless a tool result confirms it. Do not print tool "
                    "call JSON."
                )
            else:
                continuation = (
                    "Continue the original task from the tool results above. "
                    "Perform every remaining requested operation. Do not repeat "
                    "completed operations unless verification requires it. "
                    "If another tool is needed, call it now. Only when the entire "
                    "original task is actually complete should you give a concise "
                    "final answer. Never claim a file was created, read, verified, "
                    "or searched unless the corresponding tool result confirms it. "
                    "Do not print tool-call JSON or internal reasoning."
                )

        return self._response(
            f"I stopped after {self.MAX_TOOL_ROUNDS} tool rounds to prevent an endless loop."
        )
