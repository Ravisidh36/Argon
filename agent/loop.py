from agent.executor import ToolExecutor
from agent.context_builder import ContextBuilder


class AgentLoop:

    def __init__(self, model):
        self.model = model
        self.executor = ToolExecutor()
        self.context_builder = ContextBuilder()

    def start(self, tools=None, system_instruction=None):
        self.model.start_chat(
            tools=tools,
            system_instruction=system_instruction,
        )

    def run(self, query):

        response = self.model.generate(query)

        # No tool call, return Gemini's answer directly
        if not response.function_calls:
            return response

        # Execute the first tool call
        function_call = response.function_calls[0]
        tool_result = self.executor.execute(function_call)

        # Extract just the result string — executor returns {"success": bool, "result": str}
        result_text = tool_result.get("result", str(tool_result)) if isinstance(tool_result, dict) else str(tool_result)

        # Build context prompt using the clean result string
        prompt = self.context_builder.build(query, result_text)

        # Turn 2: send WITHOUT tool schemas so the model must answer in plain text
        final_response = self.model.generate_no_tools(prompt)

        return final_response