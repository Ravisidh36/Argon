import json
import requests

from config import LLM_ENDPOINT, MODEL_NAME
from tools import get_tool_definitions


class FunctionCallObj:
    """Standardized representation of a tool call."""

    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class ModelResponse:
    """Normalized response object used by AgentLoop."""

    def __init__(self, text: str, function_calls=None):
        self.text = text
        self.function_calls = function_calls or []


class EndpointModel:
    """
    Client wrapper for an OpenAI-compatible / vLLM / Ollama /
    FastAPI LLM endpoint.
    """

    def __init__(
        self,
        endpoint=LLM_ENDPOINT,
        model_name=MODEL_NAME
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.messages = []
        self.tools = get_tool_definitions()

    # ============================================================
    # RESPONSE HELPER
    # ============================================================

    def create_response(self, text: str):
        return ModelResponse(
            text=text,
            function_calls=[]
        )

    # ============================================================
    # START CHAT
    # ============================================================

    def start_chat(self, tools=None, system_instruction=None):

        self.messages = []

        if tools is not None:
            self.tools = tools

        if system_instruction:
            self.messages.append({
                "role": "system",
                "content": str(system_instruction)
            })

    # ============================================================
    # ARGUMENT CLEANING
    # ============================================================

    @staticmethod
    def clean_args(raw_args):
        """
        Normalizes arguments produced by local models.

        Handles both:

        {
            "path": "data/file.cpp"
        }

        and malformed/schema-like output:

        {
            "path": {
                "type": "string",
                "description": "...",
                "value": "data/file.cpp"
            }
        }
        """

        if not isinstance(raw_args, dict):
            return raw_args

        cleaned = {}

        for key, value in raw_args.items():

            if isinstance(value, dict) and "value" in value:
                cleaned[key] = value["value"]
            else:
                cleaned[key] = value

        return cleaned

    # ============================================================
    # PARSE TOOL CALLS
    # ============================================================

    def _parse_tool_calls(self, choice, reply_text):

        function_calls = []

        # --------------------------------------------------------
        # 1. Native OpenAI / vLLM tool calling
        # --------------------------------------------------------

        tool_calls = choice.get("tool_calls")

        if tool_calls:

            for tc in tool_calls:

                try:
                    function = tc["function"]

                    fn_name = function["name"]
                    fn_args = function.get("arguments", {})

                    if isinstance(fn_args, str):
                        fn_args = json.loads(fn_args)

                    fn_args = self.clean_args(fn_args)

                    function_calls.append(
                        FunctionCallObj(
                            fn_name,
                            fn_args
                        )
                    )

                except Exception:
                    continue

        # --------------------------------------------------------
        # 2. Fallback for models that print JSON tool calls
        # --------------------------------------------------------

        if not function_calls:

            cleaned_text = reply_text.strip()

            # Remove Markdown code fences
            if cleaned_text.startswith("```"):

                lines = cleaned_text.splitlines()

                if len(lines) >= 2:

                    lines = lines[1:]

                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]

                    cleaned_text = "\n".join(lines).strip()

            # Parse:
            #
            # {
            #   "name": "write_file",
            #   "arguments": {...}
            # }
            #
            if cleaned_text.startswith("{"):

                try:

                    parsed = json.loads(cleaned_text)

                    if (
                        isinstance(parsed, dict)
                        and "name" in parsed
                    ):

                        fn_name = parsed["name"]
                        raw_args = parsed.get(
                            "arguments",
                            {}
                        )

                        if isinstance(raw_args, str):
                            raw_args = json.loads(raw_args)

                        raw_args = self.clean_args(raw_args)

                        function_calls.append(
                            FunctionCallObj(
                                fn_name,
                                raw_args
                            )
                        )

                except (json.JSONDecodeError, TypeError):
                    pass

        return function_calls

    # ============================================================
    # NORMAL GENERATION
    # ============================================================
    def generate_no_tools(self, content):
        """
        Generate a final response without exposing tool schemas.
        Used after a tool has already executed.
        """

        clean_messages = []

        # Keep the system instruction
        if self.messages and self.messages[0]["role"] == "system":
            clean_messages.append(self.messages[0])

        # Give the model only the user's request + tool result
        clean_messages.append({
            "role": "user",
            "content": str(content)
        })

        payload = {
            "model": self.model_name,
            "messages": clean_messages,
            "temperature": 0.2
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180
            )

            resp.raise_for_status()

            data = resp.json()

            choice = data["choices"][0]["message"]
            reply_text = choice.get("content", "") or ""

            function_calls = []

        except Exception as e:
            return ModelResponse(
                text=f"Error connecting to custom endpoint ({self.endpoint}): {str(e)}",
                function_calls=[]
            )
    def generate(self, content):
        """
        Send a user message to the endpoint and normalize the response.

        Supports:
        1. Native OpenAI/vLLM tool_calls
        2. A single JSON tool call printed by the model
        3. Multiple JSON tool calls printed by the model
        """

        self.messages.append({
            "role": "user",
            "content": str(content)
        })

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": self.tools,
            "temperature": 0.2,
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )

            resp.raise_for_status()

            data = resp.json()

            choice = data["choices"][0]["message"]

            reply_text = choice.get("content", "") or ""

            function_calls = []

            # =====================================================
            # ARGUMENT CLEANER
            # =====================================================

            def clean_args(raw_args):

                if not isinstance(raw_args, dict):
                    return raw_args

                cleaned = {}

                for key, value in raw_args.items():

                    # Qwen sometimes produces:
                    #
                    # "path": {
                    #     "type": "string",
                    #     "description": "...",
                    #     "value": "test.txt"
                    # }
                    #
                    # We only want:
                    #
                    # "path": "test.txt"

                    if isinstance(value, dict):

                        if "value" in value:
                            cleaned[key] = value["value"]

                        else:
                            cleaned[key] = value

                    else:
                        cleaned[key] = value

                return cleaned

            # =====================================================
            # 1. NATIVE TOOL CALLS
            # =====================================================

            native_tool_calls = choice.get("tool_calls", [])

            if native_tool_calls:

                for tool_call in native_tool_calls:

                    try:

                        function = tool_call["function"]

                        fn_name = function["name"]

                        fn_args = function.get(
                            "arguments",
                            {}
                        )

                        # Arguments are often returned as JSON string
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)

                        fn_args = clean_args(fn_args)

                        function_calls.append(
                            FunctionCallObj(
                                fn_name,
                                fn_args
                            )
                        )

                    except Exception:
                        # Ignore malformed individual tool calls
                        # instead of crashing the whole agent.
                        continue

            # =====================================================
            # 2. FALLBACK JSON TOOL CALLS
            #
            # Used when Qwen prints tool calls as normal text.
            # =====================================================

            if not function_calls:
                cleaned_text = reply_text.strip()

                # Remove Markdown code fences
                if cleaned_text.startswith("```"):
                    lines = cleaned_text.splitlines()

                    if lines:
                        lines = lines[1:]

                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]

                    cleaned_text = "\n".join(lines).strip()

                # ---------------------------------------------------------
                # Parse JSON tool calls
                #
                # Supports:
                # 1. Single JSON object
                # 2. JSON array
                # 3. Multiple consecutive JSON objects
                # ---------------------------------------------------------

                try:
                    # First try normal JSON parsing.
                    parsed = json.loads(cleaned_text)

                    parsed_calls = (
                        [parsed]
                        if isinstance(parsed, dict)
                        else parsed
                        if isinstance(parsed, list)
                        else []
                    )

                    for call in parsed_calls:
                        if not isinstance(call, dict):
                            continue

                        if "name" not in call:
                            continue

                        fn_name = call["name"]
                        raw_args = call.get("arguments", {})

                        if isinstance(raw_args, str):
                            raw_args = json.loads(raw_args)

                        raw_args = self.clean_args(raw_args)

                        function_calls.append(
                            FunctionCallObj(
                                fn_name,
                                raw_args
                            )
                        )

                except (json.JSONDecodeError, TypeError, ValueError):

                    # -----------------------------------------------------
                    # Qwen may output:
                    #
                    # {"name":"write_file",...}
                    # {"name":"write_file",...}
                    #
                    # This is NOT valid JSON as a single document.
                    # Parse the objects one-by-one.
                    # -----------------------------------------------------

                    decoder = json.JSONDecoder()
                    position = 0
                    length = len(cleaned_text)

                    try:
                        while position < length:

                            # Skip whitespace/newlines
                            while (
                                position < length
                                and cleaned_text[position].isspace()
                            ):
                                position += 1

                            if position >= length:
                                break

                            parsed, next_position = decoder.raw_decode(
                                cleaned_text,
                                position
                            )

                            position = next_position

                            if not isinstance(parsed, dict):
                                continue

                            if "name" not in parsed:
                                continue

                            fn_name = parsed["name"]

                            raw_args = parsed.get(
                                "arguments",
                                {}
                            )

                            if isinstance(raw_args, str):
                                raw_args = json.loads(raw_args)

                            raw_args = self.clean_args(raw_args)

                            function_calls.append(
                                FunctionCallObj(
                                    fn_name,
                                    raw_args
                                )
                            )

                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                # ---------------------------------------------
                # Remove markdown fences
                # ---------------------------------------------

                if cleaned_text.startswith("```"):

                    lines = cleaned_text.splitlines()

                    if lines:
                        lines = lines[1:]

                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]

                    cleaned_text = "\n".join(lines).strip()

                # ---------------------------------------------
                # Try to parse JSON
                # ---------------------------------------------

                try:

                    parsed = json.loads(cleaned_text)

                    # -----------------------------------------
                    # Single tool call
                    #
                    # {
                    #   "name": "write_file",
                    #   "arguments": {...}
                    # }
                    # -----------------------------------------

                    if (
                        isinstance(parsed, dict)
                        and "name" in parsed
                    ):

                        fn_name = parsed["name"]

                        raw_args = parsed.get(
                            "arguments",
                            {}
                        )

                        if isinstance(raw_args, str):
                            raw_args = json.loads(raw_args)

                        raw_args = clean_args(raw_args)

                        function_calls.append(
                            FunctionCallObj(
                                fn_name,
                                raw_args
                            )
                        )

                    # -----------------------------------------
                    # Multiple tool calls
                    #
                    # [
                    #   {
                    #     "name": "write_file",
                    #     "arguments": {...}
                    #   },
                    #   {
                    #     "name": "write_file",
                    #     "arguments": {...}
                    #   }
                    # ]
                    # -----------------------------------------

                    elif isinstance(parsed, list):

                        for call in parsed:

                            if not isinstance(call, dict):
                                continue

                            if "name" not in call:
                                continue

                            fn_name = call["name"]

                            raw_args = call.get(
                                "arguments",
                                {}
                            )

                            if isinstance(raw_args, str):
                                raw_args = json.loads(raw_args)

                            raw_args = clean_args(raw_args)

                            function_calls.append(
                                FunctionCallObj(
                                    fn_name,
                                    raw_args
                                )
                            )

                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError
                ):
                    pass

            # =====================================================
            # IMPORTANT:
            #
            # Do NOT store fake JSON tool calls as assistant
            # conversation messages.
            # =====================================================

            if not function_calls:

                self.messages.append({
                    "role": "assistant",
                    "content": reply_text
                })

            return ModelResponse(
                text="" if function_calls else reply_text,
                function_calls=function_calls
            )

        except Exception as e:

            return ModelResponse(
                text=(
                    f"Error connecting to custom endpoint "
                    f"({self.endpoint}): {str(e)}"
                ),
                function_calls=[]
            )