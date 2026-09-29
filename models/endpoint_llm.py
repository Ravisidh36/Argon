import json
import requests

from config import LLM_ENDPOINT, MODEL_NAME
from tools import get_tool_definitions


class FunctionCallObj:
    """Normalized tool call used by AgentLoop."""

    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class ModelResponse:
    """Normalized response used by AgentLoop."""

    def __init__(self, text: str, function_calls=None):
        self.text = text
        self.function_calls = function_calls or []


class EndpointModel:
    """OpenAI-compatible client for Ollama/vLLM/custom endpoints."""

    def __init__(self, endpoint=LLM_ENDPOINT, model_name=MODEL_NAME):
        self.endpoint = endpoint
        self.model_name = model_name
        self.messages = []
        self.tools = get_tool_definitions()
        self.valid_tool_names = self._get_valid_tool_names()

    def _get_valid_tool_names(self):
        names = set()
        for tool in self.tools:
            try:
                function = tool["function"]
                name = function.get("name")
                if name:
                    names.add(name)
            except (AttributeError, KeyError, TypeError):
                continue
        return names

    def create_response(self, text: str):
        return ModelResponse(text=text, function_calls=[])

    def start_chat(self, tools=None, system_instruction=None):
        self.messages = []
        if tools is not None:
            self.tools = tools
            self.valid_tool_names = self._get_valid_tool_names()
        if system_instruction:
            self.messages.append({
                "role": "system",
                "content": str(system_instruction),
            })

    @staticmethod
    def clean_args(raw_args):
        """Normalize proper and common Qwen-style malformed arguments."""
        if not isinstance(raw_args, dict):
            return raw_args

        cleaned = {}
        for key, value in raw_args.items():
            if isinstance(value, dict):
                if "value" in value:
                    cleaned[key] = value["value"]
                elif "description" in value and set(value).issubset(
                    {"type", "description"}
                ):
                    cleaned[key] = value["description"]
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _normalize_tool_name(self, name):
        """Map common model-invented aliases to registered tool names."""
        if name in self.valid_tool_names:
            return name

        aliases = {
            "create_file": "write_file",
            "writeFile": "write_file",
            "createFile": "write_file",
            "readFile": "read_file",
            "listFiles": "list_files",
            "appendFile": "append_file",
            "delete_file": "dlt_file",
            "deleteFile": "dlt_file",
            "remove_file": "dlt_file",
            "removeFile": "dlt_file",
            "search_web": "web_search",
            "webSearch": "web_search",
            "calculate_expression": "calculate",
            "calculator": "calculate",
            "retrieve": "retrieve_document",
            "search_documents": "retrieve_document",
            "index": "index_document",
        }

        normalized = aliases.get(name)
        if normalized in self.valid_tool_names:
            return normalized

        return None

    def _make_call(self, parsed):
        if not isinstance(parsed, dict) or "name" not in parsed:
            return None

        normalized_name = self._normalize_tool_name(parsed["name"])
        if normalized_name is None:
            return None

        args = parsed.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        return FunctionCallObj(normalized_name, self.clean_args(args))

    def _parse_text_tool_calls(self, text: str):
        """Parse one, array, or consecutive JSON tool calls emitted as text."""
        text = self._strip_fences(text)
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                call = self._make_call(parsed)
                return [call] if call else []
            if isinstance(parsed, list):
                calls = []
                for item in parsed:
                    try:
                        call = self._make_call(item)
                        if call:
                            calls.append(call)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                return calls
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        decoder = json.JSONDecoder()
        calls = []
        position = 0
        while position < len(text):
            start = text.find("{", position)
            if start == -1:
                break
            try:
                parsed, end = decoder.raw_decode(text, start)
                call = self._make_call(parsed)
                if call:
                    calls.append(call)
                position = end
            except (json.JSONDecodeError, TypeError, ValueError):
                position = start + 1
        return calls

    def _parse_tool_calls(self, choice):
        native = choice.get("tool_calls") or []
        calls = []
        for tc in native:
            try:
                fn = tc["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                normalized_name = self._normalize_tool_name(fn["name"])
                if normalized_name is None:
                    continue
                calls.append(FunctionCallObj(
                    normalized_name,
                    self.clean_args(args),
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        if calls:
            return calls
        return self._parse_text_tool_calls(choice.get("content", "") or "")

    def generate(self, content):
        self.messages.append({"role": "user", "content": str(content)})

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": self.tools,
            "temperature": 0.2,
            "stream": False,
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]
            reply_text = choice.get("content", "") or ""
            function_calls = self._parse_tool_calls(choice)

            if not function_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": reply_text,
                })

            return ModelResponse(
                text="" if function_calls else reply_text,
                function_calls=function_calls,
            )

        except Exception as e:
            return ModelResponse(
                text=f"Error connecting to custom endpoint ({self.endpoint}): {str(e)}",
                function_calls=[],
            )

    def generate_no_tools(self, content):
        """One clean synthesis call with no tool schemas."""
        clean_messages = []
        if self.messages and self.messages[0].get("role") == "system":
            clean_messages.append(self.messages[0])
        clean_messages.append({"role": "user", "content": str(content)})

        payload = {
            "model": self.model_name,
            "messages": clean_messages,
            "temperature": 0.2,
            "stream": False,
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]
            return ModelResponse(
                text=choice.get("content", "") or "",
                function_calls=[],
            )

        except Exception as e:
            return ModelResponse(
                text=f"Error connecting to custom endpoint ({self.endpoint}): {str(e)}",
                function_calls=[],
            )
