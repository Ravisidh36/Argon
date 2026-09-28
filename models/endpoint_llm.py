import json
import requests
from config import LLM_ENDPOINT, MODEL_NAME

class FunctionCallObj:
    """Standardized representation of a tool call matching executor expectation."""
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

class ModelResponse:
    """Normalized response object for AgentLoop."""
    def __init__(self, text: str, function_calls=None):
        self.text = text
        self.function_calls = function_calls or []

from tools import get_tool_definitions

class EndpointModel:
    """
    Client wrapper that talks to your custom self-hosted endpoint
    (OpenAI-compatible / vLLM / Ollama / FastAPI backend).
    Users do NOT need any API keys to use this.
    """
    def __init__(self, endpoint=LLM_ENDPOINT, model_name=MODEL_NAME):
        self.endpoint = endpoint
        self.model_name = model_name
        self.messages = []
        self.tools = get_tool_definitions()

    def start_chat(self, tools=None, system_instruction=None):
        self.messages = []
        if tools is not None:
            self.tools = tools
        if system_instruction:
            self.messages.append({
                "role": "system",
                "content": str(system_instruction)
            })

    def generate_no_tools(self, content):
        """
        Generate a final answer without providing any tool schemas.
        Use this for Turn 2 (synthesis step) so the model cannot re-trigger tools.
        """
        self.messages.append({
            "role": "user",
            "content": str(content)
        })

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.2
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]["message"]
            reply_text = choice.get("content", "") or ""

            self.messages.append({"role": "assistant", "content": reply_text})
            return ModelResponse(text=reply_text, function_calls=[])

        except Exception as e:
            return ModelResponse(
                text=f"Error connecting to custom endpoint ({self.endpoint}): {str(e)}",
                function_calls=[]
            )

    def generate(self, content):
        self.messages.append({
            "role": "user",
            "content": str(content)
        })

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": self.tools,
            "temperature": 0.2
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            # Handles standard OpenAI / vLLM format
            choice = data["choices"][0]["message"]
            reply_text = choice.get("content", "") or ""

            # Check for tool / function calls in standard format
            function_calls = []
            
            def clean_args(raw_args):
                """Flattens any nested {'value': ...} or {'type': ...} from local LLM outputs."""
                if not isinstance(raw_args, dict):
                    return raw_args
                cleaned = {}
                for k, v in raw_args.items():
                    if isinstance(v, dict) and "value" in v:
                        cleaned[k] = v["value"]
                    else:
                        cleaned[k] = v
                return cleaned

            if "tool_calls" in choice and choice["tool_calls"]:
                for tc in choice["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"]["arguments"]
                    if isinstance(fn_args, str):
                        fn_args = json.loads(fn_args)
                    function_calls.append(FunctionCallObj(fn_name, clean_args(fn_args)))
            
            # Fallback: Check if model printed tool call directly as JSON or markdown in reply_text
            cleaned_text = reply_text.strip()
            if cleaned_text.startswith("```"):
                # Strip markdown code fence e.g. ```json ... ```
                lines = cleaned_text.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    if lines[-1].startswith("```"):
                        cleaned_text = "\n".join(lines[1:-1]).strip()
                    else:
                        cleaned_text = "\n".join(lines[1:]).strip()

            if not function_calls and cleaned_text.startswith("{") and "name" in cleaned_text:
                try:
                    parsed = json.loads(cleaned_text)
                    if "name" in parsed:
                        fn_name = parsed["name"]
                        raw_args = parsed.get("arguments", {})
                        if isinstance(raw_args, str):
                            raw_args = json.loads(raw_args)
                        function_calls.append(FunctionCallObj(fn_name, clean_args(raw_args)))
                except Exception:
                    pass

            # Record assistant reply in conversation memory
            self.messages.append({
                "role": "assistant",
                "content": reply_text
            })

            return ModelResponse(text=reply_text, function_calls=function_calls)

        except Exception as e:
            return ModelResponse(
                text=f"Error connecting to custom endpoint ({self.endpoint}): {str(e)}",
                function_calls=[]
            )
