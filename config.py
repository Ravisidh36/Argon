import os
from dotenv import load_dotenv

load_dotenv()

# Option to use Gemini (API) or your Self-Hosted Custom Endpoint (No API Key)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "endpoint")  # "endpoint" or "gemini"

# Your self-hosted fine-tuned model endpoint URL
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")

# Fallback API keys (only if MODEL_PROVIDER == "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")