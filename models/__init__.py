from config import MODEL_PROVIDER

def get_model():
    """
    Factory function returning the chosen LLM model instance based on config.
    Defaults to self-hosted EndpointModel (No API key required).
    """
    if MODEL_PROVIDER == "gemini":
        from models.gemini import GeminiModel
        return GeminiModel()
    else:
        from models.endpoint_llm import EndpointModel
        return EndpointModel()
