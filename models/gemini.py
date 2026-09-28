from google import genai
from google.genai import types
from config import GEMINI_API_KEY, CHAT_MODEL
from retrieval import index_document, retrieve
from tools import file_access, calculator, web_search

def get_gemini_tools():

    return [
        types.Tool(
            function_declarations=[
                index_document.index_document_tool,
                retrieve.retrieve_document_tool,
                file_access.file_read,
                file_access.file_append,
                file_access.file_write,
                file_access.files_list,
                file_access.file_delete,
                calculator.calculate_tool,
                web_search.web_search_tool,
            ]
        )
    ]

class GeminiModel:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chat = None

    def start_chat(self, tools=None, system_instruction=None):
        config = types.GenerateContentConfig(
            tools=tools if tools is not None else get_gemini_tools(),
            system_instruction=system_instruction,
        )

        self.chat = self.client.chats.create(
            model=CHAT_MODEL,
            config=config,
        )

    def generate(self, content):
        """
        Sends a message to the active Gemini chat session
        and returns the response.
        """
        if self.chat is None:
            raise RuntimeError("Chat has not been started.")

        return self.chat.send_message(content)