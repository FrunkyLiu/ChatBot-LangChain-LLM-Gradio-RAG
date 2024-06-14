from abc import abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage
from .RAG import RAG
import typing as tp


class ChatBotTemplate:
    def __init__(self, LLM_model:BaseChatModel, RAG: RAG = None) -> None:
        """Define an abstract base class for chatbot templates"""
        self.LLM_model = LLM_model
        self.rag = RAG
        self.chain = None
        self.history = []

    @abstractmethod
    def set_chainflow(self) -> None:
        """Abstract method to set up the chain flow."""
        pass

    @abstractmethod
    def input_message_preprocess(self, message: str, history: tp.List[AnyMessage]):
        """Abstract method to preprocess the input message."""
        pass

    @abstractmethod
    def get_llm_model(self, model:str):
        """Initialize and return the LLM model."""
        pass

    @abstractmethod
    def get_RAG(self, documents: tp.List[str]):
        """Initialize and return the RAG instance."""
        pass

    def check_history(self, history_gradio: tp.List):
        """Check and synchronize the chat history."""
        if not history_gradio:
            self.history = []
        else:
            self.history = self.history[:len(history_gradio)*2]
        
    def add_history(self, content:str, MessageType:AnyMessage):
        """Add a message to the chat history."""
        self.history.append(
            MessageType(content=content)
        )

    def respond(self, message: str, history: tp.List[tuple[str, str]], **kwargs):
        """Respond to a message using the chain."""
        assert self.chain, "Please set up the chainflow first."
        self.check_history(history)
        res = ''
        for chunk in self.chain.stream(self.input_message_preprocess(message, self.history)):
            if 'answer' in chunk:
                res += chunk['answer']
                yield res
            elif hasattr(chunk, 'content'):
                res += chunk.content
                yield res

        self.add_history(message, HumanMessage)
        self.add_history(res, AIMessage)