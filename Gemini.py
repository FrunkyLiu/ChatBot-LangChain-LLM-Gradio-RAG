from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AnyMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Backend.ChatBot import ChatBotTemplate
from Backend.RAG import RAG
import typing as tp
import os

# Set the environment variable for the Google API key
os.environ['GOOGLE_API_KEY'] = "YOUR GOOGLE GEMINI API KEY"

# Define the main prompt template
prompt = """Answer the questions based on the provided documents below. \n
documents: {context}\n
question: {input}
"""

# Define the prompt template for history-aware retrieval
prompt_history = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

class Gemini(ChatBotTemplate):
    def __init__(self, document_path, model: str = 'gemini-1.5-pro-latest') -> None:
        """Define a specific implementation of the ChatBotTemplate"""
        self.LLM_model = self.get_llm_model(model)
        self.agent_prompt = self.get_agent_prompt(prompt)
        self.history_prompt = self.get_history_aware_prompt(prompt_history)
        self.embedding_encoder = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
        documents = self.load_documents(document_path)
        self.rag = self.get_RAG(documents)
        super().__init__(self.LLM_model, self.rag)
        self.set_chainflow()
    
    def load_documents(self, path):
        """Load and split documents from a given path."""
        loader = TextLoader(path)
        mydatas = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
        all_splits = text_splitter.split_documents(mydatas)
        return all_splits

    def get_llm_model(self, model:str):
        """Initialize and return the LLM model."""
        return ChatGoogleGenerativeAI(model=model)
    
    def get_RAG(self, documents: tp.List[str]):
        """Initialize and return the RAG instance."""
        rag = RAG(self.embedding_encoder, retriever_search_kwargs={"k":3})
        rag.set_documents2DB(documents)
        return rag

    def get_agent_prompt(self, prompt: str):
        """Create and return the agent prompt template."""
        return ChatPromptTemplate.from_messages([
            MessagesPlaceholder('chat_history'),
            ('human', prompt)
        ])
    
    def get_history_aware_prompt(self, prompt: str):
        """Create and return the history-aware prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def input_message_preprocess(self, message:str, history:tp.List[AnyMessage]):
        """Preprocess the input message and history."""
        return {"input": message,
                "chat_history": history}

    def set_chainflow(self) -> None:
        """Set up the chain flow using the LLM model, RAG retriever, and prompts."""
        history_aware_retriever = create_history_aware_retriever(self.LLM_model, self.rag.get_retriever(), self.history_prompt)
        question_answer_chain = create_stuff_documents_chain(self.LLM_model, self.agent_prompt)
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.chain = chain