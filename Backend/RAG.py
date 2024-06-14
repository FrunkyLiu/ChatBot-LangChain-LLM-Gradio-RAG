from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import typing as tp


class RAG:
    def __init__(self,
                 embedding_encodr, 
                 retriever_search_kwargs: tp.Dict[str, str] = {}):
        """Define the RAG class"""
        self.embedding_encoder = embedding_encodr
        self.retriever_search_kwargs = retriever_search_kwargs
        self.db = None

    def initialize_db(self):
        """Initialize the FAISS database."""

        if self.db:
            db = self.db
        else:
            dimensions: int = len(self.embedding_encoder.embed_query('test'))
            db = FAISS(
                embedding_function=self.embedding_encoder,
                index=IndexFlatL2(dimensions),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.db = db
        return db
    
    def set_texts2DB(self, texts: tp.List[str]):
        """Add texts to the FAISS database."""
        db = self.initialize_db()
        len_texts = len(texts)
        for idx in range(0, len_texts, 100):
            t = texts[idx:idx+100]
            db.add_texts(t)
        return

    def set_documents2DB(self, documents: tp.List):
        """Add documents to the FAISS database."""
        db = self.initialize_db()
        len_documents = len(documents)
        for idx in range(0, len_documents, 100):
            d = documents[idx:idx+100]
            db.add_documents(d)
        return

    def get_retriever(self):
        """Return a retriever instance based on the FAISS database."""
        assert self.db is not None, "Please set up the database first."
        return self.db.as_retriever(search_kwargs=self.retriever_search_kwargs)