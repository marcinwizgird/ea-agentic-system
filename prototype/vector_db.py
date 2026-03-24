"""
Component 1: Vector Database (ChromaDB)
Local ChromaDB with HuggingFace embeddings.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

# =============================================================================
# Imports with fallbacks
# =============================================================================

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =============================================================================
# VectorDB Class
# =============================================================================

class VectorDB:
    """ChromaDB vector database with HuggingFace embeddings."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_mock: bool = False
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_mock = use_mock
        self._mock_docs = []  # For mock mode
        
        if use_mock or not (HF_AVAILABLE and CHROMA_AVAILABLE):
            print("✓ VectorDB: Mock mode (in-memory)")
            self.store = None
            self.embeddings = None
            return
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"✓ Embeddings: {embedding_model}")
        
        # Initialize ChromaDB
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        print(f"✓ ChromaDB: {persist_directory}/{collection_name}")
    
    def add_documents(self, documents: List[Document], ids: List[str] = None) -> List[str]:
        """Add documents to the store."""
        if self.store is None:
            self._mock_docs.extend(documents)
            return [f"mock_{i}" for i in range(len(documents))]
        return self.store.add_documents(documents, ids=ids)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Add texts with metadata."""
        if self.store is None:
            for i, text in enumerate(texts):
                doc = Document(page_content=text, metadata=metadatas[i] if metadatas else {})
                self._mock_docs.append(doc)
            return [f"mock_{i}" for i in range(len(texts))]
        return self.store.add_texts(texts, metadatas=metadatas)
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if self.store is None:
            # Simple keyword matching for mock
            query_lower = query.lower()
            results = []
            for doc in self._mock_docs:
                if query_lower in doc.page_content.lower():
                    results.append(doc)
                    if len(results) >= k:
                        break
            return results[:k] if results else self._mock_docs[:k]
        return self.store.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = 5):
        """Get LangChain retriever."""
        if self.store is None:
            return None
        return self.store.as_retriever(search_kwargs={"k": k})
    
    def count(self) -> int:
        """Get document count."""
        if self.store is None:
            return len(self._mock_docs)
        try:
            return self.store._collection.count()
        except:
            return 0
    
    def clear(self):
        """Clear all documents."""
        if self.store is None:
            self._mock_docs = []
            return
        try:
            collection = self.store._collection
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
        except:
            pass


# =============================================================================
# Factory
# =============================================================================

def create_vector_db(
    collection_name: str = "org_documents",
    persist_directory: str = "./chroma_db",
    use_mock: bool = False
) -> VectorDB:
    """Create VectorDB instance."""
    # Auto-detect if dependencies available
    auto_mock = not (HF_AVAILABLE and CHROMA_AVAILABLE)
    return VectorDB(
        collection_name=collection_name, 
        persist_directory=persist_directory,
        use_mock=use_mock or auto_mock
    )


if __name__ == "__main__":
    db = create_vector_db()
    db.add_texts(["Test document"], [{"type": "test"}])
    print(f"Count: {db.count()}")
    results = db.search("test")
    print(f"Search: {len(results)} results")
