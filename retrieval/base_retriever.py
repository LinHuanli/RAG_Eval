from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.
    All retriever implementations should inherit from this class.
    """
    
    @abstractmethod
    def load_documents(self, source: str) -> List[Dict[str, Any]]:
        """
        Load documents from a source.
        
        Args:
            source: Source identifier (e.g., file path)
            
        Returns:
            List of document dictionaries
        """
        pass
    
    @abstractmethod
    def preprocess_documents(self) -> None:
        """
        Preprocess the loaded documents for indexing/retrieval.
        This method should be called after loading documents.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of document dictionaries with relevance scores
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        pass