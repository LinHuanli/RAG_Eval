import json
import math
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from retrieval.base_retriever import BaseRetriever
from config.config_manager import get_config

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Exception raised when document loading fails."""
    pass


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever implementation for document search.
    Uses the BM25 ranking function to retrieve documents based on term frequency.
    """
    
    def __init__(self, docs_path: Optional[str] = None, k1: Optional[float] = None, 
                 b: Optional[float] = None):
        """
        Initialize the BM25 retriever.
        
        Args:
            docs_path: Path to the JSON file containing documents (overrides config)
            k1: BM25 k1 parameter (overrides config)
            b: BM25 b parameter (overrides config)
        """
        config = get_config()
        
        # Initialize parameters from config or arguments
        self.docs_path = docs_path or config.get("data", "docs_path")
        self.k1 = k1 or config.get("retrieval", "bm25", "k1", default=1.5)
        self.b = b or config.get("retrieval", "bm25", "b", default=0.75)
        
        # Initialize state variables
        self.documents = []
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_freqs = {}  # df values for each term
        self.idf = {}  # idf values for each term
        self.doc_lens = []  # length of each document
        self.term_freqs = []  # term frequencies for each document
        
        # Load and preprocess documents if path is provided
        if self.docs_path:
            self.documents = self.load_documents(self.docs_path)
            self.doc_count = len(self.documents)
            self.preprocess_documents()
    
    def load_documents(self, source: str) -> List[Dict[str, Any]]:
        """
        Load documents from a JSON file.
        
        Args:
            source: Path to the JSON file
            
        Returns:
            List of document dictionaries
            
        Raises:
            DocumentLoadError: If loading or parsing fails
        """
        try:
            logger.info(f"Loading documents from {source}")
            with open(source, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Failed to load documents from {source}: {str(e)}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg)
    
    def preprocess_documents(self) -> None:
        """
        Preprocess all documents to calculate necessary statistics for BM25.
        Computes term frequencies, document frequencies, and IDF values.
        """
        logger.info("Preprocessing documents for BM25 indexing")
        
        # Initialize term frequencies for each document
        self.term_freqs = [{} for _ in range(self.doc_count)]
        self.doc_lens = []
        
        # Calculate term frequencies and document lengths
        total_len = 0
        for i, doc in enumerate(self.documents):
            try:
                content = doc.get('content', '')
                if not content:
                    logger.warning(f"Document at index {i} has no content")
                    content = ""
                
                # Tokenize document content
                tokens = self.tokenize(content)
                self.doc_lens.append(len(tokens))
                total_len += len(tokens)
                
                # Count term frequencies in this document
                for token in tokens:
                    if token not in self.term_freqs[i]:
                        self.term_freqs[i][token] = 0
                    self.term_freqs[i][token] += 1
            
            except Exception as e:
                logger.error(f"Error preprocessing document at index {i}: {str(e)}")
                # Add placeholder for failed document
                self.doc_lens.append(0)
        
        # Calculate document frequencies for each term
        self.doc_freqs = {}
        for token in set(token for doc_terms in self.term_freqs for token in doc_terms):
            self.doc_freqs[token] = sum(1 for doc_terms in self.term_freqs if token in doc_terms)
        
        # Calculate average document length
        if self.doc_count > 0:
            self.avg_doc_len = total_len / self.doc_count
        
        # Calculate IDF values for each term
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # BM25 IDF formula
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        
        logger.info("Document preprocessing complete")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split on word boundaries
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query using BM25 scoring.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of top_k documents with their scores
        """
        if not self.documents:
            logger.warning("No documents loaded. Please load documents before searching.")
            return []
        
        query_tokens = self.tokenize(query)
        scores = [0] * self.doc_count
        
        # Calculate BM25 score for each document
        for token in query_tokens:
            if token not in self.idf:
                continue
                
            for i in range(self.doc_count):
                if token not in self.term_freqs[i]:
                    continue
                    
                # Skip documents with zero length
                if self.doc_lens[i] == 0:
                    continue
                
                # BM25 scoring formula
                tf = self.term_freqs[i][token]
                doc_len = self.doc_lens[i]
                numerator = self.idf[token] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[i] += numerator / denominator
        
        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                results.append({
                    'page_id': self.documents[idx].get('page_id', idx),
                    'content': self.documents[idx].get('content', ''),
                    'score': scores[idx]
                })
        
        return results