from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseGenerator(ABC):
    """
    Abstract base class for answer generators.
    All generator implementations should inherit from this class.
    """
    
    @abstractmethod
    def generate(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate an answer to a question based on the provided context.
        
        Args:
            question: The question to answer
            context: List of context documents
            
        Returns:
            Generated answer text
        """
        pass
    
    @abstractmethod
    def generate_qa_pair(self, context: str) -> Dict[str, str]:
        """
        Generate a question-answer pair from a context document.
        
        Args:
            context: The context document text
            
        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        pass
    
    @abstractmethod
    def evaluate_factuality(self, question: str, reference_answer: str, 
                           generated_answer: str) -> float:
        """
        Evaluate the factual accuracy of a generated answer.
        
        Args:
            question: The question
            reference_answer: The reference answer
            generated_answer: The generated answer to evaluate
            
        Returns:
            Factuality score between 0 and 1
        """
        pass