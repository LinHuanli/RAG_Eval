import re
import logging
import json
from typing import List, Dict, Any, Set, Tuple, Optional

from utils.api_client import get_api_client, APIError
from config.config_manager import get_config

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Implements various metrics for evaluating RAG system performance.
    Includes metrics for both retrieval and generation quality.
    """
    
    @staticmethod
    def retrieval_recall(retrieved_docs: List[Dict[str, Any]], 
                         reference_docs: List[Dict[str, Any]], k: int = 5) -> float:
        """
        Calculate recall@k score for retrieved documents based on page_id.
        
        Args:
            retrieved_docs: List of retrieved documents with page_id
            reference_docs: List of reference documents with page_id
            k: Number of top documents to consider
            
        Returns:
            Recall score (float)
        """
        if not reference_docs:
            logger.debug("No reference documents provided for recall calculation")
            return 0.0
        
        # Get sets of document IDs
        retrieved_ids = set(doc.get("page_id") for doc in retrieved_docs[:k] if "page_id" in doc)
        reference_ids = set(doc.get("page_id") for doc in reference_docs if "page_id" in doc)
        
        if not reference_ids:
            logger.debug("No valid reference document IDs found")
            return 0.0
        
        # Calculate recall
        relevant_retrieved = retrieved_ids.intersection(reference_ids)
        recall = len(relevant_retrieved) / len(reference_ids)
        
        return recall
    
    @staticmethod
    def retrieval_mrr(retrieved_docs: List[Dict[str, Any]], 
                      reference_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) for retrieved documents based on page_id.
        
        Args:
            retrieved_docs: List of retrieved documents with page_id
            reference_docs: List of reference documents with page_id
            
        Returns:
            MRR score (float)
        """
        if not reference_docs or not retrieved_docs:
            logger.debug("Empty document list provided for MRR calculation")
            return 0.0
        
        reference_ids = set(doc.get("page_id") for doc in reference_docs if "page_id" in doc)
        
        if not reference_ids:
            logger.debug("No valid reference document IDs found")
            return 0.0
        
        # Find the first occurrence of a reference document in the retrieved list
        for i, doc in enumerate(retrieved_docs):
            if doc.get("page_id") in reference_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def normalize_and_tokenize(text: str) -> List[str]:
        """
        Normalize and tokenize text for F1 score calculation.
        
        Args:
            text: Input text
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into tokens and remove empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]
        return tokens
    
    @classmethod
    def answer_f1(cls, generated_answer: str, reference_answer: str) -> float:
        """
        Calculate F1 score between generated answer and reference answer.
        
        Args:
            generated_answer: Generated answer string
            reference_answer: Reference answer string
            
        Returns:
            F1 score (float)
        """
        gen_tokens = cls.normalize_and_tokenize(generated_answer)
        ref_tokens = cls.normalize_and_tokenize(reference_answer)
        
        # Handle empty inputs
        if not gen_tokens or not ref_tokens:
            logger.debug("Empty token list in F1 calculation")
            return 0.0
        
        # Calculate precision, recall, and F1
        common_tokens = set(gen_tokens).intersection(set(ref_tokens))
        
        precision = len(common_tokens) / len(gen_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def answer_factuality(question: str, reference_answer: str, 
                          generated_answer: str) -> float:
        """
        Calculate factuality score using LLM to evaluate how many key facts
        from the reference are included in the generated answer.
        
        Args:
            question: Question string
            reference_answer: Reference answer string
            generated_answer: Generated answer string
            
        Returns:
            Factuality score (float)
        """
        config = get_config()
        api_client = get_api_client()
        
        # Get the factuality prompt from config and format it
        prompt_template = config.get("evaluation", "prompts", "factuality_prompt", default="")
        if not prompt_template:
            logger.error("Factuality prompt template not found in configuration")
            return 0.0
        
        prompt = prompt_template.format(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        
        try:
            # Get response from LLM
            response_text = api_client.generate_completion(prompt)
            
            # Parse the JSON response
            result = api_client.extract_json_from_response(response_text)
            factuality_score = result.get("factuality_score", 0.0)
            
            return factuality_score
            
        except APIError as e:
            logger.error(f"Error calculating factuality: {str(e)}")
            return 0.0