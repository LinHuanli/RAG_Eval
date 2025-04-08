import logging
from typing import List, Dict, Any, Optional

from generation.base_generator import BaseGenerator
from generation.prompts import get_prompt
from utils.api_client import get_api_client, APIError
from config.config_manager import get_config

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Exception raised for errors during text generation."""
    pass


class LLMGenerator(BaseGenerator):
    """
    Generator that uses a Language Model API for answer generation.
    """
    
    def __init__(self, api_client=None):
        """
        Initialize the LLM generator.
        
        Args:
            api_client: API client to use (default: get global client)
        """
        self.config = get_config()
        self.api_client = api_client or get_api_client()
    
    def generate(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate an answer to a question based on the provided context.
        
        Args:
            question: The question to answer
            context: List of context documents
            
        Returns:
            Generated answer text
            
        Raises:
            GenerationError: If answer generation fails
        """
        try:
            # Format context for the prompt
            context_text = "\n\n".join([
                f"Document (ID: {doc.get('page_id', 'unknown')}):\n{doc.get('content', '')}" 
                for doc in context
            ])
            
            # Get and format the prompt template
            prompt = get_prompt("rag_prompt").format(
                question=question,
                context=context_text
            )
            
            # Generate the answer
            logger.info(f"Generating answer for question: {question}")
            response = self.api_client.generate_completion(prompt)
            
            return response.strip()
            
        except APIError as e:
            error_msg = f"API error during answer generation: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)
    
    def generate_qa_pair(self, context: str) -> Dict[str, str]:
        """
        Generate a question-answer pair from a context document.
        
        Args:
            context: The context document text
            
        Returns:
            Dictionary with 'question' and 'answer' keys
            
        Raises:
            GenerationError: If QA pair generation fails
        """
        try:
            # Get and format the prompt template
            prompt = get_prompt("qa_generation").format(
                content=context
            )
            
            # Generate the QA pair
            logger.info("Generating QA pair from context")
            response = self.api_client.generate_completion(prompt)
            
            # Extract and parse JSON from the response
            qa_pair = self.api_client.extract_json_from_response(response)
            
            return {
                "question": qa_pair.get("question", ""),
                "answer": qa_pair.get("answer", "")
            }
            
        except APIError as e:
            error_msg = f"API error during QA pair generation: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)
        except Exception as e:
            error_msg = f"Error generating QA pair: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)
    
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
            
        Raises:
            GenerationError: If factuality evaluation fails
        """
        try:
            # Get the factuality prompt from config and format it
            prompt = get_prompt("factuality_prompt").format(
                question=question,
                reference_answer=reference_answer,
                generated_answer=generated_answer
            )
            
            # Get response from LLM
            logger.info(f"Evaluating factuality for question: {question}")
            response = self.api_client.generate_completion(prompt)
            
            # Parse the JSON response
            result = self.api_client.extract_json_from_response(response)
            factuality_score = result.get("factuality_score", 0.0)
            
            return factuality_score
            
        except APIError as e:
            error_msg = f"API error during factuality evaluation: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)
        except Exception as e:
            error_msg = f"Error evaluating factuality: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg)


# Singleton instance
_llm_generator = None


def get_llm_generator() -> LLMGenerator:
    """
    Get the global LLM generator instance.
    
    Returns:
        The LLM generator instance
    """
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMGenerator()
    return _llm_generator