import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI

from config.config_manager import get_config

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class APIClient:
    """
    Unified client for interacting with LLM APIs.
    Handles authentication, rate limiting, and error handling.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: API base URL (overrides config)
            api_key: API key (overrides config)
            model: Model name (overrides config)
        """
        config = get_config()
        
        self.base_url = base_url or config.get("api", "base_url")
        self.api_key = api_key or config.get("api", "key")
        self.model = model or config.get("api", "model")
        
        if not self.api_key:
            raise APIError("API key is required. Set it via environment variable, config file, or constructor.")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Rate limiting parameters
        self.max_retries = 3
        self.initial_retry_delay = 2
        self.max_retry_delay = 30
    
    def generate_completion(self, prompt: str, 
                           headers: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a completion using the configured LLM.
        
        Args:
            prompt: Text prompt to send to the LLM
            headers: Optional additional HTTP headers
            
        Returns:
            The generated text completion
            
        Raises:
            APIError: If the API call fails after retries
        """
        default_headers = {
            "HTTP-Referer": "https://rag-evaluation.com",
            "X-Title": "RAG Evaluation Tool",
        }
        
        # Merge default headers with any custom headers
        if headers:
            default_headers.update(headers)
        
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers=default_headers,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return completion.choices[0].message.content
            
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff with jitter
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                else:
                    raise APIError(f"API call failed after {self.max_retries} attempts: {str(e)}")
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from an API response.
        
        Args:
            response: Text response from the API
            
        Returns:
            Parsed JSON data
            
        Raises:
            APIError: If JSON extraction or parsing fails
        """
        try:
            # Find JSON content - look for content between triple backticks if present
            if "```json" in response and "```" in response.split("```json")[1]:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response and "```" in response.split("```")[1]:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response
            
            # Parse the JSON
            return json.loads(json_str)
            
        except (json.JSONDecodeError, IndexError) as e:
            raise APIError(f"Failed to extract JSON from response: {str(e)}")


# Singleton instance for global access
_api_client_instance = None


def get_api_client() -> APIClient:
    """
    Get the global API client instance, creating it if necessary.
    
    Returns:
        The API client instance
    """
    global _api_client_instance
    if _api_client_instance is None:
        _api_client_instance = APIClient()
    return _api_client_instance