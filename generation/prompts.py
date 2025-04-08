import logging
from typing import Dict, Optional

from config.config_manager import get_config

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates for the RAG system.
    Provides a centralized way to access and customize prompts.
    """
    
    def __init__(self):
        """Initialize the prompt manager with prompts from configuration."""
        self.config = get_config()
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load all prompts from configuration."""
        # Load generation prompts
        gen_prompts = self.config.get("generation", "prompts", default={})
        if gen_prompts:
            for name, template in gen_prompts.items():
                self.prompts[name] = template
        
        # Load evaluation prompts
        eval_prompts = self.config.get("evaluation", "prompts", default={})
        if eval_prompts:
            for name, template in eval_prompts.items():
                self.prompts[name] = template
    
    def get_prompt(self, prompt_name: str, default: Optional[str] = None) -> str:
        """
        Get a prompt template by name.
        
        Args:
            prompt_name: Name of the prompt
            default: Default value if prompt is not found
            
        Returns:
            The prompt template or default value
        """
        if prompt_name not in self.prompts:
            logger.warning(f"Prompt '{prompt_name}' not found")
            return default or ""
        
        return self.prompts[prompt_name]
    
    def add_prompt(self, prompt_name: str, template: str) -> None:
        """
        Add or update a prompt template.
        
        Args:
            prompt_name: Name of the prompt
            template: The prompt template
        """
        self.prompts[prompt_name] = template
        logger.debug(f"Added prompt: {prompt_name}")


# Singleton instance for global access
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Returns:
        The prompt manager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(prompt_name: str, default: Optional[str] = None) -> str:
    """
    Convenience function to get a prompt by name.
    
    Args:
        prompt_name: Name of the prompt
        default: Default value if prompt is not found
        
    Returns:
        The prompt template
    """
    return get_prompt_manager().get_prompt(prompt_name, default)