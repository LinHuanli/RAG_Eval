import os
import yaml
from typing import Dict, Any, Optional
import argparse


class ConfigManager:
    """
    Centralized configuration management for the RAG evaluation system.
    Handles loading configs from files, environment variables, and command line arguments.
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a custom configuration file. If None, uses the default.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self._apply_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return {}
    
    def _apply_env_vars(self) -> None:
        """Apply environment variables to override configuration."""
        # Override API key from environment if available
        if os.environ.get("RAG_API_KEY"):
            self.config["api"]["key"] = os.environ.get("RAG_API_KEY")
        
        # Override base URL from environment if available
        if os.environ.get("RAG_BASE_URL"):
            self.config["api"]["base_url"] = os.environ.get("RAG_BASE_URL")
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Map argument names to config paths
        arg_to_config = {
            "api_key": ["api", "key"],
            "base_url": ["api", "base_url"],
            "model": ["api", "model"],
            "top_k": ["retrieval", "top_k"],
            "docs_path": ["data", "docs_path"],
            "output_path": ["data", "output_path"],
            "num_samples": ["generation", "num_samples"]
        }
        
        # Update config with command line args if provided
        for arg_name, config_path in arg_to_config.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self._set_nested_config(config_path, getattr(args, arg_name))
    
    def _set_nested_config(self, path_list: list, value: Any) -> None:
        """
        Set a value in the nested config dictionary.
        
        Args:
            path_list: List of keys defining the path in the nested dictionary
            value: Value to set
        """
        current = self.config
        for key in path_list[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path_list[-1]] = value
    
    def get(self, *path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a path of keys.
        
        Args:
            *path: Path of keys to the desired configuration value
            default: Default value to return if the path doesn't exist
            
        Returns:
            The configuration value or the default
        """
        current = self.config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current
    
    def save_config(self, path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Create a singleton instance for global access
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        The configuration manager
    """
    return config_manager