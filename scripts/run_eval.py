#!/usr/bin/env python3
import argparse
import logging
import os
import json
from typing import Dict, Any, Optional

from config.config_manager import get_config, ConfigManager
from evaluation.evaluator import RAGEvaluator
from utils.logging_utils import setup_logging
from utils.api_client import get_api_client
from utils.file_utils import DirectoryManager

logger = logging.getLogger(__name__)


def evaluate_rag_results(results_file: str, output_file: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate RAG system results using multiple metrics.
    
    Args:
        results_file: Path to the JSON file with RAG results
        output_file: Path to save the evaluation metrics
        run_name: Name of the evaluation run (for organizing outputs)
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Check if results file exists
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return {}
        
        # Initialize evaluator
        logger.info(f"Initializing evaluator with results from {results_file}")
        evaluator = RAGEvaluator(results_file)
        
        # Run evaluation
        logger.info("Running evaluation on RAG results")
        metrics = evaluator.evaluate()
        
        # Determine output file path
        if output_file:
            saved_path = output_file
        elif run_name:
            # Use the run directory if a run name is provided
            run_dir = DirectoryManager.get_run_dir(run_name)
            saved_path = os.path.join(run_dir, "evaluation_metrics.json")
        else:
            # Use the default naming based on results file
            saved_path = results_file.replace(".json", ".metrics.json")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        
        # Save metrics
        with open(saved_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved evaluation metrics to {saved_path}")
        
        # Print metrics summary
        evaluator.print_metrics_summary()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return {}


def main(args=None):
    """
    Main function for the evaluation script.
    
    Args:
        args: Parsed command-line arguments (for testing/integration)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
        parser.add_argument("--results_file", type=str, required=True, 
                           help="Path to RAG results file")
        parser.add_argument("--output_file", type=str, 
                           help="Path to save metrics (default: derived from results file)")
        parser.add_argument("--run_name", type=str,
                           help="Name of the evaluation run (for organizing outputs)")
        parser.add_argument("--api_key", type=str, help="API key for factuality evaluation")
        parser.add_argument("--base_url", type=str, help="API base URL")
        parser.add_argument("--model", type=str, help="Model to use for factuality evaluation")
        parser.add_argument("--log_level", type=str, 
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                          help="Logging level")
        parser.add_argument("--config", type=str, help="Path to custom config file")
        args = parser.parse_args()
    
    # Initialize configuration
    config = get_config()
    if hasattr(args, "config") and args.config:
        config = ConfigManager(args.config)
    
    # Update config with command line args
    config.update_from_args(args)
    
    # Setup logging
    setup_logging(args.log_level if hasattr(args, "log_level") else None)
    
    try:
        # Add debug logging
        logger.info(f"Evaluating results file: {args.results_file}")
        if args.output_file:
            logger.info(f"Output will be saved to: {args.output_file}")
        if hasattr(args, "run_name") and args.run_name:
            logger.info(f"Using run name: {args.run_name}")

        # Check if results file exists
        if not os.path.exists(args.results_file):
            logger.error(f"Results file not found: {args.results_file}")
            return 1
        
        # Run evaluation with run_name if available
        run_name = args.run_name if hasattr(args, "run_name") else None
        metrics = evaluate_rag_results(args.results_file, args.output_file, run_name)
        
        if not metrics:
            logger.error("Evaluation failed to produce any metrics")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())