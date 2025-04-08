#!/usr/bin/env python3
import argparse
import logging
import random
from typing import List, Dict, Any
from tqdm import tqdm

from config.config_manager import get_config
from utils.file_utils import load_json, save_json, ensure_directory
from utils.logging_utils import setup_logging
from generation.llm_generator import get_llm_generator
from utils.api_client import get_api_client

logger = logging.getLogger(__name__)


def load_documents(docs_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSON file.
    
    Args:
        docs_path: Path to the JSON file with documents
        
    Returns:
        List of document dictionaries
    """
    logger.info(f"Loading documents from {docs_path}")
    documents = load_json(docs_path)
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def sample_pages(documents: List[Dict[str, Any]], num_samples: int, 
                max_page_id: int) -> List[Dict[str, Any]]:
    """
    Sample a specified number of pages from documents with page_id <= max_page_id.
    
    Args:
        documents: List of document dictionaries
        num_samples: Number of pages to sample
        max_page_id: Maximum page_id to consider
        
    Returns:
        List of sampled document dictionaries
    """
    eligible_docs = [doc for doc in documents if doc.get('page_id', 0) <= max_page_id]
    
    if not eligible_docs:
        logger.warning(f"No eligible documents found with page_id <= {max_page_id}")
        return []
    
    if len(eligible_docs) < num_samples:
        logger.warning(f"Only {len(eligible_docs)} eligible documents available, "
                      f"less than requested {num_samples}")
        return eligible_docs
    
    logger.info(f"Sampling {num_samples} documents from {len(eligible_docs)} eligible documents")
    return random.sample(eligible_docs, num_samples)


def generate_qa_pairs(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate QA pairs from a list of documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of QA pair dictionaries
    """
    generator = get_llm_generator()
    qa_pairs = []
    
    for doc in tqdm(documents, desc="Generating QA pairs"):
        try:
            content = doc.get('content', '')
            page_id = doc.get('page_id', 'unknown')
            
            if not content:
                logger.warning(f"Skipping document with ID {page_id}: empty content")
                continue
            
            # Generate QA pair from the document content
            qa_pair = generator.generate_qa_pair(content)
            
            # Create a QA entry with the specified format
            qa_entry = {
                "question": qa_pair.get("question", ""),
                "answer": qa_pair.get("answer", ""),
                "doc_info": [
                    {
                        "page_id": page_id,
                        "content": content
                    }
                ]
            }
            
            # Only add entries with valid questions and answers
            if qa_entry["question"] and qa_entry["answer"]:
                qa_pairs.append(qa_entry)
            else:
                logger.warning(f"Skipping document with ID {page_id}: failed to generate valid QA pair")
                
        except Exception as e:
            logger.error(f"Error generating QA pair for document with ID {doc.get('page_id', 'unknown')}: {str(e)}")
    
    return qa_pairs


def main(args=None):
    """
    Main function for the QA pair generation script.
    
    Args:
        args: Parsed command-line arguments (for testing/integration)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Generate QA pairs from documents")
        parser.add_argument("--docs_path", type=str, help="Path to documents JSON file")
        parser.add_argument("--output_path", type=str, help="Output path for QA pairs")
        parser.add_argument("--num_samples", type=int, help="Number of pages to sample")
        parser.add_argument("--max_page_id", type=int, help="Maximum page ID to consider")
        parser.add_argument("--api_key", type=str, help="API key")
        parser.add_argument("--base_url", type=str, help="API base URL")
        parser.add_argument("--model", type=str, help="Model to use")
        parser.add_argument("--log_level", type=str, 
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                          help="Logging level")
        parser.add_argument("--config", type=str, help="Path to custom config file")
        args = parser.parse_args()
    
    # Initialize configuration
    config = get_config()
    if hasattr(args, "config") and args.config:
        from config.config_manager import ConfigManager
        config = ConfigManager(args.config)
    
    # Update config with command line args
    config.update_from_args(args)
    
    # Setup logging
    setup_logging(args.log_level if hasattr(args, "log_level") else None)
    
    try:
        # Get parameters from config or args
        docs_path = args.docs_path or config.get("data", "docs_path")
        output_path = args.output_path or config.get("data", "output_path")
        num_samples = args.num_samples or config.get("generation", "num_samples", default=30)
        max_page_id = args.max_page_id or config.get("generation", "max_page_id", default=999999)
        
        if not docs_path:
            logger.error("No documents path specified")
            return 1
        
        if not output_path:
            logger.error("No output path specified")
            return 1
        
        # Create output directory if it doesn't exist
        ensure_directory(os.path.dirname(output_path))
        
        # Load and sample documents
        documents = load_documents(docs_path)
        sampled_docs = sample_pages(documents, num_samples, max_page_id)
        
        if not sampled_docs:
            logger.error("No documents to process")
            return 1
        
        # Generate QA pairs
        qa_pairs = generate_qa_pairs(sampled_docs)
        
        if not qa_pairs:
            logger.warning("No QA pairs generated")
        
        # Save QA pairs
        logger.info(f"Saving {len(qa_pairs)} QA pairs to {output_path}")
        save_json(qa_pairs, output_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during QA pair generation: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import os
    import sys
    sys.exit(main())