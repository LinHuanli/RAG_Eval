#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from config.config_manager import get_config, ConfigManager
from retrieval.bm25_retriever import BM25Retriever
from utils.api_client import get_api_client
from utils.file_utils import load_json, save_json, generate_output_path, DirectoryManager
from utils.logging_utils import setup_logging
from generation.prompts import get_prompt

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system implementation.
    Integrates document retrieval with LLM-based answer generation.
    """
    
    def __init__(self, retriever=None, api_client=None):
        """
        Initialize the RAG system.
        
        Args:
            retriever: Document retriever instance (default: create a new BM25Retriever)
            api_client: API client for LLM access (default: get global client)
        """
        config = get_config()
        self.top_k = config.get("retrieval", "top_k", default=10)
        
        # Initialize retriever
        self.retriever = retriever or BM25Retriever()
        
        # Initialize API client
        self.api_client = api_client or get_api_client()
    
    def process_query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            Dictionary with question, retrieved documents, and generated answer
        """
        k = top_k or self.top_k
        
        # Step 1: Retrieve relevant documents
        logger.info(f"Retrieving top {k} documents for question: {question}")
        retrieved_docs = self.retriever.search(question, top_k=k)
        
        # Step 2: Format retrieved documents for the prompt
        context = "\n\n".join([
            f"Document (ID: {doc['page_id']}):\n{doc['content']}" 
            for doc in retrieved_docs
        ])
        
        # Step 3: Create and format RAG prompt
        prompt = get_prompt("rag_prompt").format(
            question=question,
            context=context
        )
        
        # Step 4: Generate answer using the LLM
        logger.info("Generating answer with LLM")
        try:
            generated_answer = self.api_client.generate_completion(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            generated_answer = "Failed to generate an answer due to an error."
        
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generated_answer
        }


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Run RAG system for question answering")
        parser.add_argument("--docs_path", type=str, help="Path to documents JSON file")
        parser.add_argument("--test_file_path", type=str, help="Path to test file with questions")
        parser.add_argument("--test_single_question", type=str, help="Ask a single question")
        parser.add_argument("--search_top_k", type=int, help="Number of documents to retrieve")
        parser.add_argument("--base_url", type=str, help="API base URL")
        parser.add_argument("--api_key", type=str, help="API key")
        parser.add_argument("--model", type=str, help="Model to use")
        parser.add_argument("--save_output", action="store_true", help="Whether to save the results")
        parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
        parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                            help="Logging level")
        parser.add_argument("--run_name", type=str, help="Name for this evaluation run")
        args = parser.parse_args()
    
    # Setup configuration and logging
    config_manager = get_config()
    config_manager.update_from_args(args)
    setup_logging(args.log_level)
    
    logger.info(f"Running RAG with docs_path: {args.docs_path}")
    logger.info(f"Test file path: {args.test_file_path}")
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Save output: {args.save_output}")
    
    # Check if files exist
    if args.docs_path and not os.path.exists(args.docs_path):
        logger.error(f"Documents file not found: {args.docs_path}")
        return 1
        
    if args.test_file_path and not os.path.exists(args.test_file_path):
        logger.error(f"Test file not found: {args.test_file_path}")
        return 1
    
    # Initialize retriever with documents
    docs_path = args.docs_path or config_manager.get("data", "docs_path")
    retriever = BM25Retriever(docs_path=docs_path)
    
    # Initialize RAG system
    rag_system = RAGSystem(retriever=retriever)
    
    # Load test data or use single question
    if args.test_file_path:
        logger.info(f"Loading test questions from {args.test_file_path}")
        test_data = load_json(args.test_file_path)
        questions = [item["question"] for item in test_data]
        answers = [item.get("answer", "") for item in test_data]
        reference_docs = [item.get("doc_info", []) for item in test_data]
    elif args.test_single_question:
        logger.info(f"Using single test question: {args.test_single_question}")
        questions = [args.test_single_question]
        answers = [""]
        reference_docs = [[]]
    else:
        logger.error("No test file or single question provided")
        parser.print_help()
        return 1
    
    # Process each question
    results = []
    for question in tqdm(questions, desc="Processing queries"):
        result = rag_system.process_query(question, top_k=args.search_top_k)
        results.append(result)
    
    # Save or display results
    if args.save_output:
        if hasattr(args, "run_name") and args.run_name:
            run_dir = DirectoryManager.get_run_dir(args.run_name)
            output_file = os.path.join(run_dir, "rag_results.json")
        else:
            output_file = generate_output_path(args.output_dir, "run_rag.test")
        
        # Format output data
        output_data = []
        for i, result in enumerate(results):
            output_item = {
                "question": result['question'],
                "reference_answer": answers[i] if i < len(answers) else "",
                "generated_answer": result['generated_answer'],
                "reference_docs": reference_docs[i] if i < len(reference_docs) else [],
                "retrieved_docs": result['retrieved_docs']
            }
            output_data.append(output_item)
        
        save_json(output_data, output_file)
        logger.info(f"Results saved to {output_file}")
    else:
        # Display results in console
        for result in results:
            print("\n" + "="*50)
            print("Question:", result["question"])
            print("\nGenerated Answer:")
            print(result["generated_answer"])
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["retrieved_docs"][:3]):  # Show only top 3 for brevity
                print(f"Document {i+1} (ID: {doc['page_id']}, Score: {doc['score']:.4f}):")
                # Truncate long content for display
                content = doc['content']
                print(content[:200] + "..." if len(content) > 200 else content)
                print("-" * 30)
    
    return 0


if __name__ == "__main__":
    exit(main())