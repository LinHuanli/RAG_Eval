#!/usr/bin/env python3
import argparse
import logging
import sys

from config import get_config
from utils.logging_utils import setup_logging


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the main CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Financial RAG Evaluation System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Document processing command
    doc_parser = subparsers.add_parser("process-document", help="Process documents (PDF, CSV) into JSON format")
    doc_parser.add_argument("--input_file", type=str, required=True, help="Path to input document (PDF, CSV)")
    doc_parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    doc_parser.add_argument("--min_page", type=int, help="First page to process (for PDFs, 1-indexed)")
    doc_parser.add_argument("--max_page", type=int, help="Last page to process (for PDFs)")
    doc_parser.add_argument("--chunk_size", type=int, help="Maximum characters per chunk (if chunking)")
    doc_parser.add_argument("--chunk_overlap", type=int, help="Character overlap between chunks")
    doc_parser.add_argument("--text_column", type=str, help="Column name containing text (for CSVs)")
    doc_parser.add_argument("--id_column", type=str, help="Column name to use as document ID (for CSVs)")
    
    # QA generation command
    gen_parser = subparsers.add_parser("generate-qa", help="Generate QA pairs from documents")
    gen_parser.add_argument("--docs_path", type=str, help="Path to documents JSON file")
    gen_parser.add_argument("--output_path", type=str, help="Output path for QA pairs")
    gen_parser.add_argument("--num_samples", type=int, help="Number of pages to sample")
    gen_parser.add_argument("--max_page_id", type=int, help="Maximum page ID to consider")
    gen_parser.add_argument("--api_key", type=str, help="API key")
    gen_parser.add_argument("--model", type=str, help="Model to use")
    
    # RAG command
    rag_parser = subparsers.add_parser("run-rag", help="Run RAG system on questions")
    rag_parser.add_argument("--docs_path", type=str, help="Path to documents JSON file")
    rag_parser.add_argument("--test_file_path", type=str, help="Path to test questions file")
    rag_parser.add_argument("--test_single_question", type=str, help="Single question to test")
    rag_parser.add_argument("--search_top_k", type=int, help="Number of documents to retrieve")
    rag_parser.add_argument("--api_key", type=str, help="API key")
    rag_parser.add_argument("--model", type=str, help="Model to use")
    rag_parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    rag_parser.add_argument("--run_name", type=str, help="Name for this evaluation run")
    rag_parser.add_argument("--save_output", action="store_true", help="Whether to save the results")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate RAG system performance")
    eval_parser.add_argument("--results_file", type=str, required=True, 
                           help="Path to RAG results file")
    eval_parser.add_argument("--output_file", type=str, help="Path to save metrics")
    eval_parser.add_argument("--run_name", type=str, help="Name of the evaluation run (for organizing outputs)")
    eval_parser.add_argument("--api_key", type=str, help="API key for factuality evaluation")
    eval_parser.add_argument("--model", type=str, help="Model to use for factuality evaluation")
    
    # Compare runs command
    compare_parser = subparsers.add_parser("compare-runs", help="Compare multiple RAG evaluation runs")
    compare_parser.add_argument("--run1", type=str, required=True, help="Path to first run directory")
    compare_parser.add_argument("--run2", type=str, required=True, help="Path to second run directory")
    compare_parser.add_argument("--output", type=str, help="Path to save comparison report")
    compare_parser.add_argument("--names", type=str, nargs=2, help="Names for the runs in the comparison")
    
    # Common options for all parsers
    for subparser in [doc_parser, gen_parser, rag_parser, eval_parser, compare_parser]:
        subparser.add_argument("--config", type=str, help="Path to custom config file")
        subparser.add_argument("--log_level", type=str, 
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                            help="Logging level")
    
    return parser


def main():
    """Main entry point for the application."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize configuration and logging
    if hasattr(args, "config") and args.config:
        from config.config_manager import ConfigManager
        config_manager = ConfigManager(args.config)
    else:
        config_manager = get_config()
    
    # Update config with command line args
    config_manager.update_from_args(args)
    
    # Setup logging
    log_level = args.log_level if hasattr(args, "log_level") else None
    setup_logging(log_level)
    
    # Run the requested command
    if args.command == "process-document":
        # Handle document processing
        from data.document_processor import get_document_processor
        
        try:
            processor = get_document_processor()
            input_file = args.input_file
            output_file = args.output_file
            
            # Determine file type based on extension
            if input_file.lower().endswith('.pdf'):
                # Process PDF
                min_page = args.min_page if hasattr(args, "min_page") and args.min_page is not None else 1
                max_page = args.max_page if hasattr(args, "max_page") and args.max_page is not None else None
                
                docs = processor.process_pdf(input_file, output_file, min_page, max_page)
                
                # Apply chunking if requested
                if hasattr(args, "chunk_size") and args.chunk_size:
                    chunk_size = args.chunk_size
                    chunk_overlap = args.chunk_overlap or 200
                    chunked_docs = processor.chunk_documents(docs, chunk_size, chunk_overlap)
                    
                    # Save chunked documents
                    chunked_output = output_file.replace('.json', '_chunked.json')
                    from utils.file_utils import save_json
                    save_json(chunked_docs, chunked_output)
                    logging.info(f"Saved {len(chunked_docs)} chunked documents to {chunked_output}")
                
                logging.info(f"Processed {len(docs)} pages from {input_file}")
                return 0
                
            elif input_file.lower().endswith('.csv'):
                # Process CSV
                text_column = args.text_column if hasattr(args, "text_column") else None
                id_column = args.id_column if hasattr(args, "id_column") else None
                
                docs = processor.process_csv(input_file, output_file, text_column, id_column)
                logging.info(f"Processed {len(docs)} rows from {input_file}")
                return 0
                
            else:
                logging.error(f"Unsupported file type: {input_file}")
                return 1
                
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}", exc_info=True)
            return 1
            
    elif args.command == "generate-qa":
        from scripts.run_gen_qa_pairs import main as gen_main
        return gen_main(args)
    
    elif args.command == "run-rag":
        from scripts.run_rag import main as rag_main
        return rag_main(args)
    
    elif args.command == "evaluate":
        from scripts.run_eval import main as eval_main
        return eval_main(args)
    
    elif args.command == "compare-runs":
        # Handle run comparison
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            import json
            import os
            
            # Load metrics from both runs
            run1_metrics_file = os.path.join(args.run1, "evaluation_metrics.json")
            run2_metrics_file = os.path.join(args.run2, "evaluation_metrics.json")
            
            if not os.path.exists(run1_metrics_file) or not os.path.exists(run2_metrics_file):
                logging.error("Could not find metrics files in one or both run directories")
                return 1
            
            with open(run1_metrics_file, 'r') as f:
                run1_metrics = json.load(f)
            
            with open(run2_metrics_file, 'r') as f:
                run2_metrics = json.load(f)
            
            # Use provided names or derive from directories
            run1_name = args.names[0] if hasattr(args, "names") and args.names and len(args.names) > 0 else os.path.basename(args.run1)
            run2_name = args.names[1] if hasattr(args, "names") and args.names and len(args.names) > 1 else os.path.basename(args.run2)
            
            # Create comparison dataframe
            metrics_df = pd.DataFrame({
                'Metric': list(run1_metrics.keys()),
                run1_name: list(run1_metrics.values()),
                run2_name: list(run2_metrics.values()),
                'Difference': [run2_metrics[k] - run1_metrics[k] for k in run1_metrics.keys()]
            })
            
            # Create comparison chart
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(metrics_df['Metric']))
            width = 0.35
            
            # Create grouped bar chart
            plt.bar(x - width/2, metrics_df[run1_name], width, label=run1_name)
            plt.bar(x + width/2, metrics_df[run2_name], width, label=run2_name)
            
            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('Comparison of RAG Evaluation Results')
            plt.xticks(x, metrics_df['Metric'], rotation=45)
            plt.legend()
            
            plt.tight_layout()
            
            # Save comparison
            from utils.file_utils import DirectoryManager
            output_file = args.output or DirectoryManager.get_report_path(f"comparison_{run1_name}_vs_{run2_name}.png")
            plt.savefig(output_file)
            logging.info(f"Saved comparison chart to {output_file}")
            
            # Print comparison table
            print("\nComparison of Metrics:")
            print(metrics_df.to_string(index=False))
            
            # Save comparison as CSV
            csv_file = output_file.replace('.png', '.csv')
            metrics_df.to_csv(csv_file, index=False)
            logging.info(f"Saved comparison data to {csv_file}")
            
            return 0
            
        except Exception as e:
            logging.error(f"Error comparing runs: {str(e)}", exc_info=True)
            return 1
    
    else:
        logging.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())