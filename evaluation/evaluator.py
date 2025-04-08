import json
import os
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from evaluation.metrics import EvaluationMetrics
from config.config_manager import get_config

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Main class for evaluating RAG system performance.
    Orchestrates the evaluation process and aggregates metrics.
    """
    
    def __init__(self, results_file: Optional[str] = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            results_file: Path to the JSON file containing RAG results
        """
        self.results_file = results_file
        self.results_data = []
        self.metrics_results = {}
        
        if results_file:
            self.load_results()
    
    def load_results(self) -> None:
        """
        Load RAG results from a JSON file.
        
        Raises:
            FileNotFoundError: If the results file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            logger.info(f"Loading results from {self.results_file}")
            with open(self.results_file, 'r', encoding='utf-8') as file:
                self.results_data = json.load(file)
            logger.info(f"Successfully loaded {len(self.results_data)} result items")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load results: {str(e)}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate RAG performance across all metrics.
        
        Returns:
            Dictionary of metric names to average scores
        """
        if not self.results_data:
            logger.warning("No results data available for evaluation")
            return {}
        
        # Initialize metrics storage
        metrics = {
            "Retrieval_Recall@1": [],
            "Retrieval_Recall@5": [],
            "Retrieval_Recall@10": [],
            "Retrieval_MRR": [],
            "Answer_F1": [],
            "Answer_Factuality": []
        }
        
        # Calculate metrics for each item
        for item in tqdm(self.results_data, desc="Evaluating RAG performance"):
            try:
                # Calculate retrieval recall at different cutoffs
                metrics["Retrieval_Recall@1"].append(
                    EvaluationMetrics.retrieval_recall(
                        item.get("retrieved_docs", []), 
                        item.get("reference_docs", []), 
                        k=1
                    )
                )
                
                metrics["Retrieval_Recall@5"].append(
                    EvaluationMetrics.retrieval_recall(
                        item.get("retrieved_docs", []), 
                        item.get("reference_docs", []), 
                        k=5
                    )
                )
                
                metrics["Retrieval_Recall@10"].append(
                    EvaluationMetrics.retrieval_recall(
                        item.get("retrieved_docs", []), 
                        item.get("reference_docs", []), 
                        k=10
                    )
                )
                
                # Calculate retrieval MRR
                metrics["Retrieval_MRR"].append(
                    EvaluationMetrics.retrieval_mrr(
                        item.get("retrieved_docs", []), 
                        item.get("reference_docs", [])
                    )
                )
                
                # Calculate answer F1
                metrics["Answer_F1"].append(
                    EvaluationMetrics.answer_f1(
                        item.get("generated_answer", ""), 
                        item.get("reference_answer", "")
                    )
                )
                
                # Calculate answer factuality
                metrics["Answer_Factuality"].append(
                    EvaluationMetrics.answer_factuality(
                        item.get("question", ""), 
                        item.get("reference_answer", ""), 
                        item.get("generated_answer", "")
                    )
                )
            
            except Exception as e:
                logger.error(f"Error evaluating item: {str(e)}")
                # Add zero scores for this item to maintain alignment
                for metric_name in metrics:
                    if metric_name not in metrics:
                        metrics[metric_name].append(0.0)
        
        # Calculate average metrics
        self.metrics_results = {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in metrics.items()
        }
        
        return self.metrics_results
    
    def save_metrics(self, output_file: Optional[str] = None) -> str:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            output_file: Path to save the metrics (if None, derives from results_file)
            
        Returns:
            Path to the saved metrics file
        """
        if not output_file and self.results_file:
            output_file = self.results_file.replace(".json", ".metrics.json")
        
        if not output_file:
            timestamp = time.strftime("%Y%m%d.%H%M%S")
            output_file = f"./outputs/evaluation.{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.metrics_results, file, indent=2)
        
        logger.info(f"Metrics saved to {output_file}")
        return output_file
    
    def print_metrics_summary(self) -> None:
        """Print a summary of the evaluation metrics to the console."""
        if not self.metrics_results:
            logger.warning("No metrics available to print")
            return
        
        print("\n===== RAG Evaluation Results =====")
        print("\nRetrieval Performance:")
        print(f"  Recall@1:  {self.metrics_results.get('Retrieval_Recall@1', 0.0):.4f}")
        print(f"  Recall@5:  {self.metrics_results.get('Retrieval_Recall@5', 0.0):.4f}")
        print(f"  Recall@10: {self.metrics_results.get('Retrieval_Recall@10', 0.0):.4f}")
        print(f"  MRR:       {self.metrics_results.get('Retrieval_MRR', 0.0):.4f}")
        
        print("\nAnswer Quality:")
        print(f"  F1 Score:    {self.metrics_results.get('Answer_F1', 0.0):.4f}")
        print(f"  Factuality:  {self.metrics_results.get('Answer_Factuality', 0.0):.4f}")
        print("\n===================================")