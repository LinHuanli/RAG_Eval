# Financial RAG Evaluation Framework

A comprehensive, modular framework for building, tuning, and evaluating Retrieval-Augmented Generation (RAG) systems for financial documents.

## 📋 Overview

This framework automates the end-to-end process of:
1. **Processing financial documents** (PDF, CSV, etc.) into structured formats
2. **Generating representative question-answer pairs** for evaluation
3. **Implementing RAG pipelines** with configurable retrievers and generators
4. **Evaluating performance** across multiple retrieval and generation metrics

Whether you're researching RAG techniques, building production systems, or comparing LLM performance, this toolkit provides the infrastructure to rigorously measure and improve results.

## 🔍 Key Features

- **Complete Processing Pipeline**: Process PDFs and other document formats into RAG-ready formats
- **Automated QA Generation**: Create evaluation datasets without manual annotation
- **BM25 & Vector Retrieval**: Compare traditional and embedding-based approaches
- **Comprehensive Evaluation**: Measures both retrieval accuracy and answer quality
- **LLM Integration**: Works with OpenRouter-compatible APIs (Claude, GPT, etc.)
- **Flexible Configuration**: Easily adjust parameters via config files or CLI

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LinHuanli/RAG_Eavl.git
cd RAG_Eavl

# Install dependencies
pip install -r requirements.txt
```

### End-to-End Example

Process a financial document, generate QA pairs, run RAG evaluation, and analyze results:

```bash
# 1. Process a financial PDF into structured JSON
python -m main process-document \
    --input_file ./docs/financial_regulation.pdf \
    --output_file ./docs/financial_regulation.json

# 2. Generate QA pairs for evaluation
python -m main generate-qa \
    --docs_path ./docs/financial_regulation.json \
    --output_path ./docs/qa_pairs/evaluation_set.json \
    --num_samples 30

# 3. Run RAG evaluation with default BM25 retriever
python -m main run-rag \
    --docs_path ./docs/financial_regulation.json \
    --test_file_path ./docs/qa_pairs/evaluation_set.json \
    --run_name bm25_baseline \
    --save_output

# 4. Calculate and visualize evaluation metrics
python -m main evaluate \
    --results_file ./outputs/runs/bm25_baseline/rag_results.json \
    --run_name bm25_baseline
```


## 🧩 System Components

The framework is organized into modular components:

```
financial_rag_eval/
├── data/               # Document processing components
├── retrieval/          # Document retrieval implementations
├── generation/         # Answer generation and QA components
├── evaluation/         # Metrics and evaluation tools
├── config/             # Configuration management
├── utils/              # Shared utilities
└── scripts/            # Executable scripts
```

### Data Directory Structure

```
financial-rag-eval/
├── data/
│   ├── raw/            # Original documents (PDFs, etc.)
│   ├── processed/      # Processed documents (JSON)
│   └── qa_pairs/       # Generated QA pairs
├── outputs/
│   ├── runs/           # Named experiment runs 
│   │   ├── run_20250405_153206/
│   │   │   ├── rag_results.json
│   │   │   └── evaluation_metrics.json
│   └── reports/        # Analysis reports
└── logs/
    └── rag_eval.log
```

## 📊 Evaluation Metrics

The evaluation framework assesses RAG systems across multiple dimensions:

### Retrieval Accuracy
- **Recall@k**: How many relevant documents are retrieved in the top-k results
- **MRR (Mean Reciprocal Rank)**: How well the system ranks relevant documents

### Answer Quality
- **F1 Score**: Lexical overlap between generated and reference answers
- **Factuality**: LLM-evaluated accuracy of facts in generated answers

## 🛠️ Configuration

The system uses a hierarchical configuration approach:

1. **Default Configuration**: Base settings in `config/default_config.yaml`
2. **Custom Config Files**: Override defaults with your own YAML config
3. **Environment Variables**: Set values with `RAG_API_KEY`, etc.
4. **Command Line Arguments**: Highest priority for specific runs

Example of a custom configuration file:

```yaml
# custom_config.yaml
retrieval:
  method: "bm25"
  top_k: 5
  bm25:
    k1: 1.2
    b: 0.75

api:
  model: "anthropic/claude-3.5-sonnet"
  key: "${RAG_API_KEY}"  # Will be replaced from environment
```

## 📝 Advanced Usage

### Comparing Different Retrieval Methods

```bash
# Run with BM25 retrieval
python -m main run-rag \
    --docs_path ./docs/financial_regulation.json \
    --test_file ./docs/qa_pairs/evaluation_set.json \
    --run_name bm25_experiment \
    --config ./configs/bm25_config.yaml

# Run with vector retrieval
python -m main run-rag \
    --docs_path ./docs/financial_regulation.json \
    --test_file ./docs/qa_pairs/evaluation_set.json \
    --run_name vector_experiment \
    --config ./configs/vector_config.yaml

# Compare results
python -m main compare-runs \
    --run1 ./outputs/runs/bm25_experiment \
    --run2 ./outputs/runs/vector_experiment \
    --output ./outputs/reports/retrieval_comparison.pdf
```

### Processing Large Document Collections

For large document sets, use the chunking capability:

```bash
python -m main process-document \
    --input_file ./docs/large_report.pdf \
    --output_file ./docs/large_report_chunked.json \
    --chunk_size 1000 \
    --chunk_overlap 200
```

## 🤝 Contributing

Contributions are welcome!

## 📄 License

This project is licensed under the Apache-License 2.0 - see the LICENSE file for details.
