import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
import fitz  # PyMuPDF
import pandas as pd

from config.config_manager import get_config
from utils.file_utils import ensure_directory, save_json

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Exception raised for errors during document processing."""
    pass


class DocumentProcessor:
    """
    Process and transform documents from various formats into a standardized structure.
    Supports PDF, CSV, and other document formats.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.config = get_config()
    
    def process_pdf(self, pdf_path: str, output_path: str, 
                    min_page: int = 1, max_page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF document into a structured JSON format.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the processed JSON
            min_page: First page to process (1-indexed)
            max_page: Last page to process (None for all pages)
            
        Returns:
            List of document dictionaries
            
        Raises:
            DocumentProcessingError: If PDF processing fails
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            num_pages = len(pdf_document)
            
            if max_page is None:
                max_page = num_pages
            
            # Validate page range
            if min_page < 1:
                min_page = 1
            if max_page > num_pages:
                max_page = num_pages
            
            # Process each page
            documents = []
            for page_num in range(min_page - 1, max_page):
                try:
                    page = pdf_document[page_num]
                    text = page.get_text()
                    
                    # Remove excessive whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if text:
                        documents.append({
                            'page_id': page_num + 1,  # 1-indexed page numbers
                            'content': text,
                            'metadata': {
                                'source': os.path.basename(pdf_path),
                                'page_number': page_num + 1,
                                'total_pages': num_pages
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
            
            # Save the processed documents
            if output_path:
                save_json(documents, output_path)
                logger.info(f"Saved {len(documents)} documents to {output_path}")
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)
    
    def process_csv(self, csv_path: str, output_path: str, 
                    text_column: Optional[str] = None, 
                    id_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a CSV file into a structured JSON format.
        
        Args:
            csv_path: Path to the CSV file
            output_path: Path to save the processed JSON
            text_column: Column name containing the main text (None to concatenate all)
            id_column: Column name to use as document ID (None for auto-increment)
            
        Returns:
            List of document dictionaries
            
        Raises:
            DocumentProcessingError: If CSV processing fails
        """
        try:
            logger.info(f"Processing CSV: {csv_path}")
            
            # Read the CSV
            df = pd.read_csv(csv_path)
            
            documents = []
            for i, row in df.iterrows():
                try:
                    # Get document content
                    if text_column and text_column in df.columns:
                        content = str(row[text_column])
                    else:
                        # Concatenate all columns
                        content = " ".join(str(val) for val in row.values if not pd.isna(val))
                    
                    # Get document ID
                    if id_column and id_column in df.columns:
                        doc_id = row[id_column]
                    else:
                        doc_id = i + 1
                    
                    documents.append({
                        'page_id': doc_id,
                        'content': content,
                        'metadata': {
                            'source': os.path.basename(csv_path),
                            'row_index': i,
                            'columns': list(df.columns)
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error processing row {i}: {str(e)}")
            
            # Save the processed documents
            if output_path:
                save_json(documents, output_path)
                logger.info(f"Saved {len(documents)} documents to {output_path}")
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to process CSV {csv_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                        chunk_size: int = 1000, 
                        overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split large documents into smaller chunks with overlap.
        
        Args:
            documents: List of document dictionaries
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunked document dictionaries
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        
        chunked_docs = []
        for doc in documents:
            content = doc.get('content', '')
            page_id = doc.get('page_id', 0)
            metadata = doc.get('metadata', {})
            
            if len(content) <= chunk_size:
                # Document is already small enough
                chunked_docs.append(doc)
                continue
            
            # Split content into chunks
            chunk_start = 0
            chunk_index = 0
            
            while chunk_start < len(content):
                chunk_end = min(chunk_start + chunk_size, len(content))
                
                # If not at the end, try to break at a sentence boundary
                if chunk_end < len(content):
                    # Look for sentence boundaries (., !, ?)
                    for i in range(min(100, chunk_end - chunk_start)):
                        if content[chunk_end - i - 1] in ['.', '!', '?'] and (
                                chunk_end - i < len(content) and content[chunk_end - i].isspace()):
                            chunk_end = chunk_end - i
                            break
                
                # Create the chunk
                chunk_content = content[chunk_start:chunk_end].strip()
                
                # Only add non-empty chunks
                if chunk_content:
                    chunked_docs.append({
                        'page_id': f"{page_id}-{chunk_index}",
                        'content': chunk_content,
                        'metadata': {
                            **metadata,
                            'parent_id': page_id,
                            'chunk_index': chunk_index,
                            'chunk_start': chunk_start,
                            'chunk_end': chunk_end
                        }
                    })
                
                # Update for next chunk
                chunk_start = chunk_end - overlap
                chunk_index += 1
        
        return chunked_docs


# Singleton instance
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """
    Get the global document processor instance.
    
    Returns:
        The document processor instance
    """
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor