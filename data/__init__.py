"""
Document processing components for the Financial RAG Evaluation System.
"""

from data.document_processor import (
    DocumentProcessor,
    get_document_processor,
    DocumentProcessingError
)

__all__ = ['DocumentProcessor', 'get_document_processor', 'DocumentProcessingError']