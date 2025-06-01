import PyPDF2
import io
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Utility class for processing PDF files"""
    
    @staticmethod
    def extract_text_and_metadata(pdf_content: bytes) -> Dict[str, Any]:
        """
        Extract text content and metadata from PDF bytes
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
        """
        result = {
            "text_content": "",
            "metadata": {},
            "page_count": 0,
            "extraction_successful": False,
            "error": None,
            "processing_info": {}
        }
        
        try:
            pdf_stream = io.BytesIO(pdf_content)

            pdf_reader = PyPDF2.PdfReader(pdf_stream)

            result["page_count"] = len(pdf_reader.pages)
            result["processing_info"]["total_pages"] = len(pdf_reader.pages)

            if pdf_reader.metadata:
                metadata = {}
                for key, value in pdf_reader.metadata.items():
                    clean_key = key.lstrip('/')
                    metadata[clean_key] = str(value) if value else ""
                
                result["metadata"] = {
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "subject": metadata.get("Subject", ""),
                    "creator": metadata.get("Creator", ""),
                    "producer": metadata.get("Producer", ""),
                    "creation_date": metadata.get("CreationDate", ""),
                    "modification_date": metadata.get("ModDate", ""),
                    "keywords": metadata.get("Keywords", ""),
                    "raw_metadata": metadata 
                }

            extracted_text = []
            successful_pages = 0
            failed_pages = 0
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        extracted_text.append(f"=== Page {page_num + 1} ===\n{page_text.strip()}")
                        successful_pages += 1
                    else:
                        failed_pages += 1
                        logger.warning(f"Page {page_num + 1} appears to be empty or unreadable")
                except Exception as page_error:
                    failed_pages += 1
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                    extracted_text.append(f"=== Page {page_num + 1} ===\n[TEXT EXTRACTION FAILED: {str(page_error)}]")
            
            result["text_content"] = "\n\n".join(extracted_text)
            result["processing_info"].update({
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "extraction_method": "PyPDF2",
                "total_characters": len(result["text_content"])
            })

            if result["text_content"].strip():
                result["extraction_successful"] = True
            else:
                result["error"] = "No readable text content found in PDF"
                
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            result["extraction_successful"] = False
        
        finally:
            if 'pdf_stream' in locals():
                pdf_stream.close()
        
        return result
    
    @staticmethod
    def extract_text_only(pdf_content: bytes) -> str:
        """
        Simple method to extract just the text content from PDF
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Extracted text as string
        """
        result = PDFProcessor.extract_text_and_metadata(pdf_content)
        return result["text_content"]
    
    @staticmethod
    def get_pdf_summary(pdf_content: bytes) -> Dict[str, Any]:
        """
        Get a quick summary of PDF properties
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Summary information about the PDF
        """
        result = PDFProcessor.extract_text_and_metadata(pdf_content)
        
        return {
            "page_count": result["page_count"],
            "has_text": len(result["text_content"].strip()) > 0,
            "character_count": len(result["text_content"]),
            "title": result["metadata"].get("title", "Unknown"),
            "author": result["metadata"].get("author", "Unknown"),
            "extraction_successful": result["extraction_successful"],
            "processing_info": result["processing_info"]
        }

    @staticmethod
    def is_valid_pdf(pdf_content: bytes) -> bool:
        """
        Check if the content is a valid PDF file
        
        Args:
            pdf_content: Raw file content as bytes
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_stream = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            # Try to access pages to verify it's readable
            _ = len(pdf_reader.pages)
            pdf_stream.close()
            return True
        except Exception:
            return False

# Helper function for backward compatibility
def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF content (backward compatibility)"""
    return PDFProcessor.extract_text_only(pdf_content)
