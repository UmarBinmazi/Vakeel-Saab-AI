import logging
from typing import List, Dict, Any, Union
import re
from core.utils import num_tokens_from_string

# Configure logging
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Advanced document chunking with token awareness and semantic boundaries."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 50,
                 respect_sections: bool = True,
                 respect_semantic_boundaries: bool = True):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to consider
            respect_sections: Whether to avoid breaking across sections
            respect_semantic_boundaries: Whether to respect semantic boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sections = respect_sections
        self.respect_semantic_boundaries = respect_semantic_boundaries
        
        # Common semantic boundary markers
        self.boundary_markers = [
            "\n\n",     # Paragraph breaks
            ". ",       # Sentence breaks
            "? ",       # Question breaks
            "! ",       # Exclamation breaks
            "; ",       # Semicolon breaks
            ":",        # Colon breaks
            ", ",       # Comma breaks
            " "         # Word breaks (last resort)
        ]
        
    def chunk_document(self, text: str, metadata: List[Dict] = None) -> List[Dict]:
        """
        Chunk document with token awareness and metadata.
        
        Args:
            text: Full document text
            metadata: Optional metadata from document processing
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            logger.warning("Empty document provided for chunking")
            return []
            
        if metadata and self.respect_sections:
            # Use section-aware chunking when metadata with sections is available
            return self._chunk_with_sections(text, metadata)
        else:
            # Use token-aware chunking for plain text
            return self._chunk_with_tokens(text)
    
    def _chunk_with_tokens(self, text: str) -> List[Dict]:
        """Create token-aware chunks from text."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Calculate tokens in this paragraph
            paragraph_tokens = num_tokens_from_string(paragraph)
            
            # If paragraph alone exceeds chunk size, split it further
            if paragraph_tokens > self.chunk_size:
                sub_chunks = self._split_large_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        "text": sub_chunk,
                        "tokens": num_tokens_from_string(sub_chunk)
                    })
                continue
                
            # Check if adding paragraph would exceed chunk size
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                # Store current chunk and start a new one
                chunks.append({
                    "text": current_chunk,
                    "tokens": current_tokens
                })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Find a good boundary for overlap
                    overlap_text = self._get_trailing_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + paragraph
                    current_tokens = num_tokens_from_string(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens = num_tokens_from_string(current_chunk)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "tokens": current_tokens
            })
            
        return chunks
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split an overly large paragraph at semantic boundaries."""
        chunks = []
        remaining_text = paragraph
        
        while remaining_text and num_tokens_from_string(remaining_text) > self.chunk_size:
            # Find an appropriate split point
            split_point = self._find_split_point(remaining_text, self.chunk_size)
            
            # Extract the chunk and update remaining text
            chunk = remaining_text[:split_point].strip()
            remaining_text = remaining_text[split_point:].strip()
            
            # Add chunk if non-empty
            if chunk:
                chunks.append(chunk)
        
        # Add any remaining text as the final chunk
        if remaining_text:
            chunks.append(remaining_text)
            
        return chunks
    
    def _find_split_point(self, text: str, target_tokens: int) -> int:
        """Find an appropriate split point near the target token count."""
        approx_chars = target_tokens * 4  # Rough approximation: ~4 chars per token
        
        # Ensure we don't exceed text length
        approx_chars = min(approx_chars, len(text))
        
        # Find the best semantic boundary near the target
        best_boundary = approx_chars
        
        if self.respect_semantic_boundaries:
            # Try different boundary markers in order of preference
            for marker in self.boundary_markers:
                # Find the last occurrence of this marker before the approximate position
                last_pos = text[:approx_chars].rfind(marker)
                
                # If found and close enough to the target, use it
                if last_pos > 0 and last_pos > approx_chars * 0.7:  # At least 70% of target
                    if marker == " ":  # For word breaks, include the space in the first chunk
                        return last_pos + 1
                    return last_pos + len(marker)  # Include the boundary marker in the first chunk
                    
        return best_boundary
    
    def _get_trailing_text(self, text: str, target_tokens: int) -> str:
        """Get trailing text from a chunk for overlap."""
        # Convert target tokens to approximate characters
        approx_chars = target_tokens * 4
        
        # Ensure we don't exceed text length
        approx_chars = min(approx_chars, len(text))
        
        # Find a good semantic boundary
        start_pos = len(text) - approx_chars
        if start_pos <= 0:
            return text
            
        if self.respect_semantic_boundaries:
            # Try to find a clean starting point
            for marker in reversed(self.boundary_markers):
                # Find the first occurrence after the start position
                next_pos = text[start_pos:].find(marker)
                if next_pos > 0:
                    start_pos += next_pos + len(marker)
                    break
        
        return text[start_pos:]
    
    def _chunk_with_sections(self, text: str, metadata: List[Dict]) -> List[Dict]:
        """Create chunks respecting document sections."""
        chunks = []
        
        # Group metadata blocks by section
        sections = {}
        for block in metadata:
            section = block.get("section", "default")
            if section not in sections:
                sections[section] = []
            sections[section].append(block)
        
        # Process each section
        for section, blocks in sections.items():
            section_text = ""
            section_metadata = []
            
            # Combine block texts
            for block in blocks:
                block_text = block.get("text", "")
                if block_text:
                    if section_text:
                        section_text += "\n\n"
                    section_text += block_text
                    section_metadata.append(block)
            
            # Skip empty sections
            if not section_text:
                continue
                
            # Check if section exceeds chunk size
            section_tokens = num_tokens_from_string(section_text)
            
            if section_tokens <= self.chunk_size:
                # Section fits in one chunk
                chunks.append({
                    "text": section_text,
                    "tokens": section_tokens,
                    "metadata": section_metadata,
                    "section": section
                })
            else:
                # Split section into multiple chunks
                sub_chunks = self._chunk_with_tokens(section_text)
                
                # Add section information to each sub-chunk
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk["section"] = section
                    sub_chunk["part"] = i + 1
                    sub_chunk["metadata"] = section_metadata
                    chunks.append(sub_chunk)
        
        return chunks
    
    def merge_small_chunks(self, chunks: List[Dict], min_size: int = None) -> List[Dict]:
        """Merge small chunks to meet minimum size requirements."""
        if not chunks:
            return []
            
        if min_size is None:
            min_size = self.min_chunk_size
            
        result = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk.copy()
                continue
                
            # Check if current chunk is too small and the same section as next chunk
            if (current_chunk.get("tokens", 0) < min_size and 
                current_chunk.get("section") == chunk.get("section")):
                # Merge chunks
                merged_text = current_chunk["text"] + "\n\n" + chunk["text"]
                merged_tokens = num_tokens_from_string(merged_text)
                
                # If merged chunk is not too large, combine them
                if merged_tokens <= self.chunk_size:
                    current_chunk["text"] = merged_text
                    current_chunk["tokens"] = merged_tokens
                    
                    # Merge metadata if available
                    if "metadata" in current_chunk and "metadata" in chunk:
                        current_chunk["metadata"] = current_chunk["metadata"] + chunk["metadata"]
                    continue
            
            # Store current chunk and start with the new one
            result.append(current_chunk)
            current_chunk = chunk.copy()
        
        # Add the last chunk if any
        if current_chunk is not None:
            result.append(current_chunk)
            
        return result 