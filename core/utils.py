import os
import tiktoken
import numpy as np
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_env_variable(name, default=None):
    """Safely get environment variable with fallback."""
    value = os.environ.get(name, default)
    if value is None:
        logger.warning(f"Environment variable {name} not found.")
    return value

def setup_api_keys():
    """Set API keys from environment to ensure they're available."""
    huggingface_key = get_env_variable("HUGGINGFACE_API_TOKEN")
    groq_key = get_env_variable("GROQ_API_KEY")
    
    # Set API keys directly in environment if not already set
    if not os.environ.get("HUGGINGFACE_API_TOKEN") and huggingface_key:
        os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_key
        
    if not os.environ.get("GROQ_API_KEY") and groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    
    return {
        "huggingface_key": huggingface_key is not None,
        "groq_key": groq_key is not None
    }

def num_tokens_from_string(string, model="cl100k_base"):
    """Returns the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        logger.warning(f"Error calculating tokens: {e}")
        # Fallback to approximation if tiktoken fails
        return len(string) // 4  # Rough approximation: ~4 chars per token

def truncate_to_token_limit(texts, max_tokens, prioritize_recent=True):
    """Truncate a list of text chunks to stay under token limit.
    
    Args:
        texts: List of text strings
        max_tokens: Maximum number of tokens allowed
        prioritize_recent: If True, keep the most recent texts
        
    Returns:
        List of truncated texts that fit within token limit
    """
    if not texts:
        return []
        
    # Calculate tokens for each text
    token_counts = [num_tokens_from_string(text) for text in texts]
    total_tokens = sum(token_counts)
    
    if total_tokens <= max_tokens:
        return texts
    
    # Need to truncate
    result = []
    current_tokens = 0
    
    # If prioritizing recent, reverse the lists to process newest first
    if prioritize_recent:
        texts = list(reversed(texts))
        token_counts = list(reversed(token_counts))
    
    for text, token_count in zip(texts, token_counts):
        if current_tokens + token_count <= max_tokens:
            result.append(text)
            current_tokens += token_count
        else:
            # Try to fit partial text if space allows
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 100:  # Only if we can fit a meaningful chunk
                encoding = tiktoken.get_encoding("cl100k_base")
                encoded_text = encoding.encode(text)
                truncated_encoded = encoded_text[:remaining_tokens]
                truncated_text = encoding.decode(truncated_encoded)
                result.append(truncated_text)
            break
    
    # Return in original order if prioritizing recent
    if prioritize_recent:
        result = list(reversed(result))
        
    return result

def smart_chunk_selection(chunks, query, similarity_scores, max_tokens):
    """Select chunks based on relevance and token constraints.
    
    Args:
        chunks: List of text chunks
        query: The user query
        similarity_scores: List of similarity scores for each chunk
        max_tokens: Maximum total tokens allowed
        
    Returns:
        List of selected chunks that fit within token limit
    """
    # Calculate token count for each chunk
    token_counts = [num_tokens_from_string(chunk) for chunk in chunks]
    
    # Calculate query tokens (need to reserve space for this)
    query_tokens = num_tokens_from_string(query)
    available_tokens = max_tokens - query_tokens - 200  # Reserve some tokens for overhead
    
    if available_tokens <= 0:
        logger.warning("Query is too long, limited context available for response")
        available_tokens = max(500, max_tokens // 2)  # Ensure some minimum context
    
    # Create a list of (chunk, score, tokens) tuples
    chunk_data = list(zip(chunks, similarity_scores, token_counts))
    
    # Sort by relevance (highest score first)
    chunk_data.sort(key=lambda x: x[1], reverse=True)
    
    # Select chunks until we reach the token limit
    selected_chunks = []
    current_tokens = 0
    
    for chunk, score, tokens in chunk_data:
        if current_tokens + tokens <= available_tokens:
            selected_chunks.append(chunk)
            current_tokens += tokens
        else:
            # If we can't add the full chunk, try to break it down
            if tokens > 1000 and available_tokens - current_tokens > 500:
                # Try to fit a portion of this large chunk
                encoding = tiktoken.get_encoding("cl100k_base")
                encoded_chunk = encoding.encode(chunk)
                partial_size = available_tokens - current_tokens
                partial_chunk = encoding.decode(encoded_chunk[:partial_size])
                selected_chunks.append(partial_chunk)
            break
            
    return selected_chunks 