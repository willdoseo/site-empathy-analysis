"""
Text Processing Utilities
=========================

Helper functions for cleaning and preparing text for empathy analysis.
"""

import re
from typing import List, Optional
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Clean text for analysis.
    
    - Removes excessive whitespace
    - Normalizes line breaks
    - Removes special characters
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()


def extract_text_from_html(html: str, include_meta: bool = True) -> str:
    """
    Extract readable text from HTML.
    
    Args:
        html: Raw HTML string
        include_meta: Include title and meta description
        
    Returns:
        Cleaned text content
    """
    if not html:
        return ""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    parts = []
    
    if include_meta:
        # Title
        title = soup.find('title')
        if title:
            parts.append(title.get_text(strip=True))
        
        # Meta description
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            parts.append(meta['content'])
    
    # Main content
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    if main_content:
        parts.append(main_content.get_text(separator=' ', strip=True))
    
    text = ' '.join(parts)
    return clean_text(text)


def chunk_text(
    text: str, 
    max_length: int = 512, 
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks for analysis.
    
    Args:
        text: Text to split
        max_length: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            for sep in ['. ', '! ', '? ', '\n']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + max_length // 2:
                    end = last_sep + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def extract_headings(html: str) -> dict:
    """
    Extract headings from HTML.
    
    Returns:
        Dict with h1, h2, h3 lists
    """
    if not html:
        return {'h1': [], 'h2': [], 'h3': []}
    
    soup = BeautifulSoup(html, 'html.parser')
    
    return {
        'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
        'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
        'h3': [h.get_text(strip=True) for h in soup.find_all('h3')],
    }


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistency.
    
    - Ensures https:// prefix
    - Removes trailing slashes
    - Lowercases domain
    """
    if not url:
        return ""
    
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove trailing slash
    url = url.rstrip('/')
    
    return url

