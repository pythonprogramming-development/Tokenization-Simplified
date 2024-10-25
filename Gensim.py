"""
Gensim is a Python library for topic modeling and document similarity analysis, 
particularly known for its efficient handling of large text corpora.

`Tokenizing Text with Gensim`
- This code leverages Gensim's simple_preprocess function to break down text into tokens, a fundamental step in many NLP tasks. 
- It provides options for filtering tokens by length and removing stop words.

"""

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def tokenize_text(text: Optional[str], min_len: int = 1, max_len: int = 15, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenizes the input text using Gensim's simple_preprocess with options for token length, stop words removal.

    Args:
        text: The input text to tokenize.
        min_len: Minimum token length (default is 1).
        max_len: Maximum token length (default is 15).
        remove_stopwords: Whether to remove stop words (default is False).

    Returns:
        List of tokens from the input text.
    """
    
    if not text:
        logging.warning("Empty or None text provided. Returning empty list of tokens.")
        return []
    
    try:
        tokens = simple_preprocess(text, min_len=min_len, max_len=max_len)
        
        # remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in STOPWORDS]
        
        return tokens
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        return []


text = "This is a sentence for tokenization, including stopwords like 'this' and 'is'."
tokens = tokenize_text(text, min_len=1, max_len=15, remove_stopwords=True)
print("Tokens:", tokens)
