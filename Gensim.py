# While Gensim is primarily used for topic modeling and word embeddings, it also provides basic tokenization tools.

# The comment `# Simple and fast, often used in combination with other tokenization libraries` is
# providing a brief description or summary of the purpose and characteristics of the tokenization
# tools provided by Gensim. It highlights that Gensim's tokenization tools are simple, fast, and
# commonly used alongside other tokenization libraries for various natural language processing tasks.
# Simple and fast, often used in combination with other tokenization libraries.

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_tokenize(text, min_len=1, max_len=None, remove_stopwords_flag=False):
    """
    Tokenize the input text with customizable options.
    
    :param text: Input text to tokenize
    :param min_len: Minimum length of tokens to keep (default: 1)
    :param max_len: Maximum length of tokens to keep (default: None)
    :param remove_stopwords_flag: Whether to remove stopwords (default: False)
    :return: List of tokens
    """
    if text is None or not isinstance(text, str) or text.strip() == "":
        logging.warning("Input text is empty or None. Returning empty list.")
        return []
    
    try:
        tokens = simple_preprocess(text, min_len=min_len, max_len=max_len)
        
        if remove_stopwords_flag:
            tokens = [token for token in tokens if token not in remove_stopwords(token)]
        
        logging.info(f"Tokenization completed. Number of tokens: {len(tokens)}")
        return tokens
    except Exception as e:
        logging.error(f"Error during tokenization: {str(e)}")
        return []

# Example usage
input_text = "This is a sentence."
tokens = custom_tokenize(input_text, min_len=2, max_len=10, remove_stopwords_flag=True)

print(tokens)
