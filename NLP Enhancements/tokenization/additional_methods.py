import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

def nltk_word_tokenize(text, remove_punctuation=False, remove_stopwords=False):
    """
    Tokenize text into words using NLTK with additional options.
    
    :param text: Input text to tokenize
    :param remove_punctuation: If True, remove punctuation from tokens
    :param remove_stopwords: If True, remove stopwords from tokens
    :return: List of word tokens
    """
    if not isinstance(text, str):
        logging.warning("Input is not a string. Converting to string.")
        text = str(text)
    
    try:
        tokens = word_tokenize(text)
        
        if remove_punctuation:
            tokens = [token for token in tokens if token.isalnum()]
        
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        
        logging.info(f"Word tokenization completed. Number of tokens: {len(tokens)}")
        return tokens
    except Exception as e:
        logging.error(f"Error during word tokenization: {str(e)}")
        return []

def nltk_sentence_tokenize(text, min_length=0):
    """
    Tokenize text into sentences using NLTK with minimum length option.
    
    :param text: Input text to tokenize
    :param min_length: Minimum length of sentences to keep
    :return: List of sentence tokens
    """
    if not isinstance(text, str):
        logging.warning("Input is not a string. Converting to string.")
        text = str(text)
    
    try:
        sentences = sent_tokenize(text)
        if min_length > 0:
            sentences = [sent for sent in sentences if len(sent.split()) >= min_length]
        
        logging.info(f"Sentence tokenization completed. Number of sentences: {len(sentences)}")
        return sentences
    except Exception as e:
        logging.error(f"Error during sentence tokenization: {str(e)}")
        return []

def treebank_word_tokenize(text):
    """
    Tokenize text using the Treebank word tokenizer.
    
    :param text: Input text to tokenize
    :return: List of word tokens
    """
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)

def regex_word_tokenize(text, pattern=r'\w+'):
    """
    Tokenize text using a regular expression pattern.
    
    :param text: Input text to tokenize
    :param pattern: Regular expression pattern for tokenization
    :return: List of word tokens
    """
    tokenizer = RegexpTokenizer(pattern)
    return tokenizer.tokenize(text)

# Example usage
if __name__ == "__main__":
    sample_text = "Hello, world! This is a sample sentence. NLTK is great for NLP tasks."
    
    print("Word tokens:", nltk_word_tokenize(sample_text, remove_punctuation=True, remove_stopwords=True))
    print("Sentence tokens:", nltk_sentence_tokenize(sample_text, min_length=3))
    print("Treebank tokens:", treebank_word_tokenize(sample_text))
    print("Regex tokens:", regex_word_tokenize(sample_text))
