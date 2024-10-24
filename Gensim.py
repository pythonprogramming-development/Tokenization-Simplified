# While Gensim is primarily used for topic modeling and word embeddings, it also provides basic tokenization tools.

# Simple and fast, often used in combination with other tokenization libraries.

from gensim.utils import simple_preprocess

# Tokenize the text with min_len=1 to include single character tokens
tokens = simple_preprocess("This is a sentence.", min_len=1)

print(tokens)
