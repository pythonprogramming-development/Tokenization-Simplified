# Handles various languages and maintains linguistic annotations like part-of-speech tagging and dependency parsing.

# Handles complex tokenization issues, such as splitting contractions and handling special cases like URLs and emails.

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
tokens = [token.text for token in doc]
print(tokens)
