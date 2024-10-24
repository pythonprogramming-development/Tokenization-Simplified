# Supports tokenization along with part-of-speech tagging, named entity recognition, and dependency parsing.

# Offers multilingual support.

import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize')
doc = nlp("This is a sentence.")
tokens = [word.text for sentence in doc.sentences for word in sentence.words]
