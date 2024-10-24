# TreebankWordTokenizer: Tokenizes text based on the Penn Treebank conventions.

# WordPunctTokenizer: Splits text into words and punctuation.

# RegexpTokenizer: Allows tokenization based on regular expressions.

# TweetTokenizer: Specially designed to handle the tokenization of tweets.

# SentTokenizer (Punkt): Splits text into sentences.

# WhitespaceTokenizer: Splits text based on whitespace.

import nltk
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, RegexpTokenizer, TweetTokenizer, PunktSentenceTokenizer, WhitespaceTokenizer

# Sample text
text = "He said, 'Hello!' #greeting @someone 😊 How are you?"

# 1. TreebankWordTokenizer
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(text)
print("TreebankWordTokenizer:", treebank_tokens)

# 2. WordPunctTokenizer
wordpunct_tokenizer = WordPunctTokenizer()
wordpunct_tokens = wordpunct_tokenizer.tokenize(text)
print("WordPunctTokenizer:", wordpunct_tokens)

# 3. RegexpTokenizer
regexp_tokenizer = RegexpTokenizer(r'\w+')
regexp_tokens = regexp_tokenizer.tokenize(text)
print("RegexpTokenizer:", regexp_tokens)

# 4. TweetTokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)
print("TweetTokenizer:", tweet_tokens)

# 5. SentTokenizer (Punkt)
sent_tokenizer = PunktSentenceTokenizer()
sentences = sent_tokenizer.tokenize(text)
print("SentTokenizer (Punkt):", sentences)

# 6. WhitespaceTokenizer
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(text)
print("WhitespaceTokenizer:", whitespace_tokens)
