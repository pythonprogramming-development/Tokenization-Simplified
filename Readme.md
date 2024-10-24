pip freeze > requirements.txt
pip install packagename
before that create virtual environment so that the packages cannot conflict.
pip install -r requirements.txt
python scraper.py

python -m spacy download en_core_web_sm

Tokenization Techniques-

Unigram Language Model: Models text with individual token probabilities, ignoring context.

Character Tokenization: Breaks text into characters, useful for noisy or morphologically complex languages.

N-gram Tokenization: Generates 'n' consecutive word or character sequences to capture context.

Unigram: Single tokens.

Bigram: Two-token sequences.

Trigram: Three-token sequences.

Sentence Tokenization: Divides text into sentences for sentence-level tasks.

Word-Level and Subword-Level Hybrid Tokenization: Mixes word and subword tokenization for diverse text data.

Morpheme-Based Tokenization: Breaks words into meaningful units (morphemes) for rich morphology languages.

Treebank Tokenizer: splits text into tokens while preserving the syntactic structure required for parsing.

Tweet Tokenizer: Handles social media text features like hashtags and emojis.

Multi-Word Expression Tokenizer (MWETokenizer): Treats multi-word phrases as single tokens.