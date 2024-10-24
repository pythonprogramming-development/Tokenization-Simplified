# Specialized tokenizer that is used in BERT and similar transformer models.

# Handles both word-level and subword-level tokenization.

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("This is a sentence.")
