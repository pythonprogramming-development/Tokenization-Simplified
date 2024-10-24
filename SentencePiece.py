# An unsupervised text tokenizer and detokenizer mainly used for neural network-based text generation.

# Works independently of language and doesn't rely on whitespace, making it useful for languages with no clear word boundaries.

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='m.model')
tokens = sp.encode('This is a sentence.', out_type=str)
