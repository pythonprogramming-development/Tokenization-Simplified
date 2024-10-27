from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import json

# Data Corpus
corpus = [
    "The Nissan GT-R is known for its powerful twin-turbo V6 engine and advanced all-wheel-drive system.",
    "Japanese sports cars like the Toyota Supra and Mazda RX-7 have a cult following around the world.",
    "Nissan's GT-R, often called 'Godzilla,' is famous for its impressive acceleration and handling.",
    "Many Japanese cars are known for their reliability, high performance, and innovative technology.",
    "The Nissan GT-R is equipped with a dual-clutch transmission, enabling fast and seamless gear shifts.",
]

# To test Tokenisation
text = "The Nissan GT-R, also known as Godzilla, is one of Japan's most iconic sports cars with remarkable speed and precision."

# 1. BPE Tokenizer
def bpe_tokenizer(corpus, text):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=100, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(corpus, trainer)

    tokens = tokenizer.encode(text).tokens
    print("BPE Tokenizer Output:\n", tokens, "\n")
    return tokenizer

# 2. Unigram Tokenizer
def unigram_tokenizer(corpus, text):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.UnigramTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(corpus, trainer)

    tokenizer.save("unigram_tokenizer.json")
    with open("unigram_tokenizer.json", "r") as f:
        tokenizer_data = json.load(f)
        tokenizer_data["model"]["unk_id"] = tokenizer.token_to_id("[UNK]")

    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))
    tokens = tokenizer.encode(text).tokens
    print("Unigram Tokenizer Output:\n", tokens, "\n")
    return tokenizer

# 3. WordLevel Tokenizer
def wordlevel_tokenizer(corpus, text):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(corpus, trainer)

    tokens = tokenizer.encode(text).tokens
    print("WordLevel Tokenizer Output:\n", tokens, "\n")
    return tokenizer

# 4. WordPiece Tokenizer
def wordpiece_tokenizer(corpus, text):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(corpus, trainer)

    tokens = tokenizer.encode(text).tokens
    print("WordPiece Tokenizer Output:\n", tokens, "\n")
    return tokenizer

# Run the pipeline
def run_tokenization_pipeline(corpus, text):
    print("Original Text:\n", text, "\n")

    bpe = bpe_tokenizer(corpus, text)
    unigram = unigram_tokenizer(corpus, text)
    wordlevel = wordlevel_tokenizer(corpus, text)
    wordpiece = wordpiece_tokenizer(corpus, text)

    # Basic Pipeline: Tokenize text, add special tokens for NLP tasks, and prepare input IDs
    tokenized_outputs = {}
    for name, tokenizer in zip(["BPE", "Unigram", "WordLevel", "WordPiece"], [bpe, unigram, wordlevel, wordpiece]):
        encoding = tokenizer.encode(text)
        input_ids = encoding.ids
        attention_mask = [1] * len(input_ids)

        print(f"{name} Tokenizer - Input IDs:\n", input_ids)
        print(f"{name} Tokenizer - Attention Mask:\n", attention_mask, "\n")

        tokenized_outputs[name] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    return tokenized_outputs

if __name__ == "__main__":
    run_tokenization_pipeline(corpus, text)


# Working explained in Working.md