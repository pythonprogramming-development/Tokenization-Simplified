import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# Remove the import for gensim
# from gensim.utils import simple_preprocess

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

def nltk_word_tokenize(text):
    """
    Tokenize text into words using NLTK's word_tokenize.
    """
    return word_tokenize(text.lower())

def nltk_sentence_tokenize(text):
    """
    Tokenize text into sentences using NLTK's sent_tokenize.
    """
    return sent_tokenize(text)

def regex_tokenize(text, pattern=r'\w+'):
    """
    Tokenize text using a regular expression pattern.
    """
    tokenizer = RegexpTokenizer(pattern)
    return tokenizer.tokenize(text.lower())

def bert_tokenize(text):
    """
    Tokenize text using BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.tokenize(text)

# Remove the gensim_tokenize function since gensim is not installed
# def gensim_tokenize(text):
#     """
#     Tokenize text using Gensim's simple_preprocess.
#     """
#     return simple_preprocess(text)

def preprocess_text(text, tokenizer=nltk_word_tokenize):
    """
    Tokenize and preprocess text data.
    """
    # Tokenization
    tokens = tokenizer(text)
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return tokens

def vectorize_text(texts, tokenizer=nltk_word_tokenize):
    """
    Convert text data to numerical vectors using bag-of-words approach.
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: preprocess_text(x, tokenizer))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_naive_bayes_classifier(X_train, y_train):
    """
    Train a Naive Bayes classifier for text classification.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def plot_feature_importance(vectorizer, clf):
    """
    Plot the importance of features (words) in the classification task.
    """
    feature_importance = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    sorted_idx = np.argsort(feature_importance)
    top_features = min(20, len(feature_importance))  # Ensure we don't exceed the number of features

    plt.figure(figsize=(10, 6))
    plt.title("Top Important Features for Classification")
    plt.barh(range(top_features), feature_importance[sorted_idx[-top_features:]])
    plt.yticks(range(top_features), np.array(vectorizer.get_feature_names_out())[sorted_idx[-top_features:]])
    plt.xlabel("Log Probability Difference")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your own dataset)
    texts = [
        "This is a positive review.",
        "I didn't like the product.",
        "Great experience, highly recommended!",
        "Terrible service, avoid at all costs."
    ]
    labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

    print("Tokenization examples:")
    example_text = "Hello, world! This is an example sentence. How are you doing today?"
    
    print("\nNLTK Word Tokenization:")
    print(nltk_word_tokenize(example_text))
    
    print("\nNLTK Sentence Tokenization:")
    print(nltk_sentence_tokenize(example_text))
    
    print("\nRegex Tokenization (words only):")
    print(regex_tokenize(example_text))
    
    print("\nBERT Tokenization:")
    print(bert_tokenize(example_text))
    
    # Remove the Gensim tokenization example
    # print("\nGensim Tokenization:")
    # print(gensim_tokenize(example_text))

    # Vectorize text data using NLTK word tokenization
    X, vectorizer = vectorize_text(texts, tokenizer=nltk_word_tokenize)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Train and evaluate the model
    clf = train_naive_bayes_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

    # Visualize feature importance
    plot_feature_importance(vectorizer, clf)

    print("\nThis example demonstrates various aspects of NLP and Machine Learning:")
    print("1. Text preprocessing and tokenization (NLP basics)")
    print("2. Feature extraction using bag-of-words (Linear Algebra)")
    print("3. Naive Bayes classification (Probability and Statistics)")
    print("4. Model evaluation (Machine Learning basics)")
    print("5. Data visualization (Exploratory Data Analysis)")

# Addtion completed To this code 