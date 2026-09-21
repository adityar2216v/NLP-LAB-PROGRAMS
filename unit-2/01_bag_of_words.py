"""
================================================================================
Program 01: Bag of Words (BoW) Model
================================================================================

What is the Bag of Words (BoW) Model?
-------------------------------------
The Bag of Words (BoW) model is a foundational feature extraction and document 
representation technique in Natural Language Processing (NLP). It represents text 
documents as fixed-length numerical vectors based on word occurrences.

Key Characteristics:
1. Vocabulary Construction: Collects all unique words across the entire corpus.
2. Word Frequency Counting: Counts how many times each vocabulary word appears 
   in each document.
3. Disregards Grammar & Order: Word order, sentence structure, and grammar are 
   discarded, keeping only the presence and frequency ("a bag of words").
4. Types of BoW:
   - Binary BoW: 1 if the word is present in the document, 0 otherwise.
   - Count (Term Frequency) BoW: Integer count of word occurrences in the document.

Libraries Used:
- Pure Python: Built-in collections, math, and string operations (from scratch).
- scikit-learn: CountVectorizer for industry-standard vectorization.
- NLTK: Tokenization and preprocessing.
"""

import sys
import re
from collections import Counter

# ------------------------------------------------------------------------------
# 1. Dependency Checks
# ------------------------------------------------------------------------------
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
    from nltk.tokenize import word_tokenize
except ImportError:
    word_tokenize = None

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[Error] scikit-learn is not installed. Please run: pip install scikit-learn")
    sys.exit(1)


# ------------------------------------------------------------------------------
# 2. Implementation From Scratch (Pure Python)
# ------------------------------------------------------------------------------
def preprocess_text(text: str) -> list[str]:
    """Tokenizes text, converts to lowercase, and removes punctuation."""
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    return text_clean.split()


def build_bow_from_scratch(corpus: list[str]):
    """Demonstrates Bag of Words construction from first principles."""
    print("\n" + "=" * 70)
    print(" 1. BAG OF WORDS (FROM SCRATCH IMPLEMENTATION)")
    print("=" * 70)

    # Step 1: Tokenize each document
    tokenized_docs = [preprocess_text(doc) for doc in corpus]

    # Step 2: Build a sorted unique vocabulary
    vocab_set = set()
    for doc_tokens in tokenized_docs:
        vocab_set.update(doc_tokens)
    vocabulary = sorted(list(vocab_set))

    print(f"\n[A] Extracted Vocabulary (Size: {len(vocabulary)}):")
    print(" ", vocabulary)

    # Step 3: Compute Count Vectors and Binary Vectors
    count_matrix = []
    binary_matrix = []

    for doc_tokens in tokenized_docs:
        counts = Counter(doc_tokens)
        count_vec = [counts.get(word, 0) for word in vocabulary]
        binary_vec = [1 if counts.get(word, 0) > 0 else 0 for word in vocabulary]
        count_matrix.append(count_vec)
        binary_matrix.append(binary_vec)

    # Display Term Frequency (Count) Matrix
    print("\n[B] Document-Term Count Matrix:")
    header = f"{'Document':<12} | " + " | ".join(f"{word[:6]:>6}" for word in vocabulary)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, vec in enumerate(count_matrix, start=1):
        row = f"Doc {i:<8} | " + " | ".join(f"{count:>6}" for count in vec)
        print(row)
    print("-" * len(header))

    # Display Binary Matrix
    print("\n[C] Binary Bag of Words Matrix (Presence / Absence):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, vec in enumerate(binary_matrix, start=1):
        row = f"Doc {i:<8} | " + " | ".join(f"{val:>6}" for val in vec)
        print(row)
    print("-" * len(header))


# ------------------------------------------------------------------------------
# 3. Implementation Using scikit-learn (CountVectorizer)
# ------------------------------------------------------------------------------
def build_bow_with_sklearn(corpus: list[str]):
    """Demonstrates CountVectorizer from scikit-learn with standard options."""
    print("\n" + "=" * 70)
    print(" 2. BAG OF WORDS USING SCIKIT-LEARN (CountVectorizer)")
    print("=" * 70)

    # Initialize CountVectorizer
    # Standard CountVectorizer handles lowercase conversion and tokenization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Feature names (vocabulary)
    features = vectorizer.get_feature_names_out()
    vocab_mapping = vectorizer.vocabulary_

    print("\n[A] Learned Feature Names (Vocabulary):")
    print(" ", list(features))

    print("\n[B] Word to Column Index Mapping:")
    for word, idx in sorted(vocab_mapping.items(), key=lambda item: item[1]):
        print(f"    '{word}' -> column {idx}")

    # Convert sparse matrix to dense array for display
    dense_bow = X.toarray()

    print("\n[C] Dense Count Matrix:")
    header = f"{'Document':<12} | " + " | ".join(f"{word[:6]:>6}" for word in features)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(dense_bow, start=1):
        row_str = f"Doc {i:<8} | " + " | ".join(f"{val:>6}" for val in row)
        print(row_str)
    print("-" * len(header))

    # Compute Cosine Similarity between documents using BoW vectors
    cos_sim = cosine_similarity(X)
    print("\n[D] Pairwise Document Cosine Similarity (using BoW vectors):")
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            print(f"    Similarity(Doc {i+1}, Doc {j+1}) = {cos_sim[i][j]:.4f}")


# ------------------------------------------------------------------------------
# 4. CountVectorizer with Stopwords Removed
# ------------------------------------------------------------------------------
def build_bow_filtered(corpus: list[str]):
    """Demonstrates CountVectorizer with English stop-words filtering."""
    print("\n" + "=" * 70)
    print(" 3. BAG OF WORDS WITH STOPWORDS REMOVAL")
    print("=" * 70)

    vectorizer = CountVectorizer(stop_words='english')
    X_filtered = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    print("\n[A] Filtered Vocabulary (Stop words like 'is', 'and', 'the' removed):")
    print(" ", list(features))

    dense_filtered = X_filtered.toarray()
    print("\n[B] Filtered Count Matrix:")
    if len(features) > 0:
        header = f"{'Document':<12} | " + " | ".join(f"{word[:8]:>8}" for word in features)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, row in enumerate(dense_filtered, start=1):
            row_str = f"Doc {i:<8} | " + " | ".join(f"{val:>8}" for val in row)
            print(row_str)
        print("-" * len(header))


# ------------------------------------------------------------------------------
# 5. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    # Academic sample corpus
    sample_corpus = [
        "Natural language processing is fun and exciting.",
        "Language models use natural language processing.",
        "Computer vision and natural language processing are branches of AI."
    ]

    print("=" * 70)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 01: Bag of Words (BoW) Model")
    print("=" * 70)
    print("\nInput Corpus:")
    for i, doc in enumerate(sample_corpus, start=1):
        print(f"  Doc {i}: \"{doc}\"")

    # Run demonstrations
    build_bow_from_scratch(sample_corpus)
    build_bow_with_sklearn(sample_corpus)
    build_bow_filtered(sample_corpus)

    print("\n" + "=" * 70)
    print(" Execution Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
