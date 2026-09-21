"""
================================================================================
Program 02: Term Frequency - Inverse Document Frequency (TF-IDF)
================================================================================

What is TF-IDF?
---------------
TF-IDF (Term Frequency - Inverse Document Frequency) is a numerical weighting 
statistic used in information retrieval and text mining. It reflects how important 
a word is to a document in a collection or corpus.

Why use TF-IDF over Bag of Words?
---------------------------------
In standard Bag of Words, common words (like "the", "is", "a") or high-frequency 
domain words appear with large counts even if they provide little discriminative 
information. TF-IDF downweights words that appear frequently across all documents 
and elevates words that are distinctive to a specific document.

Mathematical Formulas:
----------------------
1. Term Frequency (TF):
   Measures the frequency of term 't' in document 'd'.
   - Normalized TF:
       TF(t, d) = (Number of times t appears in d) / (Total number of words in d)
   - Or Raw Count TF:
       TF(t, d) = count(t, d)

2. Inverse Document Frequency (IDF):
   Measures how rare or informative a word is across the corpus of N documents.
   - Standard Formula:
       IDF(t) = log(N / DF(t))
     where DF(t) is Document Frequency (number of documents containing term t).
   - Scikit-learn Smooth IDF Formula:
       IDF(t) = log((1 + N) / (1 + DF(t))) + 1
     (Prevents division by zero and ensures words with DF=N still have non-zero weight).

3. TF-IDF Score:
   TF-IDF(t, d) = TF(t, d) * IDF(t)

4. L2 (Cosine) Normalization:
   Normalizes the vector so each document vector has a Euclidean norm of 1:
   v_norm = v / sqrt(sum(v_i^2))
"""

import sys
import re
import math
import numpy as np

# ------------------------------------------------------------------------------
# 1. Dependency Checks
# ------------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[Error] scikit-learn is not installed. Please run: pip install scikit-learn")
    sys.exit(1)


# ------------------------------------------------------------------------------
# 2. Step-by-Step Implementation from Scratch
# ------------------------------------------------------------------------------
def clean_and_tokenize(text: str) -> list[str]:
    """Cleans punctuation and returns lowercase tokens."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return cleaned.split()


def compute_tfidf_from_scratch(corpus: list[str]):
    """Demonstrates step-by-step computation of TF, IDF, and TF-IDF."""
    print("\n" + "=" * 70)
    print(" 1. TF-IDF COMPUTATION FROM FIRST PRINCIPLES (FROM SCRATCH)")
    print("=" * 70)

    # Step A: Tokenization & Vocabulary
    docs_tokens = [clean_and_tokenize(doc) for doc in corpus]
    N = len(corpus)

    vocab = sorted(list({word for doc in docs_tokens for word in doc}))
    print(f"\n[A] Corpus Vocabulary ({len(vocab)} unique terms):")
    print(" ", vocab)

    # Step B: Term Frequency (Normalized: count / total_words_in_doc)
    print("\n[B] Normalized Term Frequency (TF) Table:")
    tf_matrix = []
    header = f"{'Document':<10} | " + " | ".join(f"{w[:6]:>6}" for w in vocab)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for idx, doc in enumerate(docs_tokens, start=1):
        total_words = len(doc)
        doc_tf = {}
        for word in vocab:
            count = doc.count(word)
            doc_tf[word] = count / total_words if total_words > 0 else 0.0
        tf_matrix.append(doc_tf)

        row_str = f"Doc {idx:<6} | " + " | ".join(f"{doc_tf[w]:>6.3f}" for w in vocab)
        print(row_str)
    print("-" * len(header))

    # Step C: Document Frequency (DF) and Smooth IDF Calculation
    # Using smooth formula: IDF(t) = log((1 + N) / (1 + DF(t))) + 1
    print("\n[C] Document Frequency (DF) & Smooth IDF Values:")
    idf_values = {}
    for word in vocab:
        df = sum(1 for doc in docs_tokens if word in doc)
        idf = math.log((1 + N) / (1 + df)) + 1.0
        idf_values[word] = idf
        print(f"    Term: {word:<12} | DF: {df}/{N} | IDF: {idf:.4f}")

    # Step D: Raw TF-IDF Matrix (TF * IDF)
    print("\n[D] Raw TF-IDF Matrix (TF * IDF):")
    raw_tfidf_matrix = []
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for idx, doc_tf in enumerate(tf_matrix, start=1):
        row = [doc_tf[w] * idf_values[w] for w in vocab]
        raw_tfidf_matrix.append(row)
        row_str = f"Doc {idx:<6} | " + " | ".join(f"{val:>6.3f}" for val in row)
        print(row_str)
    print("-" * len(header))

    # Step E: L2 Normalization
    print("\n[E] L2-Normalized TF-IDF Matrix (Unit Norm Vectors):")
    norm_matrix = []
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(raw_tfidf_matrix, start=1):
        norm = math.sqrt(sum(x ** 2 for x in row))
        normalized_row = [x / norm if norm > 0 else 0.0 for x in row]
        norm_matrix.append(normalized_row)
        row_str = f"Doc {idx:<6} | " + " | ".join(f"{val:>6.3f}" for val in normalized_row)
        print(row_str)
    print("-" * len(header))


# ------------------------------------------------------------------------------
# 3. Implementation Using scikit-learn (TfidfVectorizer)
# ------------------------------------------------------------------------------
def compute_tfidf_with_sklearn(corpus: list[str]):
    """Demonstrates industry-standard TF-IDF calculation using scikit-learn."""
    print("\n" + "=" * 70)
    print(" 2. TF-IDF USING SCIKIT-LEARN (TfidfVectorizer)")
    print("=" * 70)

    # Initialize TfidfVectorizer with smooth_idf and L2 norm (defaults)
    vectorizer = TfidfVectorizer(smooth_idf=True, norm='l2')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    print("\n[A] Learned Feature Names:")
    print(" ", list(feature_names))

    print("\n[B] Learned IDF Weights (scikit-learn):")
    for feature, idf in zip(feature_names, vectorizer.idf_):
        print(f"    {feature:<12} -> IDF: {idf:.4f}")

    # Convert sparse matrix to dense array
    dense_matrix = tfidf_matrix.toarray()

    print("\n[C] Scikit-Learn TF-IDF Matrix:")
    header = f"{'Document':<10} | " + " | ".join(f"{w[:6]:>6}" for w in feature_names)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(dense_matrix, start=1):
        row_str = f"Doc {i:<6} | " + " | ".join(f"{val:>6.3f}" for val in row)
        print(row_str)
    print("-" * len(header))

    # Keyword Extraction: Top 2 salient terms per document
    print("\n[D] Top Keywords Extracted per Document:")
    for i, doc in enumerate(corpus, start=1):
        scores = dense_matrix[i - 1]
        top_indices = np.argsort(scores)[::-1][:2]
        keywords = [(feature_names[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        keyword_str = ", ".join(f"'{k}' ({s:.3f})" for k, s in keywords)
        print(f"    Doc {i}: {keyword_str}")

    # Document Cosine Similarity based on TF-IDF representation
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print("\n[E] Pairwise Document Cosine Similarity Matrix:")
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            print(f"    Similarity(Doc {i+1}, Doc {j+1}) = {similarity_matrix[i][j]:.4f}")


# ------------------------------------------------------------------------------
# 4. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    # Academic corpus with distinct topic overlaps
    sample_corpus = [
        "Data science involves machine learning and statistics.",
        "Machine learning models require data preprocessing.",
        "Deep learning is a subset of machine learning and artificial intelligence."
    ]

    print("=" * 70)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 02: TF-IDF (Term Frequency - Inverse Document Frequency)")
    print("=" * 70)

    print("\nInput Corpus:")
    for i, doc in enumerate(sample_corpus, start=1):
        print(f"  Doc {i}: \"{doc}\"")

    # Run demonstrations
    compute_tfidf_from_scratch(sample_corpus)
    compute_tfidf_with_sklearn(sample_corpus)

    print("\n" + "=" * 70)
    print(" Execution Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
