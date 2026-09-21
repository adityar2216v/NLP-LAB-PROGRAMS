"""
================================================================================
Program 03: N-gram Modeling and Feature Extraction
================================================================================

What are N-grams?
-----------------
An N-gram is a contiguous sequence of 'n' items (words or characters) extracted 
from a given sample of text or speech.

Classification by Order:
- Unigram (n = 1): Single words (e.g., ["natural"], ["language"], ["processing"])
- Bigram  (n = 2): Pairs of consecutive words (e.g., ["natural", "language"])
- Trigram (n = 3): Triplets of consecutive words (e.g., ["natural", "language", "processing"])
- N-gram  (n >= 4): Sequences of 4 or more consecutive words

Why are N-grams Essential in NLP?
---------------------------------
1. Context Preservation: Unlike Bag of Words which ignores word order, N-grams 
   preserve local word sequence and phrase structure (e.g., "not good" vs "good").
2. Language Modeling: Used to calculate the probability of the next word given 
   preceding words: P(w_n | w_n-1) = Count(w_n-1, w_n) / Count(w_n-1).
3. Applications: Auto-complete, spelling correction, machine translation, text 
   generation, and sentiment analysis.

Types Demonstrated:
- Word N-grams (Unigrams, Bigrams, Trigrams)
- Character N-grams (Sub-word modeling, language identification)
- N-gram Frequency Distribution & Conditional Probabilities
- N-gram Vectorization with scikit-learn
"""

import sys
import re
from collections import Counter

# ------------------------------------------------------------------------------
# 1. Dependency Setup
# ------------------------------------------------------------------------------
try:
    import nltk
    from nltk.util import ngrams as nltk_ngrams
    from nltk import FreqDist
except ImportError:
    nltk = None

try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    print("[Error] scikit-learn is not installed. Please run: pip install scikit-learn")
    sys.exit(1)


# ------------------------------------------------------------------------------
# 2. Implementation From Scratch (Pure Python)
# ------------------------------------------------------------------------------
def clean_and_tokenize(text: str) -> list[str]:
    """Cleans punctuation and returns lowercase tokens."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return cleaned.split()


def generate_ngrams_scratch(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generates n-grams using pure Python list slicing and zip."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generate_char_ngrams(word: str, n: int) -> list[str]:
    """Generates character-level n-grams for subword analysis."""
    padded = f"^{word}$"  # Start (^) and end ($) boundary markers
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def demonstrate_ngrams_from_scratch(text: str):
    """Demonstrates n-gram generation from first principles."""
    print("\n" + "=" * 70)
    print(" 1. N-GRAM GENERATION FROM SCRATCH (PURE PYTHON)")
    print("=" * 70)

    tokens = clean_and_tokenize(text)
    print(f"\n[A] Input Text Tokens ({len(tokens)} words):")
    print(" ", tokens)

    # Word Unigrams (n=1)
    unigrams = generate_ngrams_scratch(tokens, n=1)
    print(f"\n[B] Unigrams (n=1, Total: {len(unigrams)}):")
    print(" ", [" ".join(gram) for gram in unigrams])

    # Word Bigrams (n=2)
    bigrams = generate_ngrams_scratch(tokens, n=2)
    print(f"\n[C] Bigrams (n=2, Total: {len(bigrams)}):")
    print(" ", [" ".join(gram) for gram in bigrams])

    # Word Trigrams (n=3)
    trigrams = generate_ngrams_scratch(tokens, n=3)
    print(f"\n[D] Trigrams (n=3, Total: {len(trigrams)}):")
    print(" ", [" ".join(gram) for gram in trigrams])

    # Character N-grams (Sub-word level)
    sample_word = "language"
    char_trigrams = generate_char_ngrams(sample_word, n=3)
    print(f"\n[E] Character Trigrams for word '{sample_word}' (with ^ boundary $):")
    print(" ", char_trigrams)


# ------------------------------------------------------------------------------
# 3. Implementation Using NLTK (Frequency Distribution & Probability)
# ------------------------------------------------------------------------------
def demonstrate_nltk_ngrams(text: str):
    """Demonstrates NLTK n-grams, FreqDist, and Next-Word Probability."""
    print("\n" + "=" * 70)
    print(" 2. N-GRAM ANALYSIS USING NLTK")
    print("=" * 70)

    tokens = clean_and_tokenize(text)

    if nltk is not None:
        # Generate bigrams using NLTK
        bi_list = list(nltk_ngrams(tokens, 2))
        bi_freq = FreqDist(bi_list)
        uni_freq = FreqDist(tokens)

        print("\n[A] Top 5 Most Frequent Bigrams:")
        for bg, count in bi_freq.most_common(5):
            print(f"    {' '.join(bg):<30} -> Count: {count}")

        # Transition Probability P(w2 | w1) = Count(w1, w2) / Count(w1)
        print("\n[B] Bigram Conditional Transition Probabilities: P(w2 | w1)")
        evaluated_bigrams = list(set(bi_list))[:5]  # Sample first 5 distinct bigrams
        for w1, w2 in evaluated_bigrams:
            bigram_count = bi_freq[(w1, w2)]
            unigram_count = uni_freq[w1]
            prob = bigram_count / unigram_count if unigram_count > 0 else 0.0
            print(f"    P('{w2}' | '{w1}') = {bigram_count}/{unigram_count} = {prob:.4f}")
    else:
        # Fallback if NLTK is not available
        bi_list = generate_ngrams_scratch(tokens, 2)
        counts = Counter(bi_list)
        print("\n[A] Top 5 Most Frequent Bigrams:")
        for bg, count in counts.most_common(5):
            print(f"    {' '.join(bg):<30} -> Count: {count}")


# ------------------------------------------------------------------------------
# 4. Implementation Using scikit-learn (N-gram Vectorizer)
# ------------------------------------------------------------------------------
def demonstrate_sklearn_ngram_vectorizer(corpus: list[str]):
    """Demonstrates CountVectorizer with ngram_range for feature extraction."""
    print("\n" + "=" * 70)
    print(" 3. N-GRAM VECTORIZATION USING SCIKIT-LEARN")
    print("=" * 70)

    # 1. Bigrams-only vectorizer: ngram_range=(2, 2)
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    X_bigram = bigram_vectorizer.fit_transform(corpus)
    bigram_features = bigram_vectorizer.get_feature_names_out()

    print(f"\n[A] Bigram-Only Features (ngram_range=(2, 2), Total: {len(bigram_features)}):")
    print(" ", list(bigram_features))

    # Dense Matrix Representation
    print("\n[B] Bigram Feature Matrix:")
    header = f"{'Document':<10} | " + " | ".join(f"{bg[:10]:>10}" for bg in bigram_features)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(X_bigram.toarray(), start=1):
        row_str = f"Doc {i:<6} | " + " | ".join(f"{val:>10}" for val in row)
        print(row_str)
    print("-" * len(header))

    # 2. Combined Unigram + Bigram vectorizer: ngram_range=(1, 2)
    uni_bi_vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_uni_bi = uni_bi_vectorizer.fit_transform(corpus)
    uni_bi_features = uni_bi_vectorizer.get_feature_names_out()

    print(f"\n[C] Combined Unigram + Bigram Features (ngram_range=(1, 2), Total: {len(uni_bi_features)}):")
    print(" ", list(uni_bi_features))


# ------------------------------------------------------------------------------
# 5. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    # Continuous text for n-gram frequency and probability demonstration
    sample_text = (
        "Natural language processing enables computers to understand human language. "
        "Natural language understanding and natural language generation are two essential parts of natural language processing."
    )

    # Multi-document corpus for vectorization demonstration
    sample_corpus = [
        "Natural language processing is powerful.",
        "Language processing with neural networks.",
        "Natural language understanding and natural language generation."
    ]

    print("=" * 70)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 03: N-gram Modeling and Feature Extraction")
    print("=" * 70)

    print("\nInput Sample Text:")
    print(f"  \"{sample_text}\"")

    # Run demonstrations
    demonstrate_ngrams_from_scratch(sample_text)
    demonstrate_nltk_ngrams(sample_text)
    demonstrate_sklearn_ngram_vectorizer(sample_corpus)

    print("\n" + "=" * 70)
    print(" Execution Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
