"""
================================================================================
Program 03: Stop-word Removal from a Document
================================================================================

What are Stopwords?
-------------------
Stopwords are frequently occurring words in any natural language (e.g., 'is', 
'the', 'in', 'at', 'which', 'and', 'to') that carry minimal semantic meaning 
or topical information on their own.

Why Remove Stopwords?
---------------------
1. Dimensionality Reduction: Reduces the vocabulary size and memory footprint.
2. Noise Elimination: Helps algorithms (such as text classification, sentiment 
   analysis, and topic modeling) focus on content-bearing keywords.
3. Computational Efficiency: Speeds up model training and text processing pipelines.

Library Used:
-------------
- NLTK: provides a curated list of standard stopwords for multiple languages.
"""

import sys

try:
    import nltk
    # Download required NLTK datasets for tokenization and stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    print("[Error] NLTK is not installed. Please run: pip install nltk")
    sys.exit(1)


def remove_stopwords_from_text(document: str):
    """Demonstrates tokenization, stopword identification, and removal."""
    # 1. Load standard English stopwords
    stop_words = set(stopwords.words('english'))

    # 2. Tokenize the input document into words
    tokens = word_tokenize(document)

    # 3. Filter tokens and identify which were removed
    filtered_tokens = []
    removed_words = []

    for token in tokens:
        # Check if the lowercased token is in the stopwords set
        if token.lower() in stop_words:
            removed_words.append(token)
        else:
            filtered_tokens.append(token)

    # 4. Display results clearly
    print("=" * 80)
    print("EXPERIMENT 03: STOP-WORD REMOVAL")
    print("=" * 80)

    print("\n[A] Original Document:")
    print("-" * 80)
    print(document)

    print(f"\n[B] Total Tokens Before Removal ({len(tokens)} tokens):")
    print("-" * 80)
    print(tokens)

    print(f"\n[C] Identified Stopwords Removed ({len(removed_words)} occurrences):")
    print("-" * 80)
    print(removed_words)
    print("Unique stopwords removed:", sorted(list(set(w.lower() for w in removed_words))))

    print(f"\n[D] Filtered Tokens After Removal ({len(filtered_tokens)} tokens):")
    print("-" * 80)
    print(filtered_tokens)

    # Reconstructing the filtered text for demonstration
    reconstructed_text = " ".join(filtered_tokens)
    print("\n[E] Reconstructed Cleaned Text:")
    print("-" * 80)
    print(reconstructed_text)

    # Statistical Summary
    reduction_pct = ((len(tokens) - len(filtered_tokens)) / len(tokens)) * 100
    print("\n[F] Summary Statistics:")
    print("-" * 80)
    print(f"Total Original Tokens   : {len(tokens)}")
    print(f"Total Filtered Tokens   : {len(filtered_tokens)}")
    print(f"Stopwords Filtered Out  : {len(removed_words)}")
    print(f"Reduction Percentage    : {reduction_pct:.2f}%")


def main():
    # Sample document representative of NLP/AI domain
    sample_document = (
        "Natural Language Processing is an exciting subfield of Artificial Intelligence. "
        "It enables computers to read, understand, and derive meaning from human languages "
        "in a manner that is both valuable and efficient for the users."
    )

    remove_stopwords_from_text(sample_document)

    print("\n" + "=" * 80)
    print("Stop-word removal completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
