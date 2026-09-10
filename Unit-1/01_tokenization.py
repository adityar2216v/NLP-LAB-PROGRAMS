"""
================================================================================
Program 01: Tokenization of Sentences and Words using NLTK and spaCy
================================================================================

What is Tokenization?
--------------------
Tokenization is the fundamental NLP preprocessing step of splitting continuous 
text into smaller, meaningful units called "tokens".
- Sentence Tokenization: Segments a text block into individual sentences.
- Word Tokenization: Breaks sentences into individual words, punctuation, 
  and numerical symbols.

Libraries Used:
- NLTK (Natural Language Toolkit): Classical rule- and unsupervised-model-based tokenizers.
- spaCy: High-performance, pipeline-based industrial NLP library.
"""

import sys

# ------------------------------------------------------------------------------
# 1. NLTK Tokenization Setup and Execution
# ------------------------------------------------------------------------------
try:
    import nltk
    # Automatically download required tokenization models if not already present
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    print("[Error] NLTK library is not installed. Please run: pip install nltk")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. spaCy Tokenization Setup
# ------------------------------------------------------------------------------
try:
    import spacy
except ImportError:
    print("[Error] spaCy library is not installed. Please run: pip install spacy")
    sys.exit(1)


def perform_nltk_tokenization(text: str):
    """Demonstrates sentence and word tokenization using NLTK."""
    print("\n" + "=" * 60)
    print(" 1. TOKENIZATION USING NLTK")
    print("=" * 60)

    # Sentence Tokenization using NLTK
    sentences = sent_tokenize(text)
    print(f"\n[A] Sentence Tokenization (Total sentences: {len(sentences)}):")
    for i, sentence in enumerate(sentences, start=1):
        print(f"  {i}. {sentence}")

    # Word Tokenization using NLTK
    words = word_tokenize(text)
    print(f"\n[B] Word Tokenization (Total tokens: {len(words)}):")
    print(" ", words)


def perform_spacy_tokenization(text: str):
    """Demonstrates sentence and word tokenization using spaCy."""
    print("\n" + "=" * 60)
    print(" 2. TOKENIZATION USING spaCy")
    print("=" * 60)

    try:
        # Load small English pipeline
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("\n[Warning] spaCy model 'en_core_web_sm' is not installed.")
        print("Please download it by executing:")
        print("    python -m spacy download en_core_web_sm")
        return

    # Process text through spaCy pipeline
    doc = nlp(text)

    # Sentence Tokenization using spaCy
    spacy_sentences = [sent.text.strip() for sent in doc.sents]
    print(f"\n[A] Sentence Tokenization (Total sentences: {len(spacy_sentences)}):")
    for i, sentence in enumerate(spacy_sentences, start=1):
        print(f"  {i}. {sentence}")

    # Word Tokenization using spaCy
    spacy_words = [token.text for token in doc]
    print(f"\n[B] Word Tokenization (Total tokens: {len(spacy_words)}):")
    print(" ", spacy_words)


def main():
    # Sample input paragraph for demonstration
    sample_text = (
        "Natural Language Processing (NLP) is a branch of artificial intelligence. "
        "It helps computers understand, interpret, and manipulate human language! "
        "Today, NLP powers many applications such as chatbots, voice assistants, and search engines."
    )

    print("=" * 70)
    print("EXPERIMENT 01: SENTENCE AND WORD TOKENIZATION")
    print("=" * 70)
    print("\nOriginal Input Text:")
    print("-" * 70)
    print(sample_text)
    print("-" * 70)

    # 1. Run NLTK Tokenization
    perform_nltk_tokenization(sample_text)

    # 2. Run spaCy Tokenization
    perform_spacy_tokenization(sample_text)

    print("\n" + "=" * 70)
    print("Tokenization completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
