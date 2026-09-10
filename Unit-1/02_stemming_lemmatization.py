"""
================================================================================
Program 02: Stemming and Lemmatization on Sample Text
================================================================================

Concepts Explained:
-------------------
1. Stemming:
   - A crude, heuristic rule-based process that chops off common prefixes 
     and suffixes from words (e.g., "-ing", "-ed", "-s").
   - It is fast and computationally inexpensive, but may produce non-words 
     or invalid dictionary forms (e.g., "studies" -> "studi").
   - Algorithm used here: Porter Stemmer (NLTK).

2. Lemmatization:
   - A linguistic approach that uses a complete vocabulary and morphological 
     analysis (e.g., WordNet database) to remove inflectional endings.
   - It always returns a valid, meaningful dictionary root known as a "lemma" 
     (e.g., "studies" -> "study", "children" -> "child").
   - Lemmatizers perform best when provided with the word's Part-of-Speech (POS),
     such as 'v' for verb or 'n' for noun.

Key Difference:
--------------
- Stemming relies on pattern-matching rules and may output non-words.
- Lemmatization relies on grammatical context and vocabulary lookup, 
  always outputting valid base words.
"""

import sys

try:
    import nltk
    # Download WordNet corpora required by WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    print("[Error] NLTK is not installed. Please run: pip install nltk")
    sys.exit(1)


def demonstrate_stemming_and_lemmatization():
    # Initialize the Porter Stemmer and WordNet Lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Prescribed sample words from lab curriculum
    sample_words = [
        "playing",
        "played",
        "plays",
        "studies",
        "studying",
        "running",
        "easily",
        "children"
    ]

    print("=" * 80)
    print("EXPERIMENT 02: STEMMING AND LEMMATIZATION COMPARISON")
    print("=" * 80)

    print(f"\n{'Original Word':<15} | {'Porter Stemmer':<18} | {'Lemmatizer (Noun)':<20} | {'Lemmatizer (Verb)':<18}")
    print("-" * 80)

    for word in sample_words:
        stemmed_word = stemmer.stem(word)
        # Default lemmatization treats words as nouns (pos='n')
        lemma_noun = lemmatizer.lemmatize(word, pos='n')
        # Providing pos='v' allows verbs to be correctly reduced to root verb form
        lemma_verb = lemmatizer.lemmatize(word, pos='v')

        print(f"{word:<15} | {stemmed_word:<18} | {lemma_noun:<20} | {lemma_verb:<18}")

    print("-" * 80)

    print("\nObservations for College Lab Viva:")
    print("1. 'studies':")
    print("   - Stemmer cuts it to 'studi' (not a real English word).")
    print("   - Lemmatizer (Noun) keeps 'studies', but Lemmatizer (Verb) yields 'study' (valid word).")
    print("2. 'children':")
    print("   - Stemmer produces 'children' (unable to identify irregular plural).")
    print("   - Lemmatizer yields 'child' (accurately identifies irregular lemma).")
    print("3. 'easily':")
    print("   - Stemmer strips suffix to 'easili'.")
    print("   - Lemmatizer preserves 'easily' as a valid adverb.")


def main():
    demonstrate_stemming_and_lemmatization()
    print("\n" + "=" * 80)
    print("Stemming and Lemmatization completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
