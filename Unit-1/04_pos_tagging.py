"""
================================================================================
Program 04: Part-of-Speech (POS) Tagging of a Given Sentence
================================================================================

What is Part-of-Speech (POS) Tagging?
-------------------------------------
POS tagging (also known as grammatical tagging or word-category disambiguation) 
is the process of labeling each word in a corpus with its corresponding 
grammatical category (such as noun, verb, adjective, adverb, pronoun, etc.) 
based on both its definition and its syntactic context within the sentence.

Common Penn Treebank POS Tags Reference:
----------------------------------------
- DT  : Determiner (e.g., 'the', 'a', 'these')
- JJ  : Adjective (e.g., 'quick', 'brown', 'lazy')
- NN  : Noun, singular or mass (e.g., 'fox', 'dog')
- NNS : Noun, plural (e.g., 'foxes', 'dogs')
- NNP : Proper noun, singular (e.g., 'India', 'Google')
- VB  : Verb, base form (e.g., 'jump', 'eat')
- VBZ : Verb, 3rd person singular present (e.g., 'jumps', 'reads')
- VBD : Verb, past tense (e.g., 'jumped', 'saw')
- VBG : Verb, gerund or present participle (e.g., 'jumping', 'learning')
- RB  : Adverb (e.g., 'quickly', 'easily')
- IN  : Preposition or subordinating conjunction (e.g., 'over', 'in', 'of')
- PRP : Personal pronoun (e.g., 'he', 'she', 'it', 'they')
"""

import sys

try:
    import nltk
    # Download tokenizer and modern POS tagger models for NLTK
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
except ImportError:
    print("[Error] NLTK is not installed. Please run: pip install nltk")
    sys.exit(1)


# Dictionary providing human-readable explanations for common Penn Treebank POS tags
POS_TAG_EXPLANATIONS = {
    "DT": "Determiner",
    "JJ": "Adjective (quality/attribute)",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper Noun, singular",
    "NNPS": "Proper Noun, plural",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund / present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "IN": "Preposition / subordinating conjunction",
    "PRP": "Personal Pronoun",
    "PRP$": "Possessive Pronoun",
    "CC": "Coordinating Conjunction",
    "CD": "Cardinal Number",
    "MD": "Modal auxiliary",
    ".": "Punctuation mark"
}


def perform_pos_tagging(sentence: str):
    """Tokenizes and performs POS tagging on the input sentence."""
    print("=" * 80)
    print("EXPERIMENT 04: PART-OF-SPEECH (POS) TAGGING")
    print("=" * 80)

    print("\nInput Sentence:")
    print("-" * 80)
    print(sentence)
    print("-" * 80)

    # 1. Tokenize sentence into words
    tokens = word_tokenize(sentence)

    # 2. Assign POS tags using NLTK's averaged perceptron tagger
    tagged_tokens = pos_tag(tokens)

    # 3. Display tagged words in structured tabular format
    print(f"\n{'Token':<18} | {'POS Tag':<10} | {'Grammatical Meaning'}")
    print("-" * 80)
    for word, tag in tagged_tokens:
        meaning = POS_TAG_EXPLANATIONS.get(tag, "Other grammatical symbol")
        print(f"{word:<18} | {tag:<10} | {meaning}")
    print("-" * 80)

    # 4. Compact tuple format commonly shown in textbooks
    print("\nNLTK POS Tagged Tuple Output:")
    print(tagged_tokens)


def main():
    # Prescribed classic pangram containing varied parts of speech
    sample_sentence = "The quick brown fox jumps over the lazy dog."

    perform_pos_tagging(sample_sentence)

    print("\n" + "=" * 80)
    print("POS Tagging completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
