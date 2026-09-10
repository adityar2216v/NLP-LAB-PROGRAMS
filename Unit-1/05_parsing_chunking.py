"""
================================================================================
Program 05: Parsing and Chunking using RegEx and spaCy
================================================================================

Core NLP Concepts Explained:
----------------------------
1. Part-of-Speech (POS) Tagging:
   - Assigns word-level grammatical categories (noun, verb, adjective, etc.)
   - Acts as the foundational prerequisite for shallow and deep parsing.

2. Chunking (Shallow Parsing):
   - Identifies non-overlapping, syntactically related groups of words 
     (e.g., Noun Phrases [NP], Verb Phrases [VP]) without building a full 
     hierarchical parse tree.
   - Example: In "The quick brown fox", the words together form a single NP chunk.

3. Parsing (Deep Syntactic Analysis):
   - Analyzes the complete, hierarchical grammatical structure of a sentence.
   - Constituency Parsing breaks sentences into nested clauses/phrases.
   - Dependency Parsing analyzes binary grammatical relationships between words 
     (head word vs. dependent modifier).

4. RegEx / Regexp Grammar in NLTK:
   - Regular expression patterns defined directly over POS tags rather than raw characters.
   - Example Grammar: `NP: {<DT>?<JJ>*<NN.*>+}`
     Meaning: An optional Determiner (<DT>?), followed by zero or more Adjectives (<JJ>*),
     followed by one or more Nouns (<NN.*>+).
"""

import sys

# ------------------------------------------------------------------------------
# 1. NLTK Setup
# ------------------------------------------------------------------------------
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk import pos_tag, RegexpParser
except ImportError:
    print("[Error] NLTK is not installed. Please run: pip install nltk")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. spaCy Setup
# ------------------------------------------------------------------------------
try:
    import spacy
except ImportError:
    print("[Error] spaCy is not installed. Please run: pip install spacy")
    sys.exit(1)


def demonstrate_nltk_chunking(sentence: str):
    """Demonstrates rule-based Noun Phrase chunking using NLTK RegexpParser."""
    print("\n" + "=" * 75)
    print(" 1. CHUNKING USING NLTK RegexpParser")
    print("=" * 75)

    # Step 1: Tokenize
    tokens = word_tokenize(sentence)

    # Step 2: POS Tagging
    tagged_tokens = pos_tag(tokens)
    print("\n[A] Step 1 & 2: Tokenization + POS Tagging:")
    print(" ", tagged_tokens)

    # Step 3: Define RegEx Grammar for Noun Phrases (NP)
    # Grammar specifies: an optional Determiner, zero or more Adjectives, and one or more Nouns
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}
    """
    print("\n[B] Step 3: Regexp Chunk Grammar:")
    print("  Pattern: NP: {<DT>?<JJ>*<NN.*>+}")
    print("  Meaning: Optional Determiner + Zero/More Adjectives + One/More Nouns")

    # Step 4: Parse tokens into chunk tree
    chunk_parser = RegexpParser(grammar)
    chunk_tree = chunk_parser.parse(tagged_tokens)

    # Step 5: Display resulting parse tree in text format (No GUI needed)
    print("\n[C] Step 4: Text-Based Parse Tree:")
    print("-" * 75)
    print(chunk_tree.pformat())
    print("-" * 75)

    # Step 6: Extract and display just the identified NP chunks
    print("\n[D] Extracted Noun Phrase (NP) Chunks:")
    chunk_count = 0
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
        chunk_count += 1
        phrase = " ".join(word for word, tag in subtree.leaves())
        print(f"  {chunk_count}. [NP: '{phrase}'] with tags {subtree.leaves()}")


def demonstrate_spacy_parsing_and_chunking(sentence: str):
    """Demonstrates noun chunking and dependency parsing using spaCy."""
    print("\n" + "=" * 75)
    print(" 2. CHUNKING & PARSING USING spaCy")
    print("=" * 75)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[Warning] spaCy model 'en_core_web_sm' is not installed.")
        print("Install via: python -m spacy download en_core_web_sm")
        return

    doc = nlp(sentence)

    # [A] Noun Chunks extraction via spaCy
    print("\n[A] spaCy Noun Chunks (doc.noun_chunks):")
    print("-" * 75)
    print(f"{'Chunk Phrase':<30} | {'Root Word':<15} | {'Root Dependency':<18}")
    print("-" * 75)
    for chunk in doc.noun_chunks:
        print(f"{chunk.text:<30} | {chunk.root.text:<15} | {chunk.root.dep_:<18}")
    print("-" * 75)

    # [B] Basic Dependency Parsing analysis
    print("\n[B] spaCy Dependency Parsing (Token -> Relation -> Head):")
    print("-" * 75)
    print(f"{'Token':<15} | {'POS Tag':<10} | {'Dependency Relation':<22} | {'Head Token'}")
    print("-" * 75)
    for token in doc:
        print(f"{token.text:<15} | {token.pos_:<10} | {token.dep_:<22} | {token.head.text}")
    print("-" * 75)


def main():
    # Sample sentence clearly containing multiple noun phrases
    sample_sentence = "The smart student is reading a comprehensive book in the central library."

    print("=" * 75)
    print("EXPERIMENT 05: PARSING AND CHUNKING")
    print("=" * 75)
    print("\nInput Sentence:")
    print("-" * 75)
    print(sample_sentence)
    print("-" * 75)

    # 1. NLTK Regexp Chunking
    demonstrate_nltk_chunking(sample_sentence)

    # 2. spaCy Chunking and Dependency Parsing
    demonstrate_spacy_parsing_and_chunking(sample_sentence)

    print("\n" + "=" * 75)
    print("Parsing and Chunking demonstration completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
