"""
================================================================================
Program 06: Named Entity Recognition (NER) using spaCy
================================================================================

What is Named Entity Recognition (NER)?
---------------------------------------
Named Entity Recognition (NER) is a vital Natural Language Processing task 
that automatically locates and classifies real-world named entities mentioned 
in unstructured text into predefined categories.

Key Entity Types Recognized:
----------------------------
- PERSON : Names of people (e.g., 'Sundar Pichai', 'Satya Nadella')
- ORG    : Companies, agencies, institutions (e.g., 'Google', 'Microsoft')
- GPE    : Geopolitical entities (countries, cities, states; e.g., 'California', 'India')
- DATE   : Absolute or relative dates or periods (e.g., 'September 15, 2024')
- MONEY  : Monetary values including currency units (e.g., '$10 billion')

Why is NER Useful?
------------------
1. Information Extraction: Quickly pulls structured knowledge from documents.
2. Search & Indexing: Enhances search engines by indexing key entities.
3. Knowledge Graphs: Builds node-relationship graphs from raw web data.
4. Content Categorization & News Aggregation: Tags articles by companies and leaders.

Required spaCy Model:
---------------------
This program utilizes the 'en_core_web_sm' English pipeline.
If not installed, install it using:
    python -m spacy download en_core_web_sm
"""

import sys

try:
    import spacy
except ImportError:
    print("[Error] spaCy library is not installed.")
    print("Please install it using: pip install spacy")
    sys.exit(1)


def perform_named_entity_recognition(document: str):
    """Loads spaCy model and extracts named entities with categories and descriptions."""
    # Attempt to load the pre-trained English model with a friendly fallback
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print("\n" + "!" * 75)
        print(f"[ERROR] spaCy model '{model_name}' was not found!")
        print("To download and install the required model, run:")
        print(f"    python -m spacy download {model_name}")
        print("!" * 75)
        sys.exit(1)

    print("=" * 80)
    print("EXPERIMENT 06: NAMED ENTITY RECOGNITION (NER) USING SPACY")
    print("=" * 80)

    print("\n[A] Original Input Document:")
    print("-" * 80)
    print(document)
    print("-" * 80)

    # Process text through the spaCy NLP pipeline
    doc = nlp(document)

    print("\n[B] Detected Named Entities:")
    print("-" * 80)
    print(f"{'Entity Text':<25} | {'Label':<12} | {'Label Explanation'}")
    print("-" * 80)

    # Dictionary to group entities by label for clean summary display
    grouped_entities = {}

    for ent in doc.ents:
        # spacy.explain provides an official description of the entity tag
        explanation = spacy.explain(ent.label_) or "Named entity"
        print(f"{ent.text:<25} | {ent.label_:<12} | {explanation}")

        grouped_entities.setdefault(ent.label_, []).append(ent.text)

    print("-" * 80)

    # Grouped Summary
    print("\n[C] Summary of Entities Grouped by Category:")
    print("-" * 80)
    for label, items in grouped_entities.items():
        print(f"  * {label:<10} ({spacy.explain(label)}): {', '.join(items)}")

    print(f"\nTotal Named Entities Identified: {len(doc.ents)}")


def main():
    # Sample document covering PERSON, ORG, GPE, DATE, and MONEY entities
    sample_document = (
        "Sundar Pichai, the CEO of Google, announced in California on "
        "September 15, 2024, that the company will invest $10 billion "
        "in India to boost artificial intelligence and cloud infrastructure."
    )

    perform_named_entity_recognition(sample_document)

    print("\n" + "=" * 80)
    print("Named Entity Recognition completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
