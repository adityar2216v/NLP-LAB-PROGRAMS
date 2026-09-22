"""
================================================================================
Program 13: Information Extraction (IE) from Structured/Unstructured Documents
================================================================================

What is Information Extraction (IE)?
------------------------------------
Information Extraction (IE) is the automated task of extracting structured 
information (entities, relations, attributes, and events) from unstructured 
or semi-structured natural language text documents.

Core Stages in an Information Extraction Pipeline:
--------------------------------------------------
1. Tokenization & Sentence Segmentation: Breaks raw text into manageable units.
2. Attribute & Pattern Extraction: Detects structured tokens using regular 
   expression automata (Email addresses, Phone numbers, Currency, Dates, URLs).
3. Named Entity Recognition (NER): Identifies domain entities:
   - PERSON: Individuals, founders, executives.
   - ORGANIZATION: Companies, institutions, agencies.
   - LOCATION / GPE: Cities, states, countries.
4. Relation Extraction (RE): Discovers semantic predicates connecting entity pairs:
   - (Subject, Predicate, Object) triples.
   - e.g., ("Sundar Pichai", "CEO_OF", "Alphabet Inc.")
   - e.g., ("Alphabet Inc.", "HEADQUARTERED_IN", "Mountain View, California")
5. Knowledge Synthesis: Transforms extracted facts into structured database 
   records and machine-readable JSON schemas.

Libraries Used:
- re (Standard Library): High-performance regular expression pattern matching.
- NLTK: Sentence splitting and POS tagging for grammatical dependency cues.
- JSON: Structured data export.
"""

import sys
import re
import json
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


# ------------------------------------------------------------------------------
# 1. Regular Expression Attribute Extractors
# ------------------------------------------------------------------------------
REGEX_PATTERNS = {
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "Phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "Money": r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|USD))?",
    "Date": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}\b",
    "URL": r"https?://(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[A-Za-z0-9._%+-]*)*"
}


def extract_regex_attributes(text: str) -> dict[str, list[str]]:
    """Extracts structured values (emails, phones, currencies, dates) using RegEx."""
    attributes = {}
    for attr_name, pattern in REGEX_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            attributes[attr_name] = sorted(list(set(matches)))
    return attributes


# ------------------------------------------------------------------------------
# 2. Rule-Based Named Entity Extraction
# ------------------------------------------------------------------------------
def extract_named_entities(text: str) -> dict[str, list[str]]:
    """
    Extracts key named entities (Persons, Organizations, Locations) based on 
    capitalization patterns, titles, and lexical context.
    """
    entities = defaultdict(set)

    # Common corporate designators
    corp_suffixes = r"(?:Inc\.|Corp\.|Corporation|LLC|Ltd\.|Technologies|Group)"
    org_pattern = rf"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\s+{corp_suffixes}\b"
    for match in re.finditer(org_pattern, text):
        entities["ORGANIZATION"].add(match.group())

    # Well-known tech giants without suffixes
    tech_giants = ["Google", "Microsoft", "Apple", "Amazon", "Alphabet", "Meta", "Tesla", "Nvidia"]
    for tg in tech_giants:
        if re.search(rf"\b{tg}\b", text):
            entities["ORGANIZATION"].add(tg)

    # Person pattern with titles: (Dr.|Mr.|Ms.|CEO|Founder) [First] [Last]
    person_pattern = r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.|CEO\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\b"
    # Filter common non-person words
    non_persons = {"The", "In", "On", "At", "Oncology", "January", "February", "March", 
                   "April", "May", "June", "July", "August", "September", "October", 
                   "November", "December", "United", "Silicon", "Mountain", "California"}
    
    # Context-assisted person extraction: words preceded by "CEO", "founded by", "appointed"
    exec_pattern = r"(?:CEO|Chief Executive Officer|founded by|appointed|led by)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"
    for match in re.finditer(exec_pattern, text):
        entities["PERSON"].add(match.group(1))

    # Location pattern (City, State / Country)
    loc_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z][a-z]+)\b"
    for match in re.finditer(loc_pattern, text):
        entities["LOCATION"].add(f"{match.group(1)}, {match.group(2)}")

    return {k: sorted(list(v)) for k, v in entities.items()}


# ------------------------------------------------------------------------------
# 3. Semantic Relation Extraction (Subject - Predicate - Object Triples)
# ------------------------------------------------------------------------------
def extract_semantic_relations(text: str) -> list[dict]:
    """
    Extracts factual relational triples: (Subject Entity, Relation, Object Entity).
    """
    relations = []
    sentences = sent_tokenize(text)

    # Relation extraction patterns
    relation_rules = [
        # CEO_OF relation
        (
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+serves as (?:the )?CEO of\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)",
            "CEO_OF"
        ),
        (
            r"([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\s+appointed\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+as\s+(?:the\s+)?CEO",
            "APPOINTED_CEO"
        ),
        # HEADQUARTERED_IN relation
        (
            r"([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\s+is headquartered in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s+[A-Z][a-z]+)?)",
            "HEADQUARTERED_IN"
        ),
        # ACQUIRED relation
        (
            r"([A-Z][a-zA-Z0-9]+)\s+acquired\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\s+for\s+(\$[\d.]+\s*(?:billion|million)?)",
            "ACQUIRED_FOR"
        ),
        # FOUNDED_BY relation
        (
            r"([A-Z][a-zA-Z0-9]+)\s+was founded by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            "FOUNDED_BY"
        )
    ]

    for sent in sentences:
        for pattern, rel_type in relation_rules:
            for match in re.finditer(pattern, sent):
                groups = match.groups()
                if rel_type == "APPOINTED_CEO":
                    # Swap order so subject is Person, object is Company
                    relations.append({
                        "subject": groups[1],
                        "relation": "CEO_OF",
                        "object": groups[0],
                        "evidence": sent.strip()
                    })
                elif rel_type == "ACQUIRED_FOR":
                    relations.append({
                        "subject": groups[0],
                        "relation": "ACQUIRED",
                        "object": groups[1],
                        "amount": groups[2],
                        "evidence": sent.strip()
                    })
                else:
                    relations.append({
                        "subject": groups[0],
                        "relation": rel_type,
                        "object": groups[1],
                        "evidence": sent.strip()
                    })

    return relations


# ------------------------------------------------------------------------------
# 4. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 13: Information Extraction (IE) from Documents")
    print("=" * 80)

    sample_document = """
    On September 15, 2024, Alphabet Inc. announced a strategic leadership expansion.
    Sundar Pichai serves as CEO of Alphabet Inc., which is headquartered in Mountain View, California.
    The enterprise can be contacted at press-relations@google.com or via telephone at (650) 253-0000.
    Official announcements and annual SEC filings are published online at https://abc.xyz/investor.
    In recent financial transactions, Alphabet acquired Mandiant for $5.4 billion to strengthen cloud cybersecurity.
    Additionally, Microsoft appointed Satya Nadella as CEO to direct artificial intelligence initiatives.
    Microsoft is headquartered in Redmond, Washington.
    """

    print("\nInput Unstructured Document:")
    print("-" * 70)
    print(sample_document.strip())
    print("-" * 70)

    # 1. Regex Structured Attributes
    print("\n" + "=" * 80)
    print(" 1. STRUCTURED ATTRIBUTE EXTRACTION (REGEX PATTERNS)")
    print("=" * 80)
    attributes = extract_regex_attributes(sample_document)
    for attr, values in attributes.items():
        print(f"  {attr:<12} : {', '.join(values)}")

    # 2. Named Entity Extraction
    print("\n" + "=" * 80)
    print(" 2. NAMED ENTITY RECOGNITION (NER)")
    print("=" * 80)
    entities = extract_named_entities(sample_document)
    for category, items in entities.items():
        print(f"  {category:<14} : {', '.join(items)}")

    # 3. Relation Extraction
    print("\n" + "=" * 80)
    print(" 3. RELATION EXTRACTION (SUBJECT - PREDICATE - OBJECT TRIPLES)")
    print("=" * 80)
    relations = extract_semantic_relations(sample_document)

    header = f"{'Subject Entity':<18} | {'Relation / Predicate':<20} | {'Object Entity'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in relations:
        amt = f" (Value: {r['amount']})" if "amount" in r else ""
        print(f"{r['subject']:<18} | {r['relation']:<20} | {r['object']}{amt}")
    print("-" * len(header))

    # 4. Structured JSON Knowledge Export
    print("\n" + "=" * 80)
    print(" 4. SYNTHESIZED STRUCTURED KNOWLEDGE SCHEMA (JSON OUTPUT)")
    print("=" * 80)
    knowledge_base = {
        "document_metadata": {
            "dates_mentioned": attributes.get("Date", []),
            "emails_extracted": attributes.get("Email", []),
            "phone_numbers": attributes.get("Phone", []),
            "urls_extracted": attributes.get("URL", [])
        },
        "entities": entities,
        "relations": relations
    }
    print(json.dumps(knowledge_base, indent=2))

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
