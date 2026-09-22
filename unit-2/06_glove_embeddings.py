"""
================================================================================
Program 06: GloVe Embeddings Loading and Vector Representation
================================================================================

What is GloVe (Global Vectors for Word Representation)?
------------------------------------------------------
GloVe (Pennington, Socher, and Manning, Stanford, 2014) is an unsupervised learning 
algorithm for obtaining vector representations for words. It maps words into a 
meaningful vector space by combining the advantages of:
1. Global Matrix Factorization (like LSA): Utilizes whole-corpus global statistics.
2. Local Context Window Methods (like Word2Vec): Captures subtle semantic patterns.

Core GloVe Intuition:
---------------------
The ratio of co-occurrence probabilities of two words with a set of probe probe words 
encodes semantic relationships. GloVe optimizes the objective:

    w_i^T * w_j + b_i + b_j = log(X_ij)

Where X_ij is the number of times word i and word j appear within a context window.

GloVe File Format:
------------------
Standard GloVe files (e.g., `glove.6B.50d.txt`) are plain text files where each line 
contains:
    <word> <val_1> <val_2> <val_3> ... <val_d>

Key Concepts Demonstrated:
1. Parsing and loading GloVe vector files into an in-memory dictionary / index.
2. Looking up word embeddings and analyzing vector properties.
3. Handling Out-of-Vocabulary (OOV) terms.
4. Document representation via Average Word Embeddings (Centroid Vectors):
   v_doc = (1 / |words|) * ∑ v_word
5. Document-level semantic similarity using GloVe centroid vectors.

Libraries Used:
- NumPy / Pure Python: For vector parsing, centroid pooling, and cosine similarity.
- sklearn.metrics.pairwise: For batch cosine similarity matrix computation.
"""

import sys
import os
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------------------
# 1. Self-Contained Realistic GloVe Corpus (50-Dimensional Vectors)
#    Emulates official Stanford GloVe format without requiring a 2GB download
# ------------------------------------------------------------------------------
SAMPLE_GLOVE_DATA = """the 0.418 0.249 -0.412 0.121 0.345 -0.043 -0.477 0.056 0.112 -0.145 0.033 0.211 -0.198 0.045 -0.123 0.089 0.156 -0.034 -0.112 0.067 0.045 -0.012 0.098 -0.045 0.112 -0.089 0.034 -0.067 0.145 -0.023 0.056 -0.089 0.123 -0.045 0.078 -0.012 0.067 -0.034 0.089 -0.045 0.112 -0.056 0.034 -0.078 0.098 -0.023 0.045 -0.067 0.089 -0.012
a 0.334 0.187 -0.321 0.098 0.289 -0.032 -0.398 0.043 0.089 -0.112 0.021 0.178 -0.156 0.034 -0.098 0.076 0.134 -0.023 -0.089 0.056 0.034 -0.009 0.078 -0.034 0.089 -0.076 0.023 -0.056 0.123 -0.019 0.045 -0.076 0.098 -0.034 0.067 -0.009 0.056 -0.023 0.078 -0.034 0.089 -0.045 0.023 -0.067 0.078 -0.019 0.034 -0.056 0.078 -0.009
king 0.504 0.686 -0.702 -0.246 -0.017 0.234 0.298 -0.123 -0.345 0.678 0.123 -0.456 0.234 0.567 -0.123 0.345 -0.098 0.213 -0.432 0.123 0.456 -0.234 0.123 -0.345 0.567 -0.123 0.234 -0.456 0.123 0.345 -0.234 0.123 -0.456 0.234 0.345 -0.123 0.234 -0.345 0.456 -0.123 0.234 -0.345 0.123 0.234 -0.345 0.123 0.456 -0.234 0.123 -0.345
queen 0.489 0.702 -0.689 -0.212 -0.034 0.245 0.312 -0.109 -0.321 0.698 0.145 -0.432 0.245 0.589 -0.109 0.367 -0.087 0.234 -0.412 0.145 0.478 -0.212 0.145 -0.321 0.589 -0.109 0.245 -0.432 0.145 0.367 -0.212 0.145 -0.432 0.245 0.367 -0.109 0.245 -0.321 0.478 -0.109 0.245 -0.321 0.145 0.245 -0.321 0.145 0.478 -0.212 0.145 -0.321
man 0.321 0.456 -0.567 -0.123 0.089 0.145 0.213 -0.089 -0.234 0.456 0.089 -0.321 0.145 0.389 -0.089 0.234 -0.067 0.145 -0.312 0.089 0.321 -0.156 0.089 -0.234 0.389 -0.089 0.145 -0.321 0.089 0.234 -0.156 0.089 -0.321 0.145 0.234 -0.089 0.145 -0.234 0.321 -0.089 0.145 -0.234 0.089 0.145 -0.234 0.089 0.321 -0.156 0.089 -0.234
woman 0.312 0.478 -0.543 -0.098 0.076 0.156 0.234 -0.076 -0.212 0.478 0.098 -0.298 0.156 0.412 -0.076 0.256 -0.054 0.167 -0.289 0.098 0.345 -0.134 0.098 -0.212 0.412 -0.076 0.156 -0.298 0.098 0.256 -0.134 0.098 -0.298 0.156 0.256 -0.076 0.156 -0.212 0.345 -0.076 0.156 -0.212 0.098 0.156 -0.212 0.098 0.345 -0.134 0.098 -0.212
computer 0.123 -0.234 0.456 0.678 -0.345 0.567 -0.123 0.456 -0.089 0.123 0.345 0.567 -0.234 0.123 0.456 -0.345 0.234 0.567 -0.123 0.456 0.123 -0.234 0.456 0.678 -0.345 0.567 -0.123 0.456 -0.089 0.123 0.345 0.567 -0.234 0.123 0.456 -0.345 0.234 0.567 -0.123 0.456 0.123 -0.234 0.456 0.678 -0.345 0.567 -0.123 0.456 -0.089 0.123
software 0.145 -0.212 0.478 0.698 -0.321 0.589 -0.098 0.478 -0.067 0.145 0.367 0.589 -0.212 0.145 0.478 -0.321 0.256 0.589 -0.098 0.478 0.145 -0.212 0.478 0.698 -0.321 0.589 -0.098 0.478 -0.067 0.145 0.367 0.589 -0.212 0.145 0.478 -0.321 0.256 0.589 -0.098 0.478 0.145 -0.212 0.478 0.698 -0.321 0.589 -0.098 0.478 -0.067 0.145
programming 0.109 -0.256 0.432 0.654 -0.367 0.543 -0.145 0.432 -0.109 0.109 0.321 0.543 -0.256 0.109 0.432 -0.367 0.212 0.543 -0.145 0.432 0.109 -0.256 0.432 0.654 -0.367 0.543 -0.145 0.432 -0.109 0.109 0.321 0.543 -0.256 0.109 0.432 -0.367 0.212 0.543 -0.145 0.432 0.109 -0.256 0.432 0.654 -0.367 0.543 -0.145 0.432 -0.109 0.109
python 0.134 -0.198 0.467 0.689 -0.312 0.578 -0.089 0.467 -0.056 0.134 0.356 0.578 -0.198 0.134 0.467 -0.312 0.245 0.578 -0.089 0.467 0.134 -0.198 0.467 0.689 -0.312 0.578 -0.089 0.467 -0.056 0.134 0.356 0.578 -0.198 0.134 0.467 -0.312 0.245 0.578 -0.089 0.467 0.134 -0.198 0.467 0.689 -0.312 0.578 -0.089 0.467 -0.056 0.134
natural -0.089 0.145 0.234 0.123 -0.456 0.123 0.345 0.234 -0.123 0.089 0.156 0.234 -0.123 0.156 0.234 -0.145 0.089 0.234 0.123 0.234 -0.089 0.145 0.234 0.123 -0.456 0.123 0.345 0.234 -0.123 0.089 0.156 0.234 -0.123 0.156 0.234 -0.145 0.089 0.234 0.123 0.234 -0.089 0.145 0.234 0.123 -0.456 0.123 0.345 0.234 -0.123 0.089
language -0.076 0.156 0.256 0.145 -0.432 0.145 0.367 0.256 -0.098 0.109 0.178 0.256 -0.098 0.178 0.256 -0.123 0.109 0.256 0.145 0.256 -0.076 0.156 0.256 0.145 -0.432 0.145 0.367 0.256 -0.098 0.109 0.178 0.256 -0.098 0.178 0.256 -0.123 0.109 0.256 0.145 0.256 -0.076 0.156 0.256 0.145 -0.432 0.145 0.367 0.256 -0.098 0.109
processing -0.065 0.167 0.278 0.167 -0.412 0.167 0.389 0.278 -0.076 0.123 0.198 0.278 -0.076 0.198 0.278 -0.109 0.123 0.278 0.167 0.278 -0.065 0.167 0.278 0.167 -0.412 0.167 0.389 0.278 -0.076 0.123 0.198 0.278 -0.076 0.198 0.278 -0.109 0.123 0.278 0.167 0.278 -0.065 0.167 0.278 0.167 -0.412 0.167 0.389 0.278 -0.076 0.123
learning 0.056 0.123 0.345 0.456 -0.234 0.345 0.123 0.345 -0.045 0.234 0.234 0.345 -0.123 0.234 0.345 -0.234 0.123 0.345 0.123 0.345 0.056 0.123 0.345 0.456 -0.234 0.345 0.123 0.345 -0.045 0.234 0.234 0.345 -0.123 0.234 0.345 -0.234 0.123 0.345 0.123 0.345 0.056 0.123 0.345 0.456 -0.234 0.345 0.123 0.345 -0.045 0.234
intelligence 0.045 0.145 0.367 0.478 -0.212 0.367 0.145 0.367 -0.023 0.256 0.256 0.367 -0.098 0.256 0.367 -0.212 0.145 0.367 0.145 0.367 0.045 0.145 0.367 0.478 -0.212 0.367 0.145 0.367 -0.023 0.256 0.256 0.367 -0.098 0.256 0.367 -0.212 0.145 0.367 0.145 0.367 0.045 0.145 0.367 0.478 -0.212 0.367 0.145 0.367 -0.023 0.256
artificial 0.034 0.156 0.389 0.498 -0.198 0.389 0.167 0.389 -0.012 0.278 0.278 0.389 -0.076 0.278 0.389 -0.198 0.167 0.389 0.167 0.389 0.034 0.156 0.389 0.498 -0.198 0.389 0.167 0.389 -0.012 0.278 0.278 0.389 -0.076 0.278 0.389 -0.198 0.167 0.389 0.167 0.389 0.034 0.156 0.389 0.498 -0.198 0.389 0.167 0.389 -0.012 0.278
pizza -0.456 -0.345 0.123 -0.567 0.678 -0.345 0.456 -0.678 0.345 -0.456 0.567 -0.345 0.456 -0.678 0.123 0.456 -0.567 -0.345 0.234 -0.567 -0.456 -0.345 0.123 -0.567 0.678 -0.345 0.456 -0.678 0.345 -0.456 0.567 -0.345 0.456 -0.678 0.123 0.456 -0.567 -0.345 0.234 -0.567 -0.456 -0.345 0.123 -0.567 0.678 -0.345 0.456 -0.678 0.345 -0.456
food -0.432 -0.321 0.145 -0.543 0.698 -0.321 0.478 -0.654 0.367 -0.432 0.589 -0.321 0.478 -0.654 0.145 0.478 -0.543 -0.321 0.256 -0.543 -0.432 -0.321 0.145 -0.543 0.698 -0.321 0.478 -0.654 0.367 -0.432 0.589 -0.321 0.478 -0.654 0.145 0.478 -0.543 -0.321 0.256 -0.543 -0.432 -0.321 0.145 -0.543 0.698 -0.321 0.478 -0.654 0.367 -0.432
is 0.234 0.123 -0.234 0.056 0.189 -0.021 -0.289 0.034 0.067 -0.089 0.012 0.145 -0.123 0.023 -0.076 0.056 0.109 -0.019 -0.067 0.045 0.023 -0.008 0.056 -0.023 0.067 -0.056 0.019 -0.045 0.098 -0.012 0.034 -0.056 0.078 -0.023 0.045 -0.008 0.034 -0.019 0.056 -0.023 0.067 -0.034 0.019 -0.045 0.056 -0.012 0.023 -0.034 0.056 -0.008
and 0.256 0.145 -0.256 0.078 0.212 -0.012 -0.312 0.045 0.089 -0.098 0.023 0.167 -0.145 0.034 -0.089 0.067 0.123 -0.012 -0.078 0.056 0.034 -0.005 0.067 -0.019 0.078 -0.067 0.023 -0.056 0.109 -0.009 0.045 -0.067 0.089 -0.019 0.056 -0.005 0.045 -0.012 0.067 -0.019 0.078 -0.023 0.023 -0.056 0.067 -0.009 0.034 -0.045 0.067 -0.005"""


# ------------------------------------------------------------------------------
# 2. GloVe File Parser and Embedding Loader
# ------------------------------------------------------------------------------
def load_glove_embeddings(data_str: str) -> tuple[dict[str, np.ndarray], int]:
    """
    Parses a GloVe formatted text block or file into an in-memory word vector map.
    Returns (embeddings_dict, embedding_dimension).
    """
    embeddings = {}
    dim = None

    for line in data_str.strip().split("\n"):
        parts = line.strip().split()
        if not parts:
            continue
        word = parts[0].lower()
        vector = np.array([float(val) for val in parts[1:]], dtype=np.float32)
        if dim is None:
            dim = len(vector)
        embeddings[word] = vector

    return embeddings, dim


# ------------------------------------------------------------------------------
# 3. Document Representation via Average Word Embedding (Centroid Vector)
# ------------------------------------------------------------------------------
def get_document_centroid(text: str, embeddings: dict[str, np.ndarray], dim: int) -> np.ndarray:
    """
    Computes the centroid vector (average word embedding) for a document.
    Filters out OOV (out-of-vocabulary) tokens.
    """
    tokens = [w.strip(".,!?\"'").lower() for w in text.split()]
    valid_vectors = [embeddings[w] for w in tokens if w in embeddings]

    if not valid_vectors:
        # If all words are OOV, return a zero vector
        return np.zeros(dim, dtype=np.float32)

    # Centroid formula: v_doc = (1 / N) * sum(v_i)
    return np.mean(valid_vectors, axis=0)


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ------------------------------------------------------------------------------
# 4. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 06: GloVe Embeddings Loading and Vector Representation")
    print("=" * 75)

    # 1. Load GloVe Embeddings
    print("\n[A] Loading GloVe Embeddings Format...")
    embeddings, dim = load_glove_embeddings(SAMPLE_GLOVE_DATA)
    print(f"    Loaded Vocabulary Size: {len(embeddings)} words")
    print(f"    Embedding Dimensionality: {dim}-D")

    # 2. Word Vector Inspection
    sample_word = "computer"
    vec = embeddings[sample_word]
    print(f"\n[B] Word Vector Lookup for '{sample_word}':")
    print(f"    Length: {len(vec)} floats")
    preview = " ".join(f"{x:+.3f}" for x in vec[:8])
    print(f"    Sample Dimensions: [{preview} ...]")

    # 3. Word-to-Word Semantic Similarities
    print("\n[C] Word-Level Semantic Similarities using GloVe Vectors:")
    word_pairs = [
        ("king", "queen"),
        ("king", "man"),
        ("queen", "woman"),
        ("computer", "software"),
        ("computer", "pizza"),
        ("pizza", "food"),
        ("software", "food")
    ]
    for w1, w2 in word_pairs:
        sim = cosine_sim(embeddings[w1], embeddings[w2])
        print(f"    Similarity('{w1}', '{w2}') = {sim:+.4f}")

    # 4. Word Vector Analogy: king - man + woman ≈ queen
    print("\n[D] GloVe Vector Analogy Computation:")
    print("    Target: vector('king') - vector('man') + vector('woman')")
    analogy_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

    # Compute similarity of analogy vector against all vocabulary words
    scores = []
    for word, v in embeddings.items():
        if word not in ["king", "man"]:  # Exclude original query terms
            sim = cosine_sim(analogy_vec, v)
            scores.append((word, sim))
    scores.sort(key=lambda x: x[1], reverse=True)

    print("    Top Candidate Matches:")
    for rank, (candidate, score) in enumerate(scores[:3], start=1):
        print(f"      {rank}. {candidate:<10} (Cosine Similarity: {score:.4f})")

    # 5. Document-Level Representation (Centroid Pooling)
    print("\n" + "=" * 75)
    print(" DOCUMENT-LEVEL REPRESENTATION USING GLOVE CENTROIDS")
    print("=" * 75)

    sample_docs = [
        "python computer programming software",
        "natural language processing and machine learning",
        "delicious pizza and food",
        "the king and the queen"
    ]

    print("\n[E] Input Documents:")
    doc_vectors = []
    for i, doc in enumerate(sample_docs, start=1):
        d_vec = get_document_centroid(doc, embeddings, dim)
        doc_vectors.append(d_vec)
        print(f"    Doc {i}: \"{doc}\" (Centroid norm = {np.linalg.norm(d_vec):.4f})")

    # Compute pairwise document similarity matrix
    print("\n[F] Document-to-Document Cosine Similarity Matrix:")
    sim_mat = cosine_similarity(doc_vectors)
    header = f"{'':10} | " + " | ".join(f"Doc {j+1:>2}" for j in range(len(sample_docs)))
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(len(sample_docs)):
        row = f"Doc {i+1:<5} | " + " | ".join(f"{sim_mat[i][j]:>6.4f}" for j in range(len(sample_docs)))
        print(row)
    print("-" * len(header))

    print("\nKey Takeaways:")
    print("1. GloVe captures global co-occurrence statistics across documents.")
    print("2. Word analogies (king - man + woman = queen) hold geometrically in vector space.")
    print("3. Average word embeddings (centroids) create dense fixed-length document vectors.")

    print("\n" + "=" * 75)
    print(" Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
