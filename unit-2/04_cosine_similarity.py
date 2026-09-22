"""
================================================================================
Program 04: Cosine Similarity Computation between Text Documents
================================================================================

What is Cosine Similarity?
--------------------------
Cosine similarity is a foundational metric in Natural Language Processing (NLP) 
and Information Retrieval that measures the semantic or lexical similarity between 
two text documents by calculating the cosine of the angle between their vector 
representations.

Mathematical Formulation:
-------------------------
For two document vectors A and B:

                      A · B             ∑(A_i * B_i)
    Cosine(A, B) = ------------ = ----------------------------
                   ||A|| * ||B||   √(∑(A_i)^2) * √(∑(B_i)^2)

Where:
- A · B is the dot product of vectors A and B.
- ||A|| and ||B|| are the Euclidean (L2) norms (lengths) of vectors A and B.

Key Properties:
1. Scale Invariance: Measures document angle/orientation, not document length 
   (a short document and a long document on the same topic have high similarity).
2. Range: For non-negative word frequency vectors (BoW or TF-IDF), cosine similarity 
   ranges between:
   - 1.0 : Vectors point in identical directions (identical word distributions).
   - 0.0 : Vectors are orthogonal (share no common terms).
3. Cosine Distance: Distance = 1 - Cosine Similarity (ranges from 0 to 1).

Libraries Used:
- NumPy / Pure Python: For step-by-step vector arithmetic from first principles.
- scikit-learn: CountVectorizer, TfidfVectorizer, and cosine_similarity.
"""

import sys
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------------------
# 1. Cosine Similarity from First Principles (Scratch Implementation)
# ------------------------------------------------------------------------------
def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates dot product between two numerical vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def l2_norm(vec: list[float]) -> float:
    """Calculates Euclidean (L2) norm of a vector."""
    return math.sqrt(sum(x ** 2 for x in vec))


def cosine_similarity_scratch(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Computes cosine similarity between two vectors from first principles.
    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = l2_norm(vec_a)
    norm_b = l2_norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


# ------------------------------------------------------------------------------
# 2. Step-by-Step Demonstration on Sample Pairs
# ------------------------------------------------------------------------------
def demonstrate_step_by_step():
    print("\n" + "=" * 75)
    print(" 1. STEP-BY-STEP MATHEMATICAL COMPUTATION (FIRST PRINCIPLES)")
    print("=" * 75)

    doc1 = "artificial intelligence and machine learning"
    doc2 = "machine learning and deep learning algorithms"
    doc3 = "cooking recipes for delicious homemade pasta"

    print(f"\nDocument 1: \"{doc1}\"")
    print(f"Document 2: \"{doc2}\"")
    print(f"Document 3: \"{doc3}\"")

    # Build shared vocabulary
    corpus = [doc1, doc2, doc3]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    vocab = vectorizer.get_feature_names_out()

    print("\n[A] Shared Corpus Vocabulary (Alphabetical):")
    print(f"    {list(vocab)}")

    print("\n[B] Count Vectors:")
    for i, vec in enumerate(X, start=1):
        print(f"    Vector {i}: {vec.tolist()}")

    v1, v2, v3 = X[0].tolist(), X[1].tolist(), X[2].tolist()

    # Step-by-step between Doc 1 and Doc 2 (Related documents)
    dot_12 = dot_product(v1, v2)
    norm_1 = l2_norm(v1)
    norm_2 = l2_norm(v2)
    sim_12 = cosine_similarity_scratch(v1, v2)

    print("\n[C] Detailed Calculation for Doc 1 vs Doc 2:")
    print(f"    1. Dot Product (v1 . v2) = {dot_12}")
    print(f"    2. ||v1|| = sqrt({sum(x**2 for x in v1)}) = {norm_1:.4f}")
    print(f"    3. ||v2|| = sqrt({sum(x**2 for x in v2)}) = {norm_2:.4f}")
    print(f"    4. Cosine Sim = {dot_12} / ({norm_1:.4f} * {norm_2:.4f}) = {sim_12:.4f}")
    print(f"    5. Cosine Distance (1 - Sim) = {1 - sim_12:.4f}")

    # Calculation for Doc 1 and Doc 3 (Unrelated documents)
    dot_13 = dot_product(v1, v3)
    sim_13 = cosine_similarity_scratch(v1, v3)
    print("\n[D] Detailed Calculation for Doc 1 vs Doc 3 (Unrelated topics):")
    print(f"    1. Dot Product (v1 . v3) = {dot_13} (No shared words)")
    print(f"    2. Cosine Sim = {sim_13:.4f} (Orthogonal vectors)")


# ------------------------------------------------------------------------------
# 3. Full Pairwise Similarity Matrix Using scikit-learn
# ------------------------------------------------------------------------------
def demonstrate_pairwise_matrix(corpus: list[str]):
    print("\n" + "=" * 75)
    print(" 2. PAIRWISE SIMILARITY MATRICES (BoW vs TF-IDF)")
    print("=" * 75)

    # 1. Bag of Words Vectorization
    bow_vec = CountVectorizer()
    X_bow = bow_vec.fit_transform(corpus)
    sim_matrix_bow = cosine_similarity(X_bow)

    # 2. TF-IDF Vectorization
    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(corpus)
    sim_matrix_tfidf = cosine_similarity(X_tfidf)

    def print_matrix(matrix: np.ndarray, title: str):
        print(f"\n{title}:")
        n = len(corpus)
        header = f"{'':10} | " + " | ".join(f"Doc {j+1:>2}" for j in range(n))
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i in range(n):
            row = f"Doc {i+1:<5} | " + " | ".join(f"{matrix[i][j]:>6.4f}" for j in range(n))
            print(row)
        print("-" * len(header))

    print_matrix(sim_matrix_bow, "[A] Cosine Similarity Matrix using Bag-of-Words (Count)")
    print_matrix(sim_matrix_tfidf, "[B] Cosine Similarity Matrix using TF-IDF Weights")

    print("\n[C] Key Observations:")
    print("    - Diagonal elements are always 1.0000 (a document is identical to itself).")
    print("    - Matrices are symmetric: Sim(Doc_i, Doc_j) == Sim(Doc_j, Doc_i).")
    print("    - TF-IDF generally penalizes frequent non-informative words compared to pure counts.")


# ------------------------------------------------------------------------------
# 4. Main Driver
# ------------------------------------------------------------------------------
def main():
    sample_documents = [
        "Natural language processing helps computers comprehend human language.",
        "Deep learning and NLP algorithms process natural text and spoken words.",
        "Machine learning models classify and analyze speech and text data.",
        "Astronomers observed a distant supernova using orbital optical telescopes."
    ]

    print("=" * 75)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 04: Cosine Similarity Computation between Text Documents")
    print("=" * 75)

    print("\nSample Documents:")
    for i, doc in enumerate(sample_documents, start=1):
        print(f"  Doc {i}: \"{doc}\"")

    demonstrate_step_by_step()
    demonstrate_pairwise_matrix(sample_documents)

    print("\n" + "=" * 75)
    print(" Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
