"""
================================================================================
Program 07: Text Similarity using Word Mover's Distance (WMD)
================================================================================

What is Word Mover's Distance (WMD)?
-----------------------------------
Word Mover's Distance (Kusner et al., ICML 2015) is an advanced document distance 
metric that measures the semantic dissimilarity between two text documents. 

Rather than relying on exact word overlaps (like Bag-of-Words or TF-IDF), WMD 
represents documents as probability distributions over word embeddings (Normalized 
Bag of Words, nBoW) and calculates the minimum cumulative cost required to 
"transport" the words of document 1 to match the words of document 2.

Mathematical Formulation:
-------------------------
WMD is framed as a special case of the Earth Mover's Distance (Transportation Problem):

    min_T  ∑_{i=1}^m ∑_{j=1}^n  T_ij * c(i, j)

Subject to:
    ∑_{j=1}^n T_ij = d1_i    (Every word in Doc 1 must be transported)
    ∑_{i=1}^m T_ij = d2_j    (Every word in Doc 2 must be received)
    T_ij >= 0                (Non-negative flow)

Where:
- c(i, j) = ||x_i - x_j||_2 is the Euclidean distance between word embedding vectors.
- d1_i is the normalized frequency (count / total words) of word i in Document 1.
- d2_j is the normalized frequency of word j in Document 2.
- T_ij is the transport flow matrix indicating how much of word i moves to word j.

Why WMD is Superior to Cosine Similarity for Semantics:
-------------------------------------------------------
Consider:
  Doc A: "Obama speaks to the media in Illinois"
  Doc B: "The president greets the press in Chicago"

Under BoW / TF-IDF:
  - Exact word overlap = 0 (after stopword removal)
  - Cosine similarity = 0.0000 (Orthogonal / completely unrelated!)
Under WMD:
  - 'Obama' -> 'president', 'speaks' -> 'greets', 'media' -> 'press', 'Illinois' -> 'Chicago'
  - The word embeddings are very close, so WMD transport distance is very small!

Libraries Used:
- NumPy / SciPy (scipy.optimize.linprog): Canonical linear programming formulation.
- gensim: If available, uses `model.wv.wmdistance(...)`.
"""

import sys
import math
import numpy as np
from collections import Counter
from scipy.optimize import linprog


# ------------------------------------------------------------------------------
# 1. Semantic Embedding Space (Curated for WMD Demonstration)
# ------------------------------------------------------------------------------
EMBEDDINGS = {
    # Politics / Leadership
    "obama": np.array([2.5, 0.2, 0.1, 1.8]),
    "president": np.array([2.4, 0.3, 0.1, 1.9]),
    "leader": np.array([2.2, 0.4, 0.2, 1.7]),

    # Speech / Press
    "speaks": np.array([0.2, 2.8, 0.5, 0.1]),
    "greets": np.array([0.3, 2.7, 0.4, 0.2]),
    "addresses": np.array([0.2, 2.9, 0.6, 0.1]),

    # Journalism
    "media": np.array([0.5, 1.9, 2.8, 0.3]),
    "press": np.array([0.4, 1.8, 2.9, 0.2]),
    "journalists": np.array([0.5, 1.7, 3.0, 0.3]),

    # Geography
    "illinois": np.array([1.2, 0.1, 0.3, 3.2]),
    "chicago": np.array([1.1, 0.2, 0.4, 3.1]),

    # Unrelated Domain: Cooking / Food
    "chef": np.array([-2.0, -1.5, 0.1, -1.0]),
    "cooks": np.array([-2.1, -1.6, 0.0, -1.1]),
    "pasta": np.array([-3.0, -2.5, -0.5, -2.0]),
    "kitchen": np.array([-2.5, -2.0, -0.2, -1.5]),

    # General stopwords filtered out or low-weight
    "the": np.array([0.0, 0.0, 0.0, 0.0]),
    "to": np.array([0.0, 0.0, 0.0, 0.0]),
    "in": np.array([0.0, 0.0, 0.0, 0.0])
}

STOPWORDS = {"the", "to", "in", "a", "an", "and", "of"}


# ------------------------------------------------------------------------------
# 2. Canonical Word Mover's Distance using Linear Programming (Optimal Transport)
# ------------------------------------------------------------------------------
def compute_wmd_scratch(doc1_tokens: list[str], doc2_tokens: list[str], embeddings: dict) -> tuple[float, list]:
    """
    Computes Word Mover's Distance between two documents using the Earth Mover's 
    Distance linear programming formulation.
    """
    # Filter stopwords and words without embeddings
    d1_words = [w for w in doc1_tokens if w in embeddings and w not in STOPWORDS]
    d2_words = [w for w in doc2_tokens if w in embeddings and w not in STOPWORDS]

    if not d1_words or not d2_words:
        return float("inf"), []

    # Get unique words and normalized bag-of-words (nBoW)
    c1 = Counter(d1_words)
    c2 = Counter(d2_words)

    u1 = list(c1.keys())
    u2 = list(c2.keys())
    m, n = len(u1), len(u2)

    # Normalized word distribution vectors (sums to 1.0)
    w1 = np.array([c1[w] / len(d1_words) for w in u1], dtype=np.float64)
    w2 = np.array([c2[w] / len(d2_words) for w in u2], dtype=np.float64)

    # Cost matrix: Euclidean distance between word embedding pairs
    # c_ij = ||x_i - x_j||_2
    cost_matrix = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(embeddings[u1[i]] - embeddings[u2[j]])

    # Flatten cost matrix for linear programming: c^T * T
    c_vector = cost_matrix.flatten()

    # Formulate constraints:
    # 1. Sum over j of T_ij = w1_i for all i=1..m
    A_eq_1 = np.zeros((m, m * n))
    for i in range(m):
        for j in range(n):
            A_eq_1[i, i * n + j] = 1.0

    # 2. Sum over i of T_ij = w2_j for all j=1..n
    A_eq_2 = np.zeros((n, m * n))
    for j in range(n):
        for i in range(m):
            A_eq_2[j, i * n + j] = 1.0

    A_eq = np.vstack([A_eq_1, A_eq_2])
    b_eq = np.concatenate([w1, w2])

    # Variable bounds: T_ij >= 0
    bounds = [(0, None) for _ in range(m * n)]

    # Solve linear program
    res = linprog(c_vector, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return float("nan"), []

    wmd_value = res.fun
    flow_matrix = res.x.reshape((m, n))

    # Extract transport alignments for visualization
    alignments = []
    for i in range(m):
        for j in range(n):
            if flow_matrix[i, j] > 1e-4:
                alignments.append((u1[i], u2[j], flow_matrix[i, j], cost_matrix[i, j]))

    return wmd_value, alignments


# ------------------------------------------------------------------------------
# 3. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 07: Text Similarity using Word Mover's Distance (WMD)")
    print("=" * 75)

    doc_a = "obama speaks to the media in illinois"
    doc_b = "the president greets the press in chicago"
    doc_c = "chef cooks pasta in the kitchen"

    print("\nDocuments to Compare:")
    print(f"  Doc A: \"{doc_a}\"")
    print(f"  Doc B: \"{doc_b}\"")
    print(f"  Doc C: \"{doc_c}\"")

    tokens_a = doc_a.lower().split()
    tokens_b = doc_b.lower().split()
    tokens_c = doc_c.lower().split()

    # 1. Cosine Similarity baseline (demonstrating why Cosine fails here)
    print("\n" + "=" * 75)
    print(" 1. TRADITIONAL BAG-OF-WORDS / TF-IDF BASELINE")
    print("=" * 75)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform([doc_a, doc_b, doc_c])
    cos_sim = cosine_similarity(X)

    print(f"Cosine Similarity (Doc A, Doc B): {cos_sim[0, 1]:.4f}  <-- Identifies as ZERO similarity!")
    print(f"Cosine Similarity (Doc A, Doc C): {cos_sim[0, 2]:.4f}")
    print("Reason: Doc A and Doc B share zero common content words, so their count vectors are orthogonal.")

    # 2. Word Mover's Distance computation
    print("\n" + "=" * 75)
    print(" 2. WORD MOVER'S DISTANCE (OPTIMAL TRANSPORT CALCULATION)")
    print("=" * 75)

    dist_ab, flow_ab = compute_wmd_scratch(tokens_a, tokens_b, EMBEDDINGS)
    dist_ac, flow_ac = compute_wmd_scratch(tokens_a, tokens_c, EMBEDDINGS)
    dist_bc, flow_bc = compute_wmd_scratch(tokens_b, tokens_c, EMBEDDINGS)

    print(f"\n[A] WMD Distance(Doc A, Doc B) [Semantic Paraphrase] = {dist_ab:.4f}")
    print(f"[B] WMD Distance(Doc A, Doc C) [Unrelated Topic]     = {dist_ac:.4f}")
    print(f"[C] WMD Distance(Doc B, Doc C) [Unrelated Topic]     = {dist_bc:.4f}")

    print("\n[D] Optimal Word Transport Alignments (Doc A -> Doc B):")
    print(f"    {'Doc A Word':<12} -> {'Doc B Word':<12} | {'Flow (Weight)':<14} | {'Travel Distance'}")
    print("    " + "-" * 55)
    for w1, w2, flow, cost in flow_ab:
        print(f"    {w1:<12} -> {w2:<12} | {flow:<14.2f} | {cost:.4f}")
    print("    " + "-" * 55)

    print("\nKey Takeaways:")
    print("1. WMD measures the minimal cumulative travel cost between words across semantic space.")
    print("2. Doc A and Doc B have near-zero lexical overlap, but very low WMD (high semantic similarity).")
    print("3. Unlike BoW / TF-IDF, WMD effectively captures synonyms, paraphrases, and context.")

    print("\n" + "=" * 75)
    print(" Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
