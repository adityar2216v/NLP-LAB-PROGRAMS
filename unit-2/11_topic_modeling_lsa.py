"""
================================================================================
Program 11: Topic Modeling using Latent Semantic Analysis (LSA)
================================================================================

What is Latent Semantic Analysis (LSA)?
---------------------------------------
Latent Semantic Analysis (LSA), also termed Latent Semantic Indexing (LSI) in 
information retrieval (Deerwester et al., 1990), is an algebraic dimensionality 
reduction technique that uncovers hidden (latent) semantic concepts linking 
terms and documents.

Mathematical Formulation:
-------------------------
LSA applies Truncated Singular Value Decomposition (Truncated SVD) to a 
Term-Document Matrix (or TF-IDF matrix) X of size (m terms x n documents):

    X ≈ U_k * Σ_k * V_k^T

Where:
- U_k (m x k): Term-to-Concept matrix, representing terms in the k-dimensional concept space.
- Σ_k (k x k): Diagonal singular value matrix, encoding the strength/importance of each concept.
- V_k^T (k x n): Concept-to-Document matrix, encoding documents as coordinates in concept space.

Key Advantages:
1. Addresses Synonymy: Different words with identical underlying meaning map close 
   together in concept space (e.g., 'physician' and 'doctor').
2. Addresses Polysemy: Words with multiple meanings are disambiguated by the 
   combination of concepts.
3. High Noise Reduction: Removes low-variance noise from large sparse vocabularies.

LSA vs LDA Comparison:
----------------------
- LSA: Algebraic (SVD), continuous vector space, can have negative component loadings, 
  fast matrix operations.
- LDA: Probabilistic (Dirichlet priors), non-negative probability distributions, 
  more naturally interpretable as topic proportions.

Libraries Used:
- scikit-learn: TfidfVectorizer and TruncatedSVD.
- NumPy: For matrix operations, normalization, and cosine similarity.
"""

import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 11: Topic Modeling using Latent Semantic Analysis (LSA)")
    print("=" * 80)

    # 1. Multi-Domain Document Corpus
    corpus = [
        # Domain 1: Healthcare & Pharmacology
        "Doctors and physicians prescribe clinical antiviral medicines for viral infections",
        "Clinical trials test novel medications and vaccines to treat patient diseases",
        "Medical doctors diagnose cardiovascular disease and recommend health therapies",
        "Hospital patients receive prescription drugs, medical care, and clinical therapy",

        # Domain 2: Astrophysics & Deep Space
        "Astronomers observe distant spiral galaxies and planetary nebulae using telescopes",
        "Space exploration missions launch orbital satellite probes to study Mars and Jupiter",
        "NASA telescopes capture celestial cosmic radiation and distant supernova explosions",
        "Deep space observatory missions discover exoplanets and gravitational wave anomalies",

        # Domain 3: Machine Learning & Software
        "Software engineers write scalable python programs and train deep neural networks",
        "Machine learning algorithms and artificial intelligence process massive datasets",
        "Deep learning neural models optimize natural language and computer vision pipelines",
        "Cloud computing infrastructure accelerates model training using parallel GPU hardware"
    ]

    print(f"\nCorpus Size: {len(corpus)} documents across 3 knowledge domains.")

    # 2. TF-IDF Matrix Transformation
    print("\n" + "=" * 80)
    print(" 1. TF-IDF MATRIX TRANSFORMATION")
    print("=" * 80)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)
    vocab = tfidf_vectorizer.get_feature_names_out()

    print(f"  Vocabulary Size: {len(vocab)} distinct terms")
    print(f"  TF-IDF Matrix Shape (Docs x Terms): {X_tfidf.shape}")

    # 3. Truncated SVD (Latent Semantic Analysis)
    print("\n" + "=" * 80)
    print(" 2. TRUNCATED SINGULAR VALUE DECOMPOSITION (SVD)")
    print("=" * 80)

    NUM_COMPONENTS = 3
    lsa_model = TruncatedSVD(n_components=NUM_COMPONENTS, algorithm='randomized', random_state=42)
    doc_concept_matrix = lsa_model.fit_transform(X_tfidf)

    print(f"  Decomposed into k={NUM_COMPONENTS} latent semantic concepts.")
    print(f"  Singular Values (Concept Strengths):")
    for i, s in enumerate(lsa_model.singular_values_, start=1):
        print(f"    Concept {i}: {s:.4f}")

    explained_var = lsa_model.explained_variance_ratio_
    print(f"\n  Explained Variance Ratio per Concept:")
    for i, var in enumerate(explained_var, start=1):
        print(f"    Concept {i}: {var * 100:.2f}%")
    print(f"  Cumulative Variance Preserved: {np.sum(explained_var) * 100:.2f}%")

    # 4. Extract Top Terms per Latent Semantic Concept
    print("\n" + "=" * 80)
    print(" 3. TOP CONTRIBUTING TERMS PER LATENT CONCEPT")
    print("=" * 80)

    for i, comp in enumerate(lsa_model.components_, start=1):
        # Top positive weight terms
        top_pos_idx = comp.argsort()[:-7:-1]
        top_pos = [vocab[idx] for idx in top_pos_idx]
        print(f"\n[Concept #{i}]")
        print(f"  Dominant Keywords: {', '.join(top_pos)}")

    # 5. Documents in Low-Dimensional Concept Space
    print("\n" + "=" * 80)
    print(" 4. DOCUMENT REPRESENTATION IN CONCEPT SPACE (V_k^T)")
    print("=" * 80)

    header = f"{'Document':<8} | {'Concept 1':>10} | {'Concept 2':>10} | {'Concept 3':>10} | Dominant Concept"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for doc_idx, row in enumerate(doc_concept_matrix, start=1):
        dominant = np.argmax(np.abs(row)) + 1
        print(f"Doc {doc_idx:<4} | {row[0]:>10.4f} | {row[1]:>10.4f} | {row[2]:>10.4f} | Concept {dominant}")
    print("-" * len(header))

    # 6. Querying in Semantic Concept Space
    print("\n" + "=" * 80)
    print(" 5. SEMANTIC SEARCH QUERY IN REDUCED CONCEPT SPACE")
    print("=" * 80)

    query = ["Planetary telescopes observe deep stars in galaxies"]
    query_tfidf = tfidf_vectorizer.transform(query)
    query_concept = lsa_model.transform(query_tfidf)

    # Compute cosine similarity between query concept vector and all document concept vectors
    concept_similarities = cosine_similarity(query_concept, doc_concept_matrix)[0]
    ranked_indices = np.argsort(concept_similarities)[::-1]

    print(f"User Query: \"{query[0]}\"")
    print("Top 3 Semantically Retrieved Documents:")
    for rank, doc_id in enumerate(ranked_indices[:3], start=1):
        print(f"  {rank}. Doc {doc_id + 1} (Cosine Sim: {concept_similarities[doc_id]:.4f})")
        print(f"     \"{corpus[doc_id]}\"")

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
