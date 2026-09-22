"""
================================================================================
Program 14: Information Retrieval System with Ranking using TF-IDF
================================================================================

What is an Information Retrieval (IR) System?
---------------------------------------------
An Information Retrieval (IR) system locates, scores, and ranks unstructured text 
documents from a large corpus in response to a user's information need (search query).

The Vector Space Model (VSM) for IR:
------------------------------------
In the classic Salton Vector Space Model:
1. Document Indexing: Each document is represented as a high-dimensional vector of 
   TF-IDF weights:
       w_{t, d} = TF(t, d) * IDF(t)
2. Query Representation: The user search query is vectorized into the exact same 
   feature space:
       w_{t, q} = TF(t, q) * IDF(t)
3. Relevance Scoring: The relevance score between query q and document d is 
   computed as their Cosine Similarity:
                    q · d
       Score(q, d) = -----------
                    ||q|| * ||d||
4. Top-K Ranking: Documents with the highest cosine scores are ranked at the top.
5. Snippet Generation: Contextual excerpts containing matched keywords are extracted 
   to provide search result summaries.

Core Capabilities Demonstrated:
1. Document collection indexing with sublinear TF scaling and smooth IDF.
2. Free-text query parsing and vector projection.
3. Cosine similarity scoring and top-K document ranking.
4. Dynamic result snippet extraction highlighting matched query terms.
5. Search evaluation demonstration (Precision@K).

Libraries Used:
- scikit-learn: TfidfVectorizer, cosine_similarity.
- NumPy: Vector manipulations and sorting.
"""

import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------------------
# 1. Corpus of Technical Knowledge Articles
# ------------------------------------------------------------------------------
DOCUMENT_CORPUS = [
    {
        "id": "DOC-01",
        "title": "Introduction to Natural Language Processing",
        "text": "Natural language processing enables computers to comprehend, interpret, and manipulate human languages using statistical algorithms and computational linguistics."
    },
    {
        "id": "DOC-02",
        "title": "Deep Learning and Neural Networks in AI",
        "text": "Deep artificial neural networks learn multi-layered feature representations from large datasets, achieving breakthrough performance in vision and audio recognition."
    },
    {
        "id": "DOC-03",
        "title": "Transformer Models and Large Language Models",
        "text": "Transformer architectures utilize self-attention mechanisms to process sequential natural language text and power modern generative AI language models."
    },
    {
        "id": "DOC-04",
        "title": "Computer Vision and Convolutional Networks",
        "text": "Convolutional neural networks analyze pixel patterns in digital image and video data to perform object detection, segmentation, and automated medical imaging."
    },
    {
        "id": "DOC-05",
        "title": "Information Retrieval and Search Engine Architecture",
        "text": "Modern search engines employ inverted indices, vector space scoring, and TF-IDF relevance ranking to retrieve documents matching complex user queries."
    },
    {
        "id": "DOC-06",
        "title": "Cloud Computing and Distributed Microservices",
        "text": "Distributed cloud computing systems deploy containerized microservices across server clusters with load balancing, automated scaling, and low latency."
    }
]


# ------------------------------------------------------------------------------
# 2. Vector Space Information Retrieval Engine
# ------------------------------------------------------------------------------
class TfidfSearchEngine:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_texts = [d["text"] for d in documents]

        # Initialize TF-IDF Vectorizer with English stopwords removal and sublinear TF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        self.vocabulary = self.vectorizer.get_feature_names_out()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Processes query, calculates cosine relevance against all indexed documents, 
        and returns the top-K ranked hits.
        """
        # Transform query into the indexed vector space
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query vector and all document vectors
        # Shape: (1, num_docs)
        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Sort indices by descending score
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            score = scores[idx]
            if score > 0.0:  # Only return documents with positive relevance
                doc = self.documents[idx]
                results.append({
                    "rank": rank,
                    "id": doc["id"],
                    "title": doc["title"],
                    "score": score,
                    "text": doc["text"],
                    "snippet": self._generate_snippet(query, doc["text"])
                })
        return results

    def _generate_snippet(self, query: str, text: str) -> str:
        """Extracts and highlights snippet around query keywords."""
        q_terms = set(re.findall(r"\b\w+\b", query.lower()))
        words = text.split()
        highlighted = []
        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in q_terms:
                highlighted.append(f"[{w}]")
            else:
                highlighted.append(w)
        return " ".join(highlighted)


# ------------------------------------------------------------------------------
# 3. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 14: Information Retrieval System with Ranking using TF-IDF")
    print("=" * 80)

    print(f"\n[Index Status] Indexing {len(DOCUMENT_CORPUS)} Knowledge Base Documents...")
    engine = TfidfSearchEngine(DOCUMENT_CORPUS)
    print(f"  Indexed Feature Terms: {len(engine.vocabulary)} unique words")

    sample_queries = [
        "natural language processing models",
        "search engine ranking with tf-idf",
        "convolutional networks for computer vision",
        "distributed cloud computing architecture"
    ]

    print("\n" + "=" * 80)
    print(" 1. EXECUTING RANKED SEARCH QUERIES")
    print("=" * 80)

    for q_idx, query in enumerate(sample_queries, start=1):
        print(f"\nQuery #{q_idx}: \"{query}\"")
        hits = engine.search(query, top_k=3)

        if not hits:
            print("  No relevant documents found.")
            continue

        print(f"  Found {len(hits)} matching documents:")
        for hit in hits:
            print(f"    Rank {hit['rank']}: [{hit['id']}] {hit['title']}")
            print(f"            Relevance Score (Cosine): {hit['score']:.4f}")
            print(f"            Snippet: \"{hit['snippet']}\"")

    # Interactive-style query simulation
    print("\n" + "=" * 80)
    print(" 2. DETAILED VECTOR SPACE QUERY INSPECTION")
    print("=" * 80)

    detailed_query = "natural language attention mechanisms"
    print(f"Inspecting Query: \"{detailed_query}\"")
    q_vec = engine.vectorizer.transform([detailed_query]).toarray()[0]
    matched_features = [
        (engine.vocabulary[i], q_vec[i])
        for i in range(len(q_vec)) if q_vec[i] > 0
    ]
    print(f"Non-zero Query Vector Weights:")
    for term, weight in matched_features:
        print(f"  - '{term}': weight = {weight:.4f}")

    hits = engine.search(detailed_query, top_k=2)
    print(f"\nTop Ranked Document for Inspection:")
    if hits:
        top_hit = hits[0]
        print(f"  Winner: {top_hit['id']} - {top_hit['title']} (Cosine: {top_hit['score']:.4f})")
        print(f"  Text: \"{top_hit['text']}\"")

    print("\nKey Takeaways:")
    print("1. Vector Space Model converts free-text queries into mathematical vectors.")
    print("2. TF-IDF weighting balances term frequency within document vs corpus rarity.")
    print("3. Cosine similarity produces continuous [0.0, 1.0] ranking scores for search hits.")

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
