"""
================================================================================
Program 05: Word2Vec Word Embeddings using Gensim on a Custom Corpus
================================================================================

What is Word2Vec?
-----------------
Word2Vec (Mikolov et al., 2013, Google) is a breakthrough neural word representation 
technique that maps words into dense, continuous low-dimensional vector spaces 
(typically 50 to 300 dimensions) where words with similar semantic and syntactic 
contexts are placed close to each other.

Key Architectures:
1. Continuous Bag-of-Words (CBOW, sg=0):
   - Predicts the target word given its surrounding context words.
   - Faster to train, slightly better for frequent words.
2. Continuous Skip-gram (Skip-gram, sg=1):
   - Predicts surrounding context words given a single center target word.
   - Slower, but performs significantly better on infrequent/rare words.

Core Capabilities Demonstrated:
1. Training Word2Vec embeddings on a custom domain corpus.
2. Vector inspection and dimensionality analysis.
3. Cosine similarity between word embedding vectors.
4. Top-N most similar words for a given target word.
5. Semantic vector arithmetic & word analogies:
   vector("queen") ≈ vector("king") - vector("man") + vector("woman")

Libraries Used:
- gensim: Standard industry library for Word2Vec (`gensim.models.Word2Vec`).
- NumPy: For vector math, cosine distance, and standalone fallback neural embeddings.
"""

import sys
import math
import numpy as np
from collections import defaultdict


# ------------------------------------------------------------------------------
# 1. Standalone Pure-NumPy Word2Vec (Skip-gram) Engine
#    Ensures seamless execution across all Python environments
# ------------------------------------------------------------------------------
class SimpleWord2VecScratch:
    """A clean, lightweight Skip-gram Word2Vec model in pure NumPy."""
    def __init__(self, vector_size: int = 16, learning_rate: float = 0.05, window: int = 2):
        self.vector_size = vector_size
        self.lr = learning_rate
        self.window = window
        self.word2idx = {}
        self.idx2word = {}
        self.W_in = None
        self.W_out = None

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, sentences: list[list[str]], epochs: int = 200):
        # Build vocabulary
        vocab = set(w for s in sentences for w in s)
        self.word2idx = {w: i for i, w in enumerate(sorted(vocab))}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        vocab_size = len(vocab)

        # Initialize weights
        np.random.seed(42)
        self.W_in = np.random.randn(vocab_size, self.vector_size) * 0.1
        self.W_out = np.random.randn(self.vector_size, vocab_size) * 0.1

        # Generate training pairs (center, context)
        pairs = []
        for s in sentences:
            for i, target in enumerate(s):
                for j in range(max(0, i - self.window), min(len(s), i + self.window + 1)):
                    if i != j:
                        pairs.append((self.word2idx[target], self.word2idx[s[j]]))

        # Train with gradient descent
        for _ in range(epochs):
            for target_idx, context_idx in pairs:
                h = self.W_in[target_idx]
                u = np.dot(h, self.W_out)
                y_pred = self._softmax(u)

                e = y_pred.copy()
                e[context_idx] -= 1.0

                grad_out = np.outer(h, e)
                grad_in = np.dot(self.W_out, e)

                self.W_out -= self.lr * grad_out
                self.W_in[target_idx] -= self.lr * grad_in

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.word2idx:
            raise KeyError(f"Word '{word}' not in vocabulary.")
        return self.W_in[self.word2idx[word]]

    def similarity(self, w1: str, w2: str) -> float:
        v1 = self.get_vector(w1)
        v2 = self.get_vector(w2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    def most_similar(self, positive: list[str], negative: list[str] = None, topn: int = 5):
        if negative is None:
            negative = []
        target_vec = np.zeros(self.vector_size)
        for w in positive:
            target_vec += self.get_vector(w)
        for w in negative:
            target_vec -= self.get_vector(w)

        norm_target = np.linalg.norm(target_vec)
        if norm_target > 0:
            target_vec /= norm_target

        scores = []
        exclude = set(positive + negative)
        for word, idx in self.word2idx.items():
            if word in exclude:
                continue
            vec = self.W_in[idx]
            norm_v = np.linalg.norm(vec)
            sim = float(np.dot(target_vec, vec) / norm_v) if norm_v > 0 else 0.0
            scores.append((word, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]


# ------------------------------------------------------------------------------
# 2. Gensim / Fallback Word2Vec Wrapper
# ------------------------------------------------------------------------------
def train_and_evaluate_word2vec(tokenized_corpus: list[list[str]]):
    has_gensim = False
    model = None

    try:
        from gensim.models import Word2Vec
        print("\n[Engine] Found 'gensim' library. Training via gensim.models.Word2Vec...")
        # Train CBOW and Skip-Gram models using Gensim
        model_cbow = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=20,
            window=3,
            min_count=1,
            sg=0,  # CBOW
            epochs=150,
            seed=42
        )
        model = model_cbow
        has_gensim = True
        print("  - Gensim Word2Vec model trained successfully.")
    except Exception as e:
        print(f"\n[Engine Notice] Gensim not compiled or available ({e}).")
        print("  Using built-in pure NumPy Skip-Gram Word2Vec implementation.")
        scratch_model = SimpleWord2VecScratch(vector_size=20, learning_rate=0.05, window=3)
        scratch_model.fit(tokenized_corpus, epochs=300)
        model = scratch_model

    # Common evaluation interface
    print("\n" + "=" * 75)
    print(" 1. VOCABULARY & EMBEDDING VECTOR INSPECTION")
    print("=" * 75)

    if has_gensim:
        vocab = list(model.wv.key_to_index.keys())
        get_vec = lambda w: model.wv[w]
        get_sim = lambda w1, w2: model.wv.similarity(w1, w2)
        get_similar = lambda positive=[], negative=[], n=5: model.wv.most_similar(positive=positive, negative=negative, topn=n)
    else:
        vocab = list(model.word2idx.keys())
        get_vec = lambda w: model.get_vector(w)
        get_sim = lambda w1, w2: model.similarity(w1, w2)
        get_similar = lambda positive=[], negative=[], n=5: model.most_similar(positive=positive, negative=negative, topn=n)

    print(f"\nVocabulary Size: {len(vocab)} unique words")
    print(f"Sample Vocabulary Terms: {vocab[:15]}")

    sample_word = "python" if "python" in vocab else vocab[0]
    vector = get_vec(sample_word)
    print(f"\nVector representation for '{sample_word}' (Dimensionality = {len(vector)}):")
    formatted_vec = " ".join(f"{x:+.3f}" for x in vector[:10])
    print(f"  [{formatted_vec} ...]")

    print("\n" + "=" * 75)
    print(" 2. WORD PAIR COSINE SIMILARITY")
    print("=" * 75)

    pairs_to_test = [
        ("python", "java"),
        ("ai", "machine"),
        ("deep", "neural"),
        ("computer", "nlp"),
        ("python", "pizza"),
    ]

    print("\nPairwise Semantic Similarities:")
    for w1, w2 in pairs_to_test:
        if w1 in vocab and w2 in vocab:
            sim = get_sim(w1, w2)
            print(f"  Similarity('{w1}', '{w2}') = {sim:+.4f}")

    print("\n" + "=" * 75)
    print(" 3. TOP-N MOST SIMILAR WORDS")
    print("=" * 75)

    query_words = ["ai", "python", "learning"]
    for query in query_words:
        if query in vocab:
            print(f"\nWords most similar to '{query}':")
            similars = get_similar([query], [], 4)
            for neighbor, score in similars:
                print(f"    -> {neighbor:<12} (score = {score:.4f})")

    print("\n" + "=" * 75)
    print(" 4. VECTOR ARITHMETIC & ANALOGIES")
    print("=" * 75)
    print("Formula: positive=[target1, target2], negative=[base]")
    print("Example: 'neural' is to 'deep' as 'machine' is to (?)")

    if all(w in vocab for w in ["neural", "deep", "machine"]):
        try:
            analogies = get_similar(positive=["neural", "machine"], negative=["deep"], n=3)
            print("Analogy query (neural - deep + machine):")
            for word, score in analogies:
                print(f"    -> {word:<12} (score = {score:.4f})")
        except Exception as err:
            print(f"  Analogy skipped: {err}")


# ------------------------------------------------------------------------------
# 3. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    # Domain corpus on AI, programming, and computer science
    raw_sentences = [
        "python is a versatile programming language for artificial intelligence",
        "java is an object oriented programming language for software development",
        "machine learning and deep learning are core pillars of ai",
        "deep neural networks learn hierarchical representations from complex data",
        "natural language processing enables computers to comprehend human language",
        "computer vision and nlp are specialized domains in artificial intelligence",
        "data scientists build machine learning models using python libraries",
        "software engineers write scalable code in java and python",
        "artificial intelligence algorithms require powerful computer hardware",
        "neural networks and deep learning achieve high accuracy in nlp tasks",
        "students love eating pizza and burgers during late night coding sessions"
    ]

    tokenized_corpus = [s.lower().split() for s in raw_sentences]

    print("=" * 75)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 05: Word2Vec Word Embeddings using Gensim on a Custom Corpus")
    print("=" * 75)

    print("\nTraining Corpus Summary:")
    print(f"  Total Sentences: {len(tokenized_corpus)}")
    total_tokens = sum(len(s) for s in tokenized_corpus)
    print(f"  Total Word Tokens: {total_tokens}")

    train_and_evaluate_word2vec(tokenized_corpus)

    print("\n" + "=" * 75)
    print(" Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
