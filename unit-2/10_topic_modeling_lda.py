"""
================================================================================
Program 10: Topic Modeling using Latent Dirichlet Allocation (LDA)
================================================================================

What is Topic Modeling?
-----------------------
Topic modeling is an unsupervised machine learning technique in NLP designed to 
automatically discover latent (hidden) thematic topics across a collection of 
unstructured text documents.

What is Latent Dirichlet Allocation (LDA)?
-----------------------------------------
Developed by Blei, Ng, and Jordan (2003), LDA is a generative probabilistic 
graphical model that posits:
1. Every Document is a mixture of multiple topics with Dirichlet prior alpha (α).
2. Every Topic is a discrete probability distribution over the vocabulary with 
   Dirichlet prior beta (β).

Generative Process for Each Document d:
---------------------------------------
1. Draw document topic distribution: θ_d ~ Dirichlet(α)
2. For each word position n in document d:
   a. Sample a topic assignment: z_dn ~ Multinomial(θ_d)
   b. Sample a word token: w_dn ~ Multinomial(φ_{z_dn}), where φ_k ~ Dirichlet(β)

Key Concepts Demonstrated:
1. Preprocessing and Document-Term Count Matrix construction (CountVectorizer).
2. Training an LDA model using online Variational Bayes.
3. Extracting and interpreting top keywords per latent topic.
4. Document-Topic probability distribution (mixture proportions).
5. Inferring topic distributions for new, unseen documents.

Libraries Used:
- scikit-learn: CountVectorizer and LatentDirichletAllocation.
- NumPy: For probability manipulation and matrix formatting.
"""

import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 10: Topic Modeling using Latent Dirichlet Allocation (LDA)")
    print("=" * 80)

    # 1. Multi-Topic Document Corpus
    corpus = [
        # Topic A: Space & Astronomy
        "Astronomers use optical telescopes to observe distant galaxies and stars in space",
        "NASA launched a planetary probe mission to study orbit trajectories of Mars and Jupiter",
        "Space exploration missions discover exoplanets, stellar nebulae, and cosmic radiation",
        "Telescope observatories detect gravitational waves and celestial supernova explosions",

        # Topic B: Medicine & Healthcare
        "Doctors and physicians prescribe antiviral drugs and clinical treatments to patients",
        "Hospitals conduct clinical medical trials to evaluate drug efficacy against infections",
        "Cardiovascular diseases require regular medical diagnosis, therapy, and health monitoring",
        "Medical vaccines and antibiotic medications treat chronic illnesses and bacterial symptoms",

        # Topic C: Computing & Artificial Intelligence
        "Computer scientists train deep neural networks for artificial intelligence applications",
        "Machine learning algorithms optimize software performance on parallel GPU processors",
        "Natural language processing and computer vision systems learn representations from data",
        "Software engineers write scalable code and deploy cloud computing architectures"
    ]

    print(f"\nCorpus Size: {len(corpus)} documents across 3 distinct domains.")

    # 2. Vectorization: Document-Term Count Matrix
    # (Note: LDA requires raw counts, not TF-IDF, because it models discrete word generation)
    print("\n" + "=" * 80)
    print(" 1. DOCUMENT-TERM COUNT MATRIX GENERATION")
    print("=" * 80)

    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    dtm = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    print(f"  Vocabulary Size: {len(feature_names)} unique content terms")
    print(f"  DTM Shape (Docs x Terms): {dtm.shape}")

    # 3. Fit Latent Dirichlet Allocation Model
    print("\n" + "=" * 80)
    print(" 2. FITTING LATENT DIRICHLET ALLOCATION (LDA)")
    print("=" * 80)

    NUM_TOPICS = 3
    lda_model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=42,
        learning_method='batch',
        max_iter=30
    )
    doc_topic_dist = lda_model.fit_transform(dtm)

    print(f"  Trained LDA model with K={NUM_TOPICS} latent topics.")

    # 4. Display Top Keywords per Discovered Topic
    print("\n" + "=" * 80)
    print(" 3. EXTRACTED TOPICS AND TOP KEYWORDS")
    print("=" * 80)

    topic_names = []
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words in topic by descending probability
        top_indices = topic.argsort()[:-8:-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\n[Topic #{topic_idx + 1}]")
        print(f"  Top Keywords: {', '.join(top_words)}")
        # Infer semantic label based on top terms
        if any(w in top_words for w in ['space', 'missions', 'astronomers', 'telescopes', 'probe']):
            label = "Astronomy & Space Exploration"
        elif any(w in top_words for w in ['medical', 'clinical', 'patients', 'doctors', 'treatments']):
            label = "Medicine & Healthcare"
        else:
            label = "Computing & Artificial Intelligence"
        topic_names.append(label)
        print(f"  Interpreted Theme: {label}")

    # 5. Document-Topic Probability Distribution Matrix
    print("\n" + "=" * 80)
    print(" 4. DOCUMENT-TOPIC PROBABILITY DISTRIBUTION TABLE")
    print("=" * 80)

    header = f"{'Doc':<6} | " + " | ".join(f"Topic {k+1} ({topic_names[k][:10]})" for k in range(NUM_TOPICS)) + " | Dominant Topic"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for doc_idx, dist in enumerate(doc_topic_dist, start=1):
        dominant = np.argmax(dist) + 1
        scores_str = " | ".join(f"{prob:>21.4f}" for prob in dist)
        print(f"Doc {doc_idx:<2} | {scores_str} | Topic {dominant}")
    print("-" * len(header))

    # 6. Topic Inference on Unseen Test Query
    print("\n" + "=" * 80)
    print(" 5. TOPIC INFERENCE ON NEW / UNSEEN DOCUMENT")
    print("=" * 80)

    new_doc = ["Hospitals evaluate patient vaccines and clinical medical therapies"]
    new_dtm = vectorizer.transform(new_doc)
    new_dist = lda_model.transform(new_dtm)[0]

    print(f"Unseen Document: \"{new_doc[0]}\"")
    print("Topic Probabilities:")
    for k in range(NUM_TOPICS):
        print(f"  -> Topic {k+1} ({topic_names[k]}): {new_dist[k] * 100:.2f}%")
    best_topic = np.argmax(new_dist)
    print(f"Predicted Dominant Topic: Topic {best_topic + 1} ({topic_names[best_topic]})")

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
