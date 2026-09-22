# NLP Lab Programs

## Description

This repository contains the complete **Unit 1 and Unit 2 Natural Language Processing (NLP)** laboratory practical programs. It covers foundational text preprocessing, vector space modeling, dense embeddings, classification, topic modeling, sentiment analysis, opinion mining, information extraction, and retrieval systems essential for undergraduate and postgraduate computer science curricula. The codebase is implemented in Python using industry-standard NLP libraries (**NLTK**, **spaCy**, **scikit-learn**, **SciPy**, and **TextBlob**), formatted cleanly for academic lab evaluations, practical examinations, and viva voce preparation.

---

## Objectives

* Understand foundational NLP text preprocessing pipelines (tokenization, stemming, lemmatization, stopword removal).
* Perform grammatical syntactic analysis via POS tagging, shallow chunking, and dependency parsing.
* Extract real-world named entities (PERSON, ORG, GPE, DATE, MONEY) using NER pipelines.
* Construct statistical document representations (Bag of Words, TF-IDF, Word and Character N-grams).
* Measure document semantic similarity using vector space models (Cosine Similarity, Word Mover's Distance).
* Learn distributed word representations and vector geometries using Word2Vec and GloVe embeddings.
* Train supervised machine learning text classifiers (Multinomial Naive Bayes, Linear Support Vector Machines).
* Perform lexicon- and rule-based sentiment analysis with TextBlob and NLTK VADER.
* Uncover latent thematic concepts using unsupervised topic modeling (LDA and LSA).
* Implement aspect-based opinion mining on real customer reviews to extract product feature scorecards.
* Build information extraction pipelines to extract entities, regex attributes, and semantic relation triples.
* Design and implement a full-text Information Retrieval (IR) search engine with TF-IDF relevance ranking.

---

## Technologies Used

* **Python 3.x**
* **NLTK (Natural Language Toolkit)**
* **spaCy (Industrial-Strength NLP)**
* **scikit-learn (Machine Learning & Feature Extraction)**
* **NumPy & SciPy (Numerical Computing & Scientific Optimization)**
* **TextBlob (Simplified Text Processing & Sentiment)**

---

## Unit 1 Programs (CO1)

| S. No. | Program | File | CO |
| :---: | :--- | :--- | :---: |
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | `Unit-1/01_tokenization.py` | CO1 |
| 2 | Stemming and Lemmatization on Sample Text | `Unit-1/02_stemming_lemmatization.py` | CO1 |
| 3 | Stop-word Removal from a Document | `Unit-1/03_stopword_removal.py` | CO1 |
| 4 | Part-of-Speech (POS) Tagging of a Given Sentence | `Unit-1/04_pos_tagging.py` | CO1 |
| 5 | Parsing and Chunking using RegEx and spaCy | `Unit-1/05_parsing_chunking.py` | CO1 |
| 6 | Named Entity Recognition (NER) using spaCy | `Unit-1/06_named_entity_recognition.py` | CO1 |

---

## Unit 2 Programs (CO2)

The serial numbers within Unit 2 are organized sequentially from **1 to 14**, with the corresponding overall syllabus numbering (**8 to 21**) indicated for reference:

| Unit S. No. | Syllabus S. No. | Program Name | File | CO |
| :---: | :---: | :--- | :--- | :---: |
| **1** | 8 | Bag-of-Words (BoW) Vectorization and Representation | `unit-2/01_bag_of_words.py` | CO2 |
| **2** | 9 | TF-IDF Implementation and Comparison with BoW | `unit-2/02_tfidf.py` | CO2 |
| **3** | 10 | N-Gram Model (Uni-, Bi-, Tri-gram) Generation from Corpus | `unit-2/03_ngrams.py` | CO2 |
| **4** | 11 | Cosine Similarity Computation between Text Documents | `unit-2/04_cosine_similarity.py` | CO2 |
| **5** | 12 | Word2Vec Word Embeddings using Gensim on a Custom Corpus | `unit-2/05_word2vec_gensim.py` | CO2 |
| **6** | 13 | GloVe Embeddings Loading and Vector Representation | `unit-2/06_glove_embeddings.py` | CO2 |
| **7** | 14 | Text Similarity using Word Mover's Distance (WMD) | `unit-2/07_word_movers_distance.py` | CO2 |
| **8** | 15 | Text Classification using Naive Bayes/SVM with TF-IDF | `unit-2/08_text_classification_nb_svm.py` | CO2 |
| **9** | 16 | Sentiment Analysis using TextBlob and VADER | `unit-2/09_sentiment_analysis_textblob_vader.py` | CO2 |
| **10** | 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | `unit-2/10_topic_modeling_lda.py` | CO2 |
| **11** | 18 | Topic Modeling using Latent Semantic Analysis (LSA) | `unit-2/11_topic_modeling_lsa.py` | CO2 |
| **12** | 19 | Opinion Mining on Product/Service Reviews Dataset | `unit-2/12_opinion_mining_reviews.py` | CO2 |
| **13** | 20 | Information Extraction (IE) from Documents | `unit-2/13_information_extraction.py` | CO2 |
| **14** | 21 | Information Retrieval System with Ranking using TF-IDF | `unit-2/14_information_retrieval_tfidf.py` | CO2 |

---

## Repository Structure

```text
NLP-Lab-Programs/
├── README.md
├── requirements.txt
├── Unit-1/
│   ├── 01_tokenization.py
│   ├── 02_stemming_lemmatization.py
│   ├── 03_stopword_removal.py
│   ├── 04_pos_tagging.py
│   ├── 05_parsing_chunking.py
│   └── 06_named_entity_recognition.py
└── unit-2/
    ├── 01_bag_of_words.py
    ├── 02_tfidf.py
    ├── 03_ngrams.py
    ├── 04_cosine_similarity.py
    ├── 05_word2vec_gensim.py
    ├── 06_glove_embeddings.py
    ├── 07_word_movers_distance.py
    ├── 08_text_classification_nb_svm.py
    ├── 09_sentiment_analysis_textblob_vader.py
    ├── 10_topic_modeling_lda.py
    ├── 11_topic_modeling_lsa.py
    ├── 12_opinion_mining_reviews.py
    ├── 13_information_extraction.py
    └── 14_information_retrieval_tfidf.py
```

---

## Installation

Follow these steps to set up the lab environment on your system:

### 1. Prerequisites
Ensure **Python 3.8+** is installed. Verify with:
```bash
python --version
```

### 2. Open Project Directory
Navigate to the repository root directory:
```bash
cd NLP-Lab-Programs
```

### 3. (Optional) Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Download Required NLTK Corpora and Models
Pre-download required NLTK datasets using:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('vader_lexicon')"
```

### 6. Download spaCy English Language Model
Download the small English pipeline (`en_core_web_sm`):
```bash
python -m spacy download en_core_web_sm
```

---

## How to Run

Execute each program individually from the repository root:

```bash
# ==============================================================================
# Unit 1 Programs
# ==============================================================================
python Unit-1/01_tokenization.py
python Unit-1/02_stemming_lemmatization.py
python Unit-1/03_stopword_removal.py
python Unit-1/04_pos_tagging.py
python Unit-1/05_parsing_chunking.py
python Unit-1/06_named_entity_recognition.py

# ==============================================================================
# Unit 2 Programs
# ==============================================================================
python unit-2/01_bag_of_words.py
python unit-2/02_tfidf.py
python unit-2/03_ngrams.py
python unit-2/04_cosine_similarity.py
python unit-2/05_word2vec_gensim.py
python unit-2/06_glove_embeddings.py
python unit-2/07_word_movers_distance.py
python unit-2/08_text_classification_nb_svm.py
python unit-2/09_sentiment_analysis_textblob_vader.py
python unit-2/10_topic_modeling_lda.py
python unit-2/11_topic_modeling_lsa.py
python unit-2/12_opinion_mining_reviews.py
python unit-2/13_information_extraction.py
python unit-2/14_information_retrieval_tfidf.py
```

---

## Unit 1 Program Summaries

### 1. Tokenization of Sentences and Words (`Unit-1/01_tokenization.py`)
* **Purpose:** Break unstructured text into manageable linguistic segments.
* **Main Concept:** Sentence tokenization detects sentence boundaries (`.`, `!`, `?`), while word tokenization segments words and punctuation.
* **Libraries Used:** NLTK (`sent_tokenize`, `word_tokenize`) and spaCy (`nlp()`, `doc.sents`, `token.text`).
* **Expected Output:** Numbered list of sentences followed by an array of individual word tokens for both NLTK and spaCy.

### 2. Stemming and Lemmatization on Sample Text (`Unit-1/02_stemming_lemmatization.py`)
* **Purpose:** Reduce inflected word variants to their base or root forms.
* **Main Concept:** Stemming uses heuristic suffix-stripping rules. Lemmatization uses morphological analysis with WordNet dictionary lookup to yield valid dictionary lemmas.
* **Libraries Used:** NLTK (`PorterStemmer`, `WordNetLemmatizer`).
* **Expected Output:** Side-by-side comparative table showing Original Word, Porter Stemmer, Lemmatizer (Noun), and Lemmatizer (Verb).

### 3. Stop-word Removal from a Document (`Unit-1/03_stopword_removal.py`)
* **Purpose:** Filter out high-frequency grammatical words carrying minimal semantic signal.
* **Main Concept:** Words such as *is*, *an*, *the*, and *in* are filtered out to reduce dimensionality.
* **Library Used:** NLTK (`nltk.corpus.stopwords`, `word_tokenize`).
* **Expected Output:** Original text, raw tokens, removed stopwords, filtered tokens, cleaned text, and statistical reduction percentage.

### 4. Part-of-Speech (POS) Tagging (`Unit-1/04_pos_tagging.py`)
* **Purpose:** Assign grammatical categories to words in a sentence based on syntactic context.
* **Main Concept:** Disambiguates syntactic functions using the Penn Treebank tagset.
* **Library Used:** NLTK (`pos_tag`, `word_tokenize`).
* **Expected Output:** Structured 3-column table displaying Token, POS Tag, and grammatical explanation.

### 5. Parsing and Chunking using RegEx and spaCy (`Unit-1/05_parsing_chunking.py`)
* **Purpose:** Group words into phrases and extract dependency relations.
* **Main Concept:** Shallow parsing identifies Noun Phrases (`NP`) using regex over POS tags; dependency parsing extracts head-modifier links.
* **Libraries Used:** NLTK (`RegexpParser`, `pos_tag`) and spaCy (`doc.noun_chunks`, dependency tags).
* **Expected Output:** Text-formatted parse tree, extracted NP chunks, and spaCy dependency parsing table.

### 6. Named Entity Recognition (NER) (`Unit-1/06_named_entity_recognition.py`)
* **Purpose:** Identify and classify named entities into real-world categories.
* **Main Concept:** Detects PERSON, ORG, GPE, DATE, and MONEY entities from unstructured text.
* **Library Used:** spaCy (`en_core_web_sm`).
* **Expected Output:** Original sentence, tabular list of detected entities with categories, descriptions, and category-grouped summary.

---

## Unit 2 Program Summaries

### 1. Bag of Words (BoW) Model (`unit-2/01_bag_of_words.py`)
* **Purpose:** Represent documents as numerical word occurrence vectors across a defined vocabulary.
* **Main Concept:** Extracts unique corpus vocabulary and builds term count and binary vectors while discarding word order.
* **Libraries Used:** Pure Python (from scratch) and scikit-learn (`CountVectorizer`, `cosine_similarity`).
* **Expected Output:** Extracted vocabulary, count matrix, binary matrix, scikit-learn feature mapping, and pairwise similarity table.

### 2. Term Frequency - Inverse Document Frequency (`unit-2/02_tfidf.py`)
* **Purpose:** Evaluate the relative importance of words within documents relative to the entire corpus.
* **Main Concept:** Offsets high-frequency terms by multiplying Term Frequency ($TF$) by Inverse Document Frequency ($IDF$), followed by $L_2$ normalization.
* **Libraries Used:** NumPy (from scratch) and scikit-learn (`TfidfVectorizer`).
* **Expected Output:** Step-by-step TF table, DF and smooth IDF values, raw TF-IDF matrix, unit-norm normalized matrix, and top salient keywords.

### 3. N-gram Modeling and Feature Extraction (`unit-2/03_ngrams.py`)
* **Purpose:** Capture local sequential context and multi-word phrases beyond unigrams.
* **Main Concept:** Extracts contiguous sequences of $n$ words or characters and calculates conditional transition probabilities $P(w_n | w_{n-1})$.
* **Libraries Used:** Pure Python, NLTK (`ngrams`, `FreqDist`), and scikit-learn (`CountVectorizer(ngram_range=...)`).
* **Expected Output:** Lists of word unigrams, bigrams, trigrams; character trigrams; most frequent bigrams; transition probabilities; and scikit-learn n-gram matrices.

### 4. Cosine Similarity between Documents (`unit-2/04_cosine_similarity.py`)
* **Purpose:** Measure semantic or lexical closeness between text documents by computing vector angles.
* **Main Concept:** Evaluates $\text{Cosine}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$. Scale invariant and length independent.
* **Libraries Used:** NumPy (from scratch) and scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `cosine_similarity`).
* **Expected Output:** Step-by-step dot product and Euclidean norm calculation, pairwise similarity matrices for BoW vs TF-IDF, and cosine distance analysis.

### 5. Word2Vec Word Embeddings on Custom Corpus (`unit-2/05_word2vec_gensim.py`)
* **Purpose:** Learn dense, low-dimensional continuous vector representations for words from context.
* **Main Concept:** Compares Continuous Bag-of-Words (CBOW) and Skip-gram architectures. Demonstrates word analogies: $\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})$.
* **Libraries Used:** `gensim.models.Word2Vec` (with pure NumPy neural Skip-Gram fallback engine).
* **Expected Output:** Learned vocabulary, vector inspection, pairwise word similarities, top-N nearest neighbors, and semantic vector analogies.

### 6. GloVe Embeddings Loading & Document Vectors (`unit-2/06_glove_embeddings.py`)
* **Purpose:** Parse pre-trained global co-occurrence word embeddings and compose document-level vectors.
* **Main Concept:** GloVe factorizes global log co-occurrence counts. Documents are represented via Average Word Embeddings (Centroid Vectors): $\vec{d} = \frac{1}{|D|}\sum_{w \in D} \vec{v}_w$.
* **Libraries Used:** NumPy and scikit-learn (`cosine_similarity`).
* **Expected Output:** GloVe text format parser, word vector inspection, geometric word analogies, and document-level centroid similarity matrix.

### 7. Word Mover's Distance (WMD) (`unit-2/07_word_movers_distance.py`)
* **Purpose:** Measure document dissimilarity across semantic embedding space without requiring exact word matches.
* **Main Concept:** Formulates Earth Mover's Distance as an optimal transport linear program: $\min \sum T_{ij} c(i, j)$.
* **Libraries Used:** SciPy (`scipy.optimize.linprog`) and scikit-learn (`CountVectorizer`).
* **Expected Output:** BoW baseline showing failure on paraphrases with zero word overlap, WMD optimal transport distances, and word-to-word alignment flows.

### 8. Text Classification with Naive Bayes & SVM (`unit-2/08_text_classification_nb_svm.py`)
* **Purpose:** Categorize unstructured text into predefined classes using supervised learning.
* **Main Concept:** Compares a generative probabilistic classifier (`MultinomialNB`) with a discriminative maximum-margin classifier (`LinearSVC`) over TF-IDF features.
* **Libraries Used:** scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LinearSVC`, `train_test_split`, `classification_report`, `confusion_matrix`).
* **Expected Output:** Train/test split metrics, classification reports (Precision, Recall, F1), confusion matrix, and predictions on unseen domain queries.

### 9. Sentiment Analysis using TextBlob & VADER (`unit-2/09_sentiment_analysis_textblob_vader.py`)
* **Purpose:** Categorize emotional polarity and subjectivity in natural language text.
* **Main Concept:** Compares TextBlob pattern lexicon with VADER rule-based sentiment reasoning (handles capitalization, punctuation emphasis, negations, and contrastive conjunctions).
* **Libraries Used:** `textblob.TextBlob` and `nltk.sentiment.vader.SentimentIntensityAnalyzer`.
* **Expected Output:** Per-sentence polarity and subjectivity scores, VADER compound metrics, comparative summary table, and heuristic rule demonstration.

### 10. Topic Modeling using LDA (`unit-2/10_topic_modeling_lda.py`)
* **Purpose:** Uncover latent thematic topics across a corpus using probabilistic Bayesian modeling.
* **Main Concept:** Latent Dirichlet Allocation models documents as mixtures of topics and topics as Dirichlet distributions over words.
* **Libraries Used:** scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`).
* **Expected Output:** Document-Term count matrix, top keywords per discovered topic, document-topic distribution table, and topic inference on unseen documents.

### 11. Topic Modeling using LSA (`unit-2/11_topic_modeling_lsa.py`)
* **Purpose:** Discover latent semantic concepts using linear algebraic matrix factorization.
* **Main Concept:** Latent Semantic Analysis decomposes the TF-IDF matrix using Truncated Singular Value Decomposition (SVD): $X \approx U_k \Sigma_k V_k^T$.
* **Libraries Used:** scikit-learn (`TfidfVectorizer`, `TruncatedSVD`, `cosine_similarity`).
* **Expected Output:** Singular values, explained variance ratio per concept, top positive contributing terms, document concept coordinates, and semantic concept search.

### 12. Opinion Mining on Product Reviews Dataset (`unit-2/12_opinion_mining_reviews.py`)
* **Purpose:** Perform Aspect-Based Sentiment Analysis (ABSA) on real customer product reviews.
* **Main Concept:** Pairs specific product features/aspects (e.g., camera, battery, display) with descriptive opinion modifiers using POS tagging and syntactic rules, scoring each aspect individually.
* **Libraries Used:** NLTK (`pos_tag`, `word_tokenize`, `sent_tokenize`, `RegexpParser`, `SentimentIntensityAnalyzer`).
* **Expected Output:** Review-by-review aspect-opinion extraction, polarity classification, aggregated product feature scorecard, and business recommendations.

### 13. Information Extraction (IE) from Documents (`unit-2/13_information_extraction.py`)
* **Purpose:** Extract structured intelligence (entities, attributes, and semantic relations) from unstructured text.
* **Main Concept:** Combines regular expression pattern extractors (emails, phones, money, dates) with named entity recognition and semantic relation triple extraction `(Subject, Predicate, Object)`.
* **Libraries Used:** Pure Python (`re`, `json`) and NLTK (`sent_tokenize`).
* **Expected Output:** Extracted attribute listings, named entities by category, relation triple table, and synthesized JSON knowledge schema.

### 14. Information Retrieval System with TF-IDF Ranking (`unit-2/14_information_retrieval_tfidf.py`)
* **Purpose:** Build a full-text search engine that retrieves and ranks relevant documents in response to user queries.
* **Main Concept:** Employs the Salton Vector Space Model (VSM). Projects documents and queries into TF-IDF vector space, ranking hits via cosine similarity and generating contextual snippets.
* **Libraries Used:** scikit-learn (`TfidfVectorizer`, `cosine_similarity`) and NumPy.
* **Expected Output:** Corpus indexing statistics, ranked search results for multiple user queries with relevance scores, keyword-highlighted snippets, and vector query inspection.

---

## Learning Outcomes

Upon completing the experiments in these units, students will be able to:
1. Formulate and implement standard text preprocessing pipelines for real-world NLP applications.
2. Select appropriate tokenization algorithms and evaluate token segmentation boundaries.
3. Compare and contrast the trade-offs between stemming (speed, heuristic) and lemmatization (morphological validity).
4. Perform feature reduction using customized and standard stopword filtering.
5. Apply POS taggers to assign grammatical categories for downstream syntactic tasks.
6. Design regular expression chunk grammars and inspect syntactic dependency relations.
7. Implement Named Entity Recognition pipelines to extract structured domain intelligence from text.
8. Construct Bag of Words (BoW) representations from first principles and with scikit-learn.
9. Formulate, calculate, and interpret TF-IDF matrices and extract distinctive document keywords.
10. Generate and analyze word and character N-grams, compute language model conditional probabilities, and vectorize text using n-gram feature ranges.
11. Compute and interpret cosine similarity and cosine distance between documents from first principles and using scikit-learn.
12. Train and evaluate distributed word embedding models using Word2Vec (CBOW and Skip-gram) and perform vector arithmetic for analogies.
13. Parse pre-trained GloVe vector embeddings and construct document-level centroid vectors for semantic comparison.
14. Compute Word Mover's Distance (WMD) using optimal transport linear programming to measure text similarity without lexical overlap.
15. Design end-to-end supervised text classification pipelines comparing Multinomial Naive Bayes and Linear Support Vector Machines.
16. Perform sentiment analysis on multi-faceted text using TextBlob polarity/subjectivity and NLTK VADER grammatical heuristic rules.
17. Apply Latent Dirichlet Allocation (LDA) to discover probabilistic document-topic and topic-word distributions.
18. Factorize high-dimensional document spaces using Latent Semantic Analysis (LSA / Truncated SVD) and perform concept-space querying.
19. Implement aspect-based opinion mining to extract product feature-opinion pairs and compile aggregated product scorecards.
20. Build information extraction pipelines to extract structured attributes, named entities, and relational knowledge triples into JSON.
21. Architect full-text Information Retrieval (IR) search systems with inverted TF-IDF indexing, cosine ranking, and result snippet generation.

---

## Author

**Author:** [Your Name]

---

## Academic Information

* **Course:** Natural Language Processing
* **Units:** Unit 1 & Unit 2
* **COs:** CO1, CO2
* **Institution:** [College/University Name]
