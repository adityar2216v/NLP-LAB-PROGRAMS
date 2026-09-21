# NLP Lab Programs

## Description

This repository contains the complete **Unit 1 Natural Language Processing (NLP)** laboratory practical programs. It covers foundational text preprocessing and linguistic analysis techniques essential for undergraduate and postgraduate computer science curricula. The codebase is implemented in Python using industry-standard NLP libraries (**NLTK** and **spaCy**), formatted cleanly for academic lab evaluations, practical examinations, and viva voce preparation.

---

## Objectives

* Understand foundational NLP text preprocessing concepts and pipelines.
* Perform sentence and word tokenization using both classical and modern tokenizers.
* Understand the conceptual and practical differences between stemming and lemmatization.
* Remove non-informative stopwords from a text document.
* Perform Part-of-Speech (POS) tagging on sentences to identify grammatical categories.
* Implement shallow parsing (chunking) via regular expression grammars and dependency parsing via spaCy.
* Perform Named Entity Recognition (NER) to extract real-world entities (PERSON, ORG, GPE, DATE, MONEY).
* Gain hands-on practical experience using **NLTK** and **spaCy** APIs.

---

## Technologies Used

* **Python 3.x**
* **NLTK (Natural Language Toolkit)**
* **spaCy (Industrial-Strength NLP)**
* **scikit-learn (Machine Learning & Feature Extraction)**
* **NumPy (Numerical Computing)**

---

## Unit 1 Programs

| S. No. | Program | File | CO |
| :---: | :--- | :--- | :---: |
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | `Unit-1/01_tokenization.py` | CO1 |
| 2 | Stemming and Lemmatization on Sample Text | `Unit-1/02_stemming_lemmatization.py` | CO1 |
| 3 | Stop-word Removal from a Document | `Unit-1/03_stopword_removal.py` | CO1 |
| 4 | Part-of-Speech (POS) Tagging of a Given Sentence | `Unit-1/04_pos_tagging.py` | CO1 |
| 5 | Parsing and Chunking using RegEx and spaCy | `Unit-1/05_parsing_chunking.py` | CO1 |
| 6 | Named Entity Recognition (NER) using spaCy | `Unit-1/06_named_entity_recognition.py` | CO1 |

---

## Unit 2 Programs

| S. No. | Program | File | CO |
| :---: | :--- | :--- | :---: |
| 1 | Bag of Words (BoW) Model | `unit-2/01_bag_of_words.py` | CO2 |
| 2 | Term Frequency - Inverse Document Frequency (TF-IDF) | `unit-2/02_tfidf.py` | CO2 |
| 3 | N-gram Modeling and Feature Extraction | `unit-2/03_ngrams.py` | CO2 |

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
    └── 03_ngrams.py
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
The programs automatically download required resources at runtime if missing. Alternatively, you can pre-download all required NLTK datasets using:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 6. Download spaCy English Language Model
Download the small English pipeline (`en_core_web_sm`):
```bash
python -m spacy download en_core_web_sm
```

---

## How to Run

Execute each program individually from the repository root:

# Unit 1 Programs
python Unit-1/01_tokenization.py
python Unit-1/02_stemming_lemmatization.py
python Unit-1/03_stopword_removal.py
python Unit-1/04_pos_tagging.py
python Unit-1/05_parsing_chunking.py
python Unit-1/06_named_entity_recognition.py

# Unit 2 Programs
python unit-2/01_bag_of_words.py
python unit-2/02_tfidf.py
python unit-2/03_ngrams.py
```

---

## Program Details

### 1. Tokenization of Sentences and Words using NLTK and spaCy (`01_tokenization.py`)
* **Purpose:** Break unstructured text into manageable linguistic segments.
* **Main Concept:** Sentence tokenization detects sentence boundaries (`.`, `!`, `?`), while word tokenization segments words and punctuation.
* **Libraries Used:** NLTK (`sent_tokenize`, `word_tokenize`) and spaCy (`nlp()`, `doc.sents`, `token.text`).
* **Short Explanation:** Compares the tokenization output between NLTK rule-based tokenizers and spaCy neural pipeline tokenizers on a sample paragraph.
* **Expected Output:** Numbered list of sentences followed by an array of individual word tokens for both NLTK and spaCy.

### 2. Stemming and Lemmatization on Sample Text (`02_stemming_lemmatization.py`)
* **Purpose:** Reduce inflected word variants to their base or root forms.
* **Main Concept:** Stemming uses heuristic suffix-stripping rules (may yield invalid non-words like `studi`). Lemmatization uses morphological analysis with WordNet dictionary lookup to yield valid dictionary lemmas (e.g., `study`, `child`).
* **Libraries Used:** NLTK (`PorterStemmer`, `WordNetLemmatizer`).
* **Short Explanation:** Iterates through a prescribed list of inflected words (`playing`, `played`, `plays`, `studies`, `studying`, `running`, `easily`, `children`) and computes both stem and noun/verb lemmas.
* **Expected Output:** A side-by-side comparative table showing Original Word, Porter Stemmer, Lemmatizer (Noun), and Lemmatizer (Verb), accompanied by viva analysis notes.

### 3. Stop-word Removal from a Document (`03_stopword_removal.py`)
* **Purpose:** Filter out high-frequency grammatical words carrying minimal semantic signal.
* **Main Concept:** Words such as *is*, *an*, *the*, and *in* are filtered out to reduce dimensionality and focus on content keywords.
* **Library Used:** NLTK (`nltk.corpus.stopwords`, `word_tokenize`).
* **Short Explanation:** Tokenizes a sample NLP paragraph, detects English stopwords using a case-insensitive check, removes them, and reassembles the cleaned text.
* **Expected Output:** Original text, list of raw tokens, list of identified stopwords removed, filtered tokens, cleaned text, and statistical reduction percentage.

### 4. Part-of-Speech (POS) Tagging of a Given Sentence (`04_pos_tagging.py`)
* **Purpose:** Assign grammatical categories to words in a sentence based on syntactic context.
* **Main Concept:** Disambiguates syntactic functions (e.g., whether *jumps* is a verb or noun) using the Penn Treebank tagset.
* **Library Used:** NLTK (`pos_tag`, `word_tokenize`).
* **Short Explanation:** Parses the sentence *"The quick brown fox jumps over the lazy dog."* and maps each token to standard tags (`DT`, `JJ`, `NN`, `VBZ`, `IN`).
* **Expected Output:** A structured 3-column table displaying Token, POS Tag, and human-readable grammatical meaning, followed by standard tuple output.

### 5. Parsing and Chunking using RegEx and spaCy (`05_parsing_chunking.py`)
* **Purpose:** Group words into syntactically correlated phrases and extract dependency relations without GUI dependencies.
* **Main Concept:** 
  1. *Chunking (Shallow Parsing):* Identifies non-overlapping phrases like Noun Phrases (`NP`) using regular expressions over POS tags.
  2. *Parsing (Deep Syntactic Analysis):* Analyzes full hierarchical head-modifier dependencies.
* **Libraries Used:** NLTK (`RegexpParser`, `pos_tag`) and spaCy (`doc.noun_chunks`, token dependency tags).
* **Short Explanation:** Defines a noun phrase grammar `NP: {<DT>?<JJ>*<NN.*>+}`, produces a text-based parse tree in NLTK, and extracts noun chunks and dependency relations using spaCy.
* **Expected Output:** Text-formatted parse tree, extracted NP chunks, and spaCy dependency parsing table (`Token`, `POS`, `Dependency Relation`, `Head Token`).

### 6. Named Entity Recognition (NER) using spaCy (`06_named_entity_recognition.py`)
* **Purpose:** Identify and classify named entities into real-world categories.
* **Main Concept:** Detects and classifies entities such as PERSON, ORG, GPE, DATE, and MONEY from unstructured text.
* **Library Used:** spaCy (`en_core_web_sm`).
* **Short Explanation:** Analyzes an industry news sentence containing diverse entity classes, displays entity labels with official descriptions (`spacy.explain`), and groups entities by category. Includes friendly error handling if the model is missing.
* **Expected Output:** Original sentence, tabular list of detected entities with categories and explanations, and a category-grouped summary.

---

## Unit 2 Program Summaries

### 1. Bag of Words (BoW) Model (`unit-2/01_bag_of_words.py`)
* **Purpose:** Represent documents as numerical word occurrence vectors across a defined vocabulary.
* **Main Concept:** Extracts unique words across a corpus, creates vocabulary indices, and constructs count (term frequency) and binary (presence/absence) vectors while ignoring word order and grammar.
* **Libraries Used:** Pure Python (from scratch) and scikit-learn (`CountVectorizer`, `cosine_similarity`).
* **Short Explanation:** Implements BoW from first principles using standard Python data structures and compares against scikit-learn's `CountVectorizer`. Demonstrates stop words removal and pairwise document similarity computation.
* **Expected Output:** Extracted vocabulary list, document-term count matrix, binary matrix, scikit-learn feature index mapping, cosine similarity table, and stopword-filtered count matrix.

### 2. Term Frequency - Inverse Document Frequency (`unit-2/02_tfidf.py`)
* **Purpose:** Evaluate the relative importance of words within documents relative to the entire corpus.
* **Main Concept:** Offsets high-frequency words by multiplying Term Frequency ($TF$) by Inverse Document Frequency ($IDF$), followed by $L_2$ Euclidean normalization.
* **Libraries Used:** Pure Python / NumPy (mathematical formulation from scratch) and scikit-learn (`TfidfVectorizer`, `cosine_similarity`).
* **Short Explanation:** Performs manual step-by-step mathematical calculations of normalized TF, Document Frequency (DF), smooth IDF values, raw TF-IDF, and $L_2$-normalized vectors. Validates against scikit-learn and extracts top keywords per document.
* **Expected Output:** Vocabulary listing, step-by-step TF table, DF and smooth IDF values, raw TF-IDF matrix, unit-norm normalized matrix, scikit-learn comparison matrix, top salient keywords per document, and pairwise cosine similarity.

### 3. N-gram Modeling and Feature Extraction (`unit-2/03_ngrams.py`)
* **Purpose:** Capture local sequential context and multi-word phrases beyond single unigrams.
* **Main Concept:** Extracts contiguous sequences of $n$ items (words or characters) to capture collocations, phrase semantics, and transition probabilities.
* **Libraries Used:** Pure Python (list slicing & zip), NLTK (`ngrams`, `FreqDist`), and scikit-learn (`CountVectorizer(ngram_range=...)`).
* **Short Explanation:** Generates unigrams, bigrams, trigrams, and character n-grams from scratch. Uses NLTK to identify top collocations and calculate conditional transition probabilities $P(w_n | w_{n-1})$. Demonstrates bigram and combined unigram/bigram feature extraction using scikit-learn.
* **Expected Output:** Lists of word unigrams, bigrams, and trigrams; character trigrams; most frequent bigrams with counts; conditional next-word probabilities; and scikit-learn n-gram feature matrices.

---

## Learning Outcomes

Upon completing the experiments in these units, students will be able to:
1. Formulate and implement standard text preprocessing pipelines for real-world NLP applications.
2. Select appropriate tokenization algorithms and evaluate token segmentation boundaries.
3. Compare and contrast the trade-offs between stemming (speed, heuristic) and lemmatization (morphological validity).
4. Perform feature reduction using customized and standard stopword filtering.
5. Apply POS taggers to assign grammatical categories for downstream syntactic tasks.
6. Design regular expression chunk grammars and inspect syntactic dependency relations.
7. Implement Named Entity Recognition pipelines to extract structured business and domain intelligence from text.
8. Construct Bag of Words (BoW) representations from first principles and with scikit-learn.
9. Formulate, calculate, and interpret TF-IDF matrices and extract distinctive document keywords.
10. Generate and analyze word and character N-grams, compute language model conditional probabilities, and vectorize text using n-gram feature ranges.

---

## Author

**Author:** [Your Name]

---

## Academic Information

* **Course:** Natural Language Processing
* **Units:** Unit 1 & Unit 2
* **COs:** CO1, CO2
* **Institution:** [College/University Name]
