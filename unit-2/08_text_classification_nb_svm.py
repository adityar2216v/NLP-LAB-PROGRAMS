"""
================================================================================
Program 08: Text Classification using Naive Bayes/SVM with TF-IDF
================================================================================

What is Text Classification?
----------------------------
Text classification is a core supervised Natural Language Processing (NLP) task 
where predefined category labels are algorithmically assigned to unstructured 
text documents based on their linguistic and statistical features.

Algorithms Implemented:
-----------------------
1. Multinomial Naive Bayes (MultinomialNB):
   - A generative probabilistic model based on Bayes' Theorem:
     P(Category | Document) ∝ P(Category) * ∏ P(word_i | Category)
   - Applies Laplace (add-1) smoothing to prevent zero probabilities.
   - Extremely fast, highly effective baseline for high-dimensional sparse text.

2. Support Vector Machine (LinearSVC):
   - A discriminative maximum-margin classifier.
   - Identifies the optimal separating hyperplane that maximizes the margin 
     (distance) between distinct document classes in the TF-IDF feature space.
   - Highly robust against the curse of dimensionality and overfitting in text.

Evaluation Metrics:
-------------------
- Accuracy: Overall percentage of correctly classified instances.
- Precision: Fraction of positive predictions that are truly positive (TP / (TP + FP)).
- Recall: Fraction of actual positive instances successfully retrieved (TP / (TP + FN)).
- F1-Score: Harmonic mean of precision and recall (2 * P * R / (P + R)).
- Confusion Matrix: Contingency table of true vs predicted classes.

Libraries Used:
- scikit-learn: TfidfVectorizer, MultinomialNB, LinearSVC, classification_report, confusion_matrix.
"""

import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------------------------
# 1. Academic Multi-Class Corpus (4 Domains: Tech, Sports, Medicine, Finance)
# ------------------------------------------------------------------------------
DATASET = [
    # Technology
    ("Python and Java are popular programming languages for building software", "Technology"),
    ("Neural networks and deep learning models accelerate artificial intelligence", "Technology"),
    ("Cloud servers and microservices provide scalable backend computing infrastructure", "Technology"),
    ("Modern graphic processing units train massive transformer language models", "Technology"),
    ("Cybersecurity specialists patch critical software vulnerabilities in operating systems", "Technology"),

    # Sports
    ("The soccer team scored a goal in extra time to win the championship", "Sports"),
    ("The tennis player served multiple aces to secure the grand slam title", "Sports"),
    ("Basketball players dribbled down the court and made a spectacular slam dunk", "Sports"),
    ("Athletes participate in marathon training to improve physical endurance", "Sports"),
    ("The cricket batsman scored a century in the decisive test match", "Sports"),

    # Medicine
    ("Doctors prescribed antibiotics and antivirals to treat the patient bacterial infection", "Medicine"),
    ("Clinical trials evaluated the efficacy of the novel oncology vaccine treatment", "Medicine"),
    ("Cardiologists monitor patient blood pressure and cardiovascular heart health", "Medicine"),
    ("Neurological studies analyze brain imaging scans for cognitive disorders", "Medicine"),
    ("Healthcare professionals recommend balanced nutrition and regular clinical checkups", "Medicine"),

    # Finance
    ("Central banks increased interest rates to combat rising consumer price inflation", "Finance"),
    ("Wall Street investors traded equities, stocks, and treasury bonds during market hours", "Finance"),
    ("Venture capital firms invested millions of dollars in financial technology startups", "Finance"),
    ("Quarterly corporate earnings reports showed robust revenue growth and profit margins", "Finance"),
    ("Commercial banks provide mortgage loans and credit facilities to businesses", "Finance")
]


def main():
    print("=" * 75)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 08: Text Classification using Naive Bayes and SVM with TF-IDF")
    print("=" * 75)

    texts, labels = zip(*DATASET)
    classes = sorted(list(set(labels)))

    print(f"\nCorpus Summary:")
    print(f"  Total Documents: {len(texts)}")
    print(f"  Target Classes ({len(classes)}): {classes}")

    # 1. Stratified Train / Test Split (75% Train, 25% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    print(f"  Training Set Size: {len(X_train_raw)} documents")
    print(f"  Testing Set Size:  {len(X_test_raw)} documents")

    # 2. Feature Extraction via TF-IDF Vectorizer
    print("\n" + "=" * 75)
    print(" 1. TF-IDF FEATURE EXTRACTION")
    print("=" * 75)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"  Learned Feature Vocabulary Size: {len(vectorizer.get_feature_names_out())} tokens")

    # 3. Model 1: Multinomial Naive Bayes
    print("\n" + "=" * 75)
    print(" 2. MULTINOMIAL NAIVE BAYES (MultinomialNB) CLASSIFIER")
    print("=" * 75)

    nb_model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"  Test Accuracy: {acc_nb * 100:.2f}%\n")
    print("  Classification Report (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb, zero_division=0))

    # 4. Model 2: Support Vector Machine (LinearSVC)
    print("=" * 75)
    print(" 3. SUPPORT VECTOR MACHINE (LinearSVC) CLASSIFIER")
    print("=" * 75)

    svm_model = LinearSVC(C=1.0, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"  Test Accuracy: {acc_svm * 100:.2f}%\n")
    print("  Classification Report (Linear SVM):")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    # Confusion Matrix for SVM
    print("  SVM Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_svm, labels=classes)
    header = f"  {'Actual \\ Pred':<15} | " + " | ".join(f"{c[:6]:>6}" for c in classes)
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, row_name in enumerate(classes):
        row_str = f"  {row_name:<15} | " + " | ".join(f"{cm[idx][j]:>6}" for j in range(len(classes)))
        print(row_str)
    print("  " + "-" * (len(header) - 2))

    # 5. Out-of-Sample Predictions on Unseen Queries
    print("\n" + "=" * 75)
    print(" 4. PREDICTIONS ON UNSEEN TEST SENTENCES")
    print("=" * 75)

    unseen_samples = [
        "Researchers engineered deep learning neural networks on python for data processing.",
        "The athlete scored a winning touchdown during the football championship tournament.",
        "The hospital physician administered medications to alleviate clinical fever symptoms.",
        "Stock market shares and interest rates surged following foreign investments."
    ]

    unseen_tfidf = vectorizer.transform(unseen_samples)
    nb_predictions = nb_model.predict(unseen_tfidf)
    svm_predictions = svm_model.predict(unseen_tfidf)

    for i, (query, nb_pred, svm_pred) in enumerate(zip(unseen_samples, nb_predictions, svm_predictions), start=1):
        print(f"\nQuery {i}: \"{query}\"")
        print(f"  -> Naive Bayes Predicted Class: {nb_pred}")
        print(f"  -> Linear SVM  Predicted Class: {svm_pred}")

    print("\n" + "=" * 75)
    print(" Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
