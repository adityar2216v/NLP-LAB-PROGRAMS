"""
================================================================================
Program 09: Sentiment Analysis using TextBlob and VADER
================================================================================

What is Sentiment Analysis?
----------------------------
Sentiment Analysis (also known as opinion mining) is the NLP task of computationally 
identifying and categorizing emotional attitudes, polarities, and subjective 
opinions expressed within textual data.

Approaches Compared in this Program:
------------------------------------
1. TextBlob Sentiment Engine:
   - Uses a pattern-matching lexicon scored by semantic orientation and intensity.
   - Provides two continuous output dimensions:
     * Polarity: Continuous float in [-1.0, 1.0], where:
       - -1.0 represents highly negative sentiment.
       -  0.0 represents neutral sentiment.
       - +1.0 represents highly positive sentiment.
     * Subjectivity: Continuous float in [0.0, 1.0], where:
       -  0.0 represents completely factual/objective text.
       -  1.0 represents personal opinion/subjective text.

2. VADER (Valence Aware Dictionary and sEntiment Reasoner):
   - A specialized rule-based sentiment analysis tool optimized for social media, 
     customer reviews, and informal conversational text (Hutto & Gilbert, 2014).
   - Incorporates grammatical and syntactic heuristic rules:
     * Punctuation emphasis ("great!!!" vs "great.")
     * Capitalization ("GREAT" vs "great")
     * Degree modifiers ("exceptionally good" vs "marginally good")
     * Contrastive conjunctions ("The food was great, BUT the service was awful")
     * Negation handling ("not terrible", "never good")
     * Slang and emoticons (":)", ":(", "meh")
   - Output metric:
     * compound: Normalized, weighted composite score in [-1.0, 1.0].
       - Positive: compound >= +0.05
       - Neutral:  -0.05 < compound < +0.05
       - Negative: compound <= -0.05
     * pos, neu, neg: Proportions of text falling into each category (sum to 1.0).

Libraries Used:
- textblob: TextBlob sentiment analyzer.
- NLTK: nltk.sentiment.vader.SentimentIntensityAnalyzer.
"""

import sys
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def classify_polarity(val: float, threshold: float = 0.05) -> str:
    if val >= threshold:
        return "Positive"
    elif val <= -threshold:
        return "Negative"
    else:
        return "Neutral"


def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 09: Sentiment Analysis using TextBlob and VADER")
    print("=" * 80)

    # Initialize VADER Analyzer
    vader = SentimentIntensityAnalyzer()

    # Benchmark test sentences covering diverse linguistic nuances
    test_cases = [
        "The movie was an absolute masterpiece with brilliant acting and stunning visuals!",
        "The customer service was terribly slow and the staff was extremely rude.",
        "The package arrived on Tuesday afternoon via postal delivery.",
        "The smartphone has an amazing screen, but the battery life is horribly disappointing.",
        "The meal was not bad at all, in fact I quite enjoyed it.",
        "I LOVED the concert, it was totally AMAZING!!!",
        "The software update is just okay, nothing particularly impressive or bad."
    ]

    print("\n" + "=" * 80)
    print(" 1. DETAILED STEP-BY-STEP SENTENCE EVALUATION")
    print("=" * 80)

    for i, sentence in enumerate(test_cases, start=1):
        print(f"\n[Case {i}] \"{sentence}\"")

        # TextBlob Analysis
        tb_blob = TextBlob(sentence)
        tb_polarity = tb_blob.sentiment.polarity
        tb_subjectivity = tb_blob.sentiment.subjectivity
        tb_label = classify_polarity(tb_polarity)

        # VADER Analysis
        vader_scores = vader.polarity_scores(sentence)
        vader_compound = vader_scores["compound"]
        vader_label = classify_polarity(vader_compound)

        print(f"  [TextBlob]")
        print(f"    - Polarity Score    : {tb_polarity:+.4f} ({tb_label})")
        print(f"    - Subjectivity Score: {tb_subjectivity:.4f} (0=Objective, 1=Subjective)")
        print(f"  [VADER]")
        print(f"    - Compound Score    : {vader_compound:+.4f} ({vader_label})")
        print(f"    - Proportions       : Pos={vader_scores['pos']:.2f}, Neu={vader_scores['neu']:.2f}, Neg={vader_scores['neg']:.2f}")

    # Comparative Summary Table
    print("\n" + "=" * 80)
    print(" 2. SIDE-BY-SIDE COMPARATIVE SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Case':<5} | {'TextBlob Pol.':<14} | {'Subjectivity':<12} | {'VADER Compound':<14} | {'VADER Label'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, sentence in enumerate(test_cases, start=1):
        tb = TextBlob(sentence)
        vs = vader.polarity_scores(sentence)
        tb_pol = tb.sentiment.polarity
        tb_sub = tb.sentiment.subjectivity
        v_comp = vs["compound"]
        v_lab = classify_polarity(v_comp)

        print(f"#{i:<4} | {tb_pol:>+14.4f} | {tb_sub:>12.4f} | {v_comp:>+14.4f} | {v_lab}")
    print("-" * len(header))

    # Nuance Deep-Dive: Capitalization & Punctuation
    print("\n" + "=" * 80)
    print(" 3. VADER HEURISTIC RULES: EMPHASIS & NEGATION ANALYSIS")
    print("=" * 80)
    pairs = [
        ("Base sentence", "The pizza was good."),
        ("Exclamation emphasis", "The pizza was good!!!"),
        ("All-caps emphasis", "The pizza was GOOD!"),
        ("Contrastive 'but'", "The pizza was good, but the crust was burnt."),
        ("Negation idiom", "The pizza was not bad.")
    ]

    for label, text in pairs:
        score = vader.polarity_scores(text)["compound"]
        print(f"  {label:<24} | \"{text:<45}\" -> Compound: {score:+.4f}")

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
