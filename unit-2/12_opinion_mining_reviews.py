"""
================================================================================
Program 12: Opinion Mining on Product/Service Reviews Dataset
================================================================================

What is Opinion Mining (Aspect-Based Sentiment Analysis - ABSA)?
----------------------------------------------------------------
While standard sentiment analysis assigns a single overall polarity score to a 
document, real-world customer reviews are nuanced and multi-faceted. A user may 
praise one feature while strongly criticizing another:
    "The camera produces stunning photos, but the battery life is atrocious."

Opinion mining (ABSA) addresses this by decomposing reviews into:
1. Aspect / Target Feature: The specific component, attribute, or property 
   being discussed (e.g., 'camera', 'battery life', 'screen', 'price').
2. Opinion Word / Modifier: The subjective descriptive words qualifying the 
   aspect (e.g., 'stunning', 'atrocious', 'crisp', 'overpriced').
3. Aspect Polarity: The sentiment orientation (Positive, Negative, Neutral) 
   specifically directed toward that feature.
4. Aggregated Product Scorecard: Quantitative rollup summarizing customer 
   satisfaction across every product dimension.

Linguistic Pipeline:
--------------------
1. Tokenization & POS Tagging: Identifies nouns (candidate aspects) and adjectives/adverbs (candidate opinions).
2. Syntactic Chunking & Collocation: Extracts multi-word noun phrase aspects (e.g., "battery life", "customer support").
3. Opinion-Aspect Association: Connects adjectives to their corresponding noun targets.
4. Polarity Scoring: Evaluates the valence of the opinion words using sentiment intensity analysis.
5. Scorecard Aggregation: Compiles mention frequencies and net positive sentiment ratios.

Libraries Used:
- NLTK: pos_tag, word_tokenize, RegexpParser, SentimentIntensityAnalyzer.
- textblob: (Optional) Polarity validation.
"""

import sys
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ------------------------------------------------------------------------------
# 1. Aspect-Opinion Extraction Engine
# ------------------------------------------------------------------------------
class OpinionMiningEngine:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Grammar to detect noun phrases (aspects) and adjectival phrases (opinions)
        self.grammar = r"""
            ASPECT: {<JJ|JJR|JJS>*<NN|NNS>+}
            OPINION: {<RB|RBR|RBS>*<JJ|JJR|JJS>+}
        """
        self.chunker = RegexpParser(self.grammar)

        # Standard canonical aspect mappings for smartphones
        self.aspect_categories = {
            "battery": ["battery", "battery life", "charge", "charging"],
            "camera": ["camera", "photo", "photos", "picture", "pictures", "lens", "video"],
            "display": ["screen", "display", "oled", "screen resolution", "brightness"],
            "performance": ["speed", "processor", "performance", "gaming", "lag", "ram"],
            "audio": ["speaker", "speakers", "sound", "audio", "mic", "call quality"],
            "price": ["price", "cost", "value", "money", "expensive", "deal"],
            "build": ["build", "design", "body", "durability", "weight", "finish"]
        }

    def _normalize_aspect(self, candidate: str) -> str:
        cand_lower = candidate.lower().strip()
        for category, keywords in self.aspect_categories.items():
            for kw in keywords:
                if kw in cand_lower:
                    return category.capitalize()
        return cand_lower.capitalize()

    def analyze_review(self, text: str) -> list[dict]:
        """Extracts aspect-opinion-sentiment triples from a review text."""
        extracted_tuples = []
        sentences = sent_tokenize(text)

        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)

            # Look for explicit linguistic patterns: [Aspect] is [Opinion] or [Opinion] [Aspect]
            # 1. Attributive pattern: "amazing camera" (JJ + NN)
            for i in range(len(tagged) - 1):
                word1, tag1 = tagged[i]
                word2, tag2 = tagged[i + 1]

                if tag1.startswith("JJ") and tag2.startswith("NN"):
                    opinion = word1
                    aspect = word2
                    score = self.vader.polarity_scores(opinion)["compound"]
                    if abs(score) >= 0.05:
                        extracted_tuples.append({
                            "sentence": sent,
                            "aspect": self._normalize_aspect(aspect),
                            "raw_aspect": aspect,
                            "opinion": opinion,
                            "score": score,
                            "polarity": "Positive" if score > 0 else "Negative"
                        })

            # 2. Predicative pattern: "camera is amazing" (NN + BE_VERB + JJ)
            for i in range(len(tagged) - 2):
                w1, t1 = tagged[i]
                w2, t2 = tagged[i + 1]
                w3, t3 = tagged[i + 2]

                if t1.startswith("NN") and w2.lower() in ["is", "was", "are", "were", "looks", "feels"] and t3.startswith("JJ"):
                    aspect = w1
                    opinion = w3
                    score = self.vader.polarity_scores(opinion)["compound"]
                    if abs(score) >= 0.05:
                        extracted_tuples.append({
                            "sentence": sent,
                            "aspect": self._normalize_aspect(aspect),
                            "raw_aspect": aspect,
                            "opinion": opinion,
                            "score": score,
                            "polarity": "Positive" if score > 0 else "Negative"
                        })

        return extracted_tuples


# ------------------------------------------------------------------------------
# 2. Main Execution Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" NATURAL LANGUAGE PROCESSING LAB - UNIT 2")
    print(" Program 12: Opinion Mining on Product/Service Reviews Dataset")
    print("=" * 80)

    reviews_dataset = [
        "The smartphone has a stunning display and wonderful camera, but the battery is terrible.",
        "Audio quality is fantastic and clear, though the build is quite fragile.",
        "The battery life is exceptional and lasts two days, but the price is expensive.",
        "Incredible performance with fast processor, but the screen is dim under sunlight.",
        "Customer service was helpful and responsive during my return process.",
        "The camera captures beautiful photos with crisp detail, absolutely excellent device!",
        "Poor battery life ruins an otherwise decent phone with good performance."
    ]

    print(f"\nAnalyzing Product Reviews Dataset ({len(reviews_dataset)} customer reviews)...")

    engine = OpinionMiningEngine()
    aspect_stats = defaultdict(lambda: {"pos": 0, "neg": 0, "scores": []})

    print("\n" + "=" * 80)
    print(" 1. ASPECT-OPINION-POLARITY EXTRACTION PER REVIEW")
    print("=" * 80)

    for idx, review in enumerate(reviews_dataset, start=1):
        print(f"\n[Review #{idx}] \"{review}\"")
        results = engine.analyze_review(review)

        if not results:
            print("  No explicit aspect-opinion pairs detected.")
            continue

        for r in results:
            aspect = r["aspect"]
            pol = r["polarity"]
            sc = r["score"]
            op = r["opinion"]

            print(f"  -> Aspect: {aspect:<12} | Opinion: '{op}' | Sentiment: {pol:<8} (score: {sc:+.2f})")

            if pol == "Positive":
                aspect_stats[aspect]["pos"] += 1
            else:
                aspect_stats[aspect]["neg"] += 1
            aspect_stats[aspect]["scores"].append(sc)

    # 2. Aggregated Product Feature Scorecard
    print("\n" + "=" * 80)
    print(" 2. AGGREGATED PRODUCT FEATURE SCORECARD")
    print("=" * 80)

    header = f"{'Feature / Aspect':<18} | {'Total Mentions':<14} | {'Positive':<10} | {'Negative':<10} | {'Avg Score':<10} | {'Satisfaction Rate'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Sort aspects by total mentions
    sorted_aspects = sorted(aspect_stats.items(), key=lambda x: len(x[1]["scores"]), reverse=True)

    for aspect, data in sorted_aspects:
        total = len(data["scores"])
        pos = data["pos"]
        neg = data["neg"]
        avg_score = sum(data["scores"]) / total if total > 0 else 0.0
        pos_ratio = (pos / total) * 100 if total > 0 else 0.0

        bar = "#" * int(pos_ratio / 10) + "-" * (10 - int(pos_ratio / 10))
        print(f"{aspect:<18} | {total:<14} | {pos:<10} | {neg:<10} | {avg_score:>+9.2f}  | {pos_ratio:>5.1f}% [{bar}]")
    print("-" * len(header))

    print("\n[Actionable Business Insights]")
    top_praised = [a for a, d in sorted_aspects if d["pos"] > d["neg"]]
    top_critiqued = [a for a, d in sorted_aspects if d["neg"] >= d["pos"]]
    print(f"  Strengths to Highlight in Marketing: {', '.join(top_praised)}")
    print(f"  Weaknesses Requiring Engineering Fix: {', '.join(top_critiqued)}")

    print("\n" + "=" * 80)
    print(" Execution Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
