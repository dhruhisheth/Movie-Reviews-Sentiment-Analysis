#!/usr/bin/env python3
"""
Sentiment Analysis for Movie Reviews
Analyzes sentiment in movie reviews using word frequency and sentiment lexicons.
"""

# ==================== IMPORTS ====================
import nltk
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from collections import Counter
from nltk.corpus import opinion_lexicon, stopwords
from typing import List, Tuple, Counter as CounterType

# ==================== INITIALIZATION ====================
def initialize_nltk() -> None:
    """Download required NLTK resources."""
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('opinion_lexicon')
    nltk.download('stopwords')

def setup_sentiment_lexicons() -> Tuple[set, set, set, set]:
    """Initialize sentiment lexicons and word sets."""
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    stop_words = set(stopwords.words('english'))
    
    # Add emphasized versions
    positive_words.update({"emph_" + w for w in positive_words})
    negative_words.update({"emph_" + w for w in negative_words})
    
    emphasis_words = {"absolutely", "really", "very", "extremely", "so"}
    
    return positive_words, negative_words, stop_words, emphasis_words

# ==================== FILE I/O FUNCTIONS ====================
def read_file(file_path: str) -> str:
    """Read content from a single file."""
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()

def read_directory(directory_path: str) -> List[str]:
    """Read all files from a directory and return their contents."""
    file_contents = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            file_contents.append(read_file(file_path))
    return file_contents

# ==================== TEXT PROCESSING FUNCTIONS ====================
def get_unique_words_frequencies(text: str, stop_words: set, emphasis_words: set) -> CounterType:
    """
    Extract word frequencies from text with preprocessing.
    Handles negation and emphasis.
    """
    words = nltk.word_tokenize(text)
    new_words = []
    negate = False
    prev_word = ""

    for word in words:
        word = word.lower()
        
        # Handle negation
        if word in ["not", "no", "n't"]:
            negate = True
            continue
        
        # Process alphabetic words not in stop words
        if word.isalpha() and word not in stop_words:
            if negate:
                word = "not_" + word
                negate = False
            if prev_word in emphasis_words:
                word = "emph_" + word
            new_words.append(word)
            prev_word = word

    return Counter(new_words)

def categorize_word_sentiments(filtered_words: CounterType, positive_words: set, negative_words: set) -> Tuple[int, int]:
    """
    Categorize words into positive and negative sentiment counts.
    Applies weighting for emphasis and strong sentiment words.
    """
    p_count, n_count = 0, 0
    
    for word, freq in filtered_words.items():
        if word in positive_words:
            if word.startswith("emph_"):
                p_count += freq * 2
            elif word in ["love", "excellent"]:
                p_count += freq * 1.5
            else:
                p_count += freq
        elif word in negative_words:
            if word.startswith("emph_"):
                n_count += freq * 2
            elif word in ["hate", "terrible"]:
                n_count += freq * 1.5
            else:
                n_count += freq
    
    return int(p_count), int(n_count)

# ==================== ANALYSIS FUNCTIONS ====================
def analyze_review_collection(reviews: List[str], stop_words: set, emphasis_words: set, 
                            positive_words: set, negative_words: set) -> Tuple[CounterType, int, int]:
    """Analyze a collection of reviews and return aggregated word frequencies and sentiment counts."""
    combined_frequencies = Counter()
    
    for review in reviews:
        frequencies = get_unique_words_frequencies(review, stop_words, emphasis_words)
        combined_frequencies += frequencies
    
    pos_score, neg_score = categorize_word_sentiments(combined_frequencies, positive_words, negative_words)
    return combined_frequencies, pos_score, neg_score

def individual_reviews_analysis(reviews: List[str], stop_words: set, emphasis_words: set, 
                              positive_words: set, negative_words: set) -> Tuple[List[int], List[int], float, float, float, float]:
    """Analyze individual reviews and return statistics."""
    pos_counts = []
    neg_counts = []

    for review in reviews:
        frequencies = get_unique_words_frequencies(review, stop_words, emphasis_words)
        p_count, n_count = categorize_word_sentiments(frequencies, positive_words, negative_words)
        pos_counts.append(p_count)
        neg_counts.append(n_count)

    # Calculate statistics
    avg_pos = np.mean(pos_counts)
    avg_neg = np.mean(neg_counts)
    std_pos = statistics.stdev(pos_counts) if len(pos_counts) > 1 else 0
    std_neg = statistics.stdev(neg_counts) if len(neg_counts) > 1 else 0

    return pos_counts, neg_counts, avg_pos, std_pos, avg_neg, std_neg

def classify_review(review: str, stop_words: set, emphasis_words: set, 
                   positive_words: set, negative_words: set) -> str:
    """Classify a single review as positive or negative."""
    freq = get_unique_words_frequencies(review, stop_words, emphasis_words)
    p_count, n_count = categorize_word_sentiments(freq, positive_words, negative_words)
    total_words = sum(freq.values())
    unique_count = len(freq)

    if total_words == 0:
        return "neg"
    if p_count == 0 and n_count == 0:
        return "neg"

    sentiment_ratio = (p_count - n_count) / (total_words + 1)

    # Adjust for review length
    if total_words < 20 and (p_count > 0 or n_count > 0):
        sentiment_ratio *= 1.5
    if unique_count > 100:
        sentiment_ratio *= 1.1

    return "pos" if sentiment_ratio >= 0 else "neg"

def test_classifier_accuracy(test_positive_files: List[str], test_negative_files: List[str], 
                           stop_words: set, emphasis_words: set, positive_words: set, negative_words: set) -> float:
    """Test classifier accuracy on test dataset."""
    correct_predictions = 0
    total_predictions = 0

    # Test positive reviews
    for review in test_positive_files:
        prediction = classify_review(review, stop_words, emphasis_words, positive_words, negative_words)
        if prediction == "pos":
            correct_predictions += 1
        total_predictions += 1

    # Test negative reviews
    for review in test_negative_files:
        prediction = classify_review(review, stop_words, emphasis_words, positive_words, negative_words)
        if prediction == "neg":
            correct_predictions += 1
        total_predictions += 1

    return correct_predictions / total_predictions

# ==================== VISUALIZATION FUNCTIONS ====================
def create_sentiment_bar_chart(positive_files_positive_score: int, positive_files_negative_score: int,
                              negative_files_positive_score: int, negative_files_negative_score: int,
                              std_pos: float, std_neg: float) -> None:
    """Create bar chart showing sentiment word counts."""
    categories = [
        "Positive Words in Positive Reviews",
        "Negative Words in Positive Reviews",
        "Positive Words in Negative Reviews",
        "Negative Words in Negative Reviews"
    ]
    values = [
        positive_files_positive_score,
        positive_files_negative_score,
        negative_files_positive_score,
        negative_files_negative_score
    ]
    errors = [std_pos, std_neg, std_pos, std_neg]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, yerr=errors, capsize=15, color=['blue', 'red', 'green', 'purple'])
    plt.xlabel("Category")
    plt.ylabel("Word Count")
    plt.title("Sentiment Word Counts in Reviews with Error Bars")
    plt.show()

def create_frequent_words_charts(positive_frequencies: CounterType, negative_frequencies: CounterType,
                               positive_words: set, negative_words: set) -> None:
    """Create charts showing most frequent positive and negative words."""
    # Filter for sentiment words only
    filtered_positive_counter = Counter({
        word: freq for word, freq in positive_frequencies.items() if word in positive_words
    })
    filtered_negative_counter = Counter({
        word: freq for word, freq in negative_frequencies.items() if word in negative_words
    })

    # Get top 4 most frequent words
    frequent_positive_words = filtered_positive_counter.most_common(4)
    frequent_negative_words = filtered_negative_counter.most_common(4)

    # Create positive words chart
    if frequent_positive_words:
        pos_words, pos_freqs = zip(*frequent_positive_words)
        plt.figure(figsize=(10, 6))
        plt.bar(pos_words, pos_freqs, color='blue')
        plt.xlabel("Positive Words")
        plt.ylabel("Frequency")
        plt.title("Most Frequent Positive Words in Reviews")
        plt.show()

    # Create negative words chart
    if frequent_negative_words:
        neg_words, neg_freqs = zip(*frequent_negative_words)
        plt.figure(figsize=(10, 6))
        plt.bar(neg_words, neg_freqs, color='red')
        plt.xlabel("Negative Words")
        plt.ylabel("Frequency")
        plt.title("Most Frequent Negative Words in Reviews")
        plt.show()

# ==================== REPORTING FUNCTIONS ====================
def print_dataset_overview(positive_files: List[str], negative_files: List[str]) -> None:
    """Print overview of the dataset."""
    print(f"Loaded {len(positive_files)} positive reviews and {len(negative_files)} negative reviews.")
    if not positive_files and not negative_files:
        print("No reviews found. Check your file paths or data availability.")
        exit()

def print_sentiment_scores(positive_files_positive_score: int, positive_files_negative_score: int,
                         negative_files_positive_score: int, negative_files_negative_score: int) -> None:
    """Print sentiment scores for different review categories."""
    print(f"Total Positive Words in Negative Reviews: {negative_files_positive_score}")
    print(f"Total Negative Words in Negative Reviews: {negative_files_negative_score}")
    print(f"Total Positive Words in Positive Reviews: {positive_files_positive_score}")
    print(f"Total Negative Words in Positive Reviews: {positive_files_negative_score}")

def print_individual_review_stats(pos_counts: List[int], neg_counts: List[int]) -> None:
    """Print statistics for individual reviews."""
    print("\nWord Counts per Review:")
    for i in range(len(pos_counts)):
        print(f"Review {i+1}: Positive Words = {pos_counts[i]}, Negative Words = {neg_counts[i]}")

def print_summary_statistics(avg_pos: float, std_pos: float, avg_neg: float, std_neg: float,
                           num_positive_reviews: int, num_negative_reviews: int,
                           positive_files_positive_score: int, positive_files_negative_score: int,
                           negative_files_positive_score: int, negative_files_negative_score: int) -> None:
    """Print summary statistics."""
    print("\nStatistics:")
    print(f"Average Positive Words per Review: {avg_pos:.2f}")
    print(f"Standard Deviation of Positive Words: {std_pos:.2f}")
    print(f"Average Negative Words per Review: {avg_neg:.2f}")
    print(f"Standard Deviation of Negative Words: {std_neg:.2f}")

    # Calculate averages per review type
    avg_pos_in_pos_reviews = positive_files_positive_score / num_positive_reviews if num_positive_reviews > 0 else 0
    avg_neg_in_pos_reviews = positive_files_negative_score / num_positive_reviews if num_positive_reviews > 0 else 0
    avg_pos_in_neg_reviews = negative_files_positive_score / num_negative_reviews if num_negative_reviews > 0 else 0
    avg_neg_in_neg_reviews = negative_files_negative_score / num_negative_reviews if num_negative_reviews > 0 else 0

    print(f"Average Positive Words per Positive Review: {avg_pos_in_pos_reviews:.2f}")
    print(f"Average Negative Words per Positive Review: {avg_neg_in_pos_reviews:.2f}")
    print(f"Average Positive Words per Negative Review: {avg_pos_in_neg_reviews:.2f}")
    print(f"Average Negative Words per Negative Review: {avg_neg_in_neg_reviews:.2f}")

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function."""
    # Start timer
    start_time = time.time()
    
    # Initialize NLTK and sentiment lexicons
    initialize_nltk()
    positive_words, negative_words, stop_words, emphasis_words = setup_sentiment_lexicons()
    
    # Set up file paths
    base_path = "/Users/dhruhisheth/Desktop/research/datatest"
    positive_path = os.path.join(base_path, "train", "pos")
    negative_path = os.path.join(base_path, "train", "neg")
    
    # Load training data
    positive_files = read_directory(positive_path)
    negative_files = read_directory(negative_path)
    
    # Print dataset overview
    print_dataset_overview(positive_files, negative_files)
    
    # Analyze positive and negative review collections
    pos_frequencies, positive_files_positive_score, positive_files_negative_score = analyze_review_collection(
        positive_files, stop_words, emphasis_words, positive_words, negative_words
    )
    
    neg_frequencies, negative_files_positive_score, negative_files_negative_score = analyze_review_collection(
        negative_files, stop_words, emphasis_words, positive_words, negative_words
    )
    
    # Print sentiment scores
    print_sentiment_scores(positive_files_positive_score, positive_files_negative_score,
                         negative_files_positive_score, negative_files_negative_score)
    
    # Analyze individual reviews
    pos_counts, neg_counts, avg_pos, std_pos, avg_neg, std_neg = individual_reviews_analysis(
        positive_files + negative_files, stop_words, emphasis_words, positive_words, negative_words
    )
    
    # Print individual review statistics
    print_individual_review_stats(pos_counts, neg_counts)
    
    # Print summary statistics
    print_summary_statistics(avg_pos, std_pos, avg_neg, std_neg, len(positive_files), len(negative_files),
                           positive_files_positive_score, positive_files_negative_score,
                           negative_files_positive_score, negative_files_negative_score)
    
    # Create visualizations
    create_sentiment_bar_chart(positive_files_positive_score, positive_files_negative_score,
                              negative_files_positive_score, negative_files_negative_score,
                              std_pos, std_neg)
    
    create_frequent_words_charts(pos_frequencies, neg_frequencies, positive_words, negative_words)
    
    # Test classifier accuracy
    test_positive_path = os.path.join(base_path, "test", "pos")
    test_negative_path = os.path.join(base_path, "test", "neg")
    
    test_positive_files = read_directory(test_positive_path)
    test_negative_files = read_directory(test_negative_path)
    
    accuracy = test_classifier_accuracy(test_positive_files, test_negative_files,
                                      stop_words, emphasis_words, positive_words, negative_words)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # End timer and print execution time
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
