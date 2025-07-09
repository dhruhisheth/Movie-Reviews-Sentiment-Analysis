# 🎬 Movie Reviews Sentiment Analysis

This project performs lexicon-based sentiment analysis on movie reviews using **NLTK**, custom heuristics for **negation**, **emphasis**, and **word frequency analysis**. It includes visualization of sentiment trends and classification accuracy on test data.

---

## 📌 Features

- ✅ Preprocessing using NLTK (tokenization, stopword removal)
- 🔁 Handles **negation** (e.g., `not good` → `not_good`)
- 📢 Detects **emphasized** words (e.g., `very happy` → `emph_happy`)
- 📈 Analyzes sentiment by matching against the **opinion lexicon**
- 📊 Visualizes sentiment word counts and frequent word distributions
- 🧪 Evaluates model performance with test data and accuracy metrics

---

## 🧠 How It Works

1. **Text Preprocessing:** Tokenizes, lowercases, removes stopwords, and handles special cases (negation, emphasis).
2. **Word Frequency Extraction:** Counts word frequency using `Counter`.
3. **Sentiment Categorization:** Classifies words as positive or negative using NLTK's opinion lexicon.
4. **Scoring & Classification:** Applies weighted scoring for strong/emphasized sentiment words.
5. **Visualization:** Displays bar charts of sentiment scores and most frequent words.
6. **Model Evaluation:** Tests accuracy using separate positive/negative review folders.

---

## 🗂️ Directory Structure

