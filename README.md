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
Sentiment Analysis/
│
├── sentiment_analysis.py # Main script
├── data/ # Ignored large text files (manually moved)
│ ├── urls_pos.txt
│ └── urls_neg.txt
├── datatest/
│ ├── train/
│ │ ├── pos/ # Positive training reviews
│ │ └── neg/ # Negative training reviews
│ └── test/
│ ├── pos/ # Positive test reviews
│ └── neg/ # Negative test reviews
├── .gitignore
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/dhruhisheth/Movie-Reviews-Sentiment-Analysis.git
cd Movie-Reviews-Sentiment-Analysis
2. Install Required Libraries
pip install nltk matplotlib numpy
3. Download Required NLTK Resources
import nltk
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('stopwords')
4. Run the Script
python3 sentiment_analysis.py
📊 Output Includes
Sentiment word count bar chart with error bars
Top frequent positive/negative words chart
Per-review positive/negative word counts
Accuracy on test data
Summary statistics of average and standard deviation
🧪 Example Use Case
A review like:
"The movie was absolutely amazing and very touching"
Would be preprocessed as:
["emph_amazing", "emph_touching"]
And classified as positive, with higher score due to emphasis.
📈 Accuracy
Accuracy is calculated by classifying test reviews and comparing predictions against ground truth labels (pos and neg folders). Final result is printed at the end of the script.
💡 Future Improvements
Add a web UI or Flask API
Integrate with transformer models (e.g., BERT) for hybrid sentiment analysis
Export results to CSV or dashboard


