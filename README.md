# 🎬 Movie Reviews Sentiment Analysis

This project performs lexicon-based sentiment analysis on movie reviews using **NLTK**, custom heuristics for **negation**, **emphasis**, and **word frequency analysis**. It includes visualization of sentiment trends and accuracy evaluation on test data.

---

## 📌 Features

- ✅ Preprocessing using NLTK (tokenization, stopword removal)
- 🔁 Handles **negation** (e.g., `not good` → `not_good`)
- 📢 Detects **emphasized** words (e.g., `very happy` → `emph_happy`)
- 📈 Word frequency analysis with sentiment weighting
- 📊 Visualizations: bar charts of sentiment scores, top words
- 🧪 Accuracy testing on labeled positive/negative reviews



## 📁 Directory Structure

```bash
Sentiment Analysis/
├── sentiment_analysis.py # Main script
├── data/ 
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
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/dhruhisheth/Movie-Reviews-Sentiment-Analysis.git
cd Movie-Reviews-Sentiment-Analysis
```

2. Install Required Libraries
```bash
pip install nltk matplotlib numpy
```

4. Download Required NLTK Resources
```bash
import nltk
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('stopwords')
```

6. Run the Script
```bash
python3 SentimentAnalysis.py
```

📊 Output Includes
Sentiment word count bar chart with error bars
Top frequent positive/negative words
Per-review positive/negative word counts
Accuracy on the test dataset
Summary statistics of average and standard deviation

💡 Example Use Case
A review like:
"The movie was absolutely amazing and very touching."
Will be processed as:
["emph_amazing", "emph_touching"]
And classified as positive, with weighted sentiment due to emphasis.
