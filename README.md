# Amazon Alexa Reviews Sentiment Analysis

Classifies Amazon Alexa customer reviews as **Positive** or **Negative**. The project explores 3,150 reviews, trains a TF-IDF + XGBoost classifier, and serves it through a small Flask app for single-text and CSV batch predictions. It turns raw review text into sentiment signals that are quick to scan.

## Project Files

| File | Description |
|------|-------------|
| `Data Exploration & Modeling.ipynb` | EDA, text cleaning, TF-IDF features, trains and compares 3 models |
| `Data/amazon_alexa.tsv` | Raw dataset: 3,150 reviews, 5 columns |
| `Data/SentimentBulk.csv` | 7-review sample input for batch prediction (`Sentence` column) |
| `Data/predictions.csv` | Sample app output for that input (adds `Predicted` column) |
| `Models/xgboost_model.pkl` | Trained XGBoost classifier used by the app |
| `Models/tfidfVectorizer.pkl` | Fitted TF-IDF vectorizer (2,500 features, 1–2 grams) |
| `app.py` | Flask app: landing page, prediction page, `POST /predict` for text or CSV |
| `templates/landing.html`, `templates/index.html` | Landing page and prediction UI |
| `*.png` (4 files) | Screenshots of the landing page, prediction UI and results |
| `requirements.txt`, `LICENSE` | Python dependencies (unpinned); MIT license |

## Dataset

Source: [Kaggle – Amazon Alexa Reviews](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews) · Tab-separated · 16 May – 31 Jul 2018 (92.5% in July) · 16 product variations.

| Column | Description |
|--------|-------------|
| `rating` | Star rating, 1–5 |
| `date` | Review date (`dd-Mon-yy`) |
| `variation` | Product variant (e.g. `Black  Dot`, `Charcoal Fabric`, `Configuration: Fire TV Stick`) |
| `verified_reviews` | Review text |
| `feedback` | Sentiment label: 1 = positive, 0 = negative |
| `length` | Character count of the review (derived in the notebook) |

## Approach

1. **Understand the data** – load the TSV, check shape, types, nulls, and the distribution of rating, feedback, variation and review length; word clouds per class.
2. **Clean** – drop the 1 null review (3,150 → 3,149). For text: keep letters only, lowercase, tokenise, POS-tag, lemmatise, remove stopwords (keeping "not").
3. **Features and split** – TF-IDF with up to 2,500 uni/bi-gram features; 70/30 split, `random_state=15` (2,204 train / 945 test).
4. **Model** – Naive Bayes, Random Forest, XGBoost; compare accuracy, precision, recall, F1 and confusion matrices; save XGBoost.
5. **Report and serve** – Flask app returns a label for one review, or a predictions CSV plus a sentiment pie chart for a batch.

## Key Insights

All dataset figures are recomputed from `amazon_alexa.tsv` (3,149 rows after cleaning).
- **Ratings skew high:** 72.6% are 5-star, 87.0% are 4+; mean rating 4.46.
- **Imbalanced labels:** 2,893 positive (91.9%) vs 256 negative (8.1%); ratings 1–2 → negative, 3–5 → positive.
- **Unhappy reviews are longer:** mean 217 vs 125 characters (median 138 vs 70).
- **Variation matters:** Charcoal Fabric scores highest among large groups (4.73 avg, 2.8% negative); plain White (4.17, 17.8% negative) and Black (4.23, 13.4%) score lowest.
- **Negative share fell over time:** 7.6% in July vs 13.4% (May) and 14.9% (June), though May–June hold only 236 reviews.

Model results on the 945-review test set (positive-class precision/recall; notebook outputs):

| Model | Accuracy | Precision | Recall | F1 | Negatives caught (of 78) |
|-------|----------|-----------|--------|----|--------------------------|
| Naive Bayes | 91.75% | 0.9175 | 1.0000 | 0.9570 | 0 |
| Random Forest | 94.29% | 0.9433 | 0.9977 | 0.9697 | 26 |
| **XGBoost (deployed)** | **95.24%** | 0.9587 | 0.9908 | 0.9745 | 41 |

Always predicting "Positive" scores 91.75%, so XGBoost gains ~3.5 points and catches only 52.6% of negatives (41 of 78).

## Tech Stack

- **Analysis and NLP:** Python 3.12, pandas, NumPy, Matplotlib, Seaborn, WordCloud, NLTK (POS tagging, lemmatising), TF-IDF n-grams
- **Modelling:** scikit-learn (Naive Bayes, Random Forest, metrics), XGBoost
- **App:** Flask, HTML/JS templates

## How to Use

1. Install dependencies: `pip install -r requirements.txt`
2. Download NLTK data: `stopwords`, `wordnet`, `punkt`, `averaged_perceptron_tagger` (newer NLTK also needs `punkt_tab` and `averaged_perceptron_tagger_eng`).
3. Start the app: `python app.py`, then open `http://localhost:5000`.
4. Predict: type a review on the page, or `POST /predict` with `{"text": "..."}` → `{"result": "Positive"}`. For batch, upload a CSV with a `Sentence` column; you get `predictions.csv` and a pie chart.
5. Notebook: run `Data Exploration & Modeling.ipynb` top to bottom. It reads `Data\amazon_alexa.tsv`; change the path separator on macOS/Linux.
