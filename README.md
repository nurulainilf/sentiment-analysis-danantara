# Sentiment Analysis of Public Opinion on BPI Danantara using IndoBERT and Naive Bayes

This project aims to analyze public sentiment toward the launch of **Badan Pengelola Investasi (BPI) Danantara**, Indonesia’s sovereign wealth fund, using a hybrid machine learning approach. It combines automatic sentiment labeling with a pre-trained **IndoBERT** model and sentiment classification using **Naive Bayes** trained on TF-IDF features.

## 🧠 Background

The launch of Danantara generated significant discussion on X (formerly Twitter), reflecting diverse public opinions ranging from optimism to skepticism. Understanding this sentiment is crucial to shaping effective public communication strategies and building institutional trust.

## 📊 Objectives

- To collect and analyze public tweets about Danantara across three time periods: pre-launch, launch day, and post-launch.
- To perform automatic sentiment labeling using **IndoBERT (indobert-base-p1)**.
- To train and evaluate a **Multinomial Naive Bayes** model using labeled data.
- To visualize sentiment distribution and identify key public concerns.

## 🗃 Dataset

- **Source:** Scraped from social media platform X using `Tweet Harvest`.
- **Total Tweets Collected:** 11,240
- **Final Cleaned Tweets:** 7,060 (after removing duplicates and bot-like accounts)

Each tweet is labeled with one of three sentiment classes:
- Positive
- Neutral
- Negative

## 🧹 Preprocessing Steps

1. Removal of spam/buzzer accounts and duplicate tweets.
2. Cleaning of symbols, numbers, URLs, and punctuation.
3. Case folding (conversion to lowercase).
4. Normalization of informal words using a slang dictionary.
5. Tokenization
6. Stopword removal (Bahasa Indonesia)
7. Stemming using **Sastrawi**

## 🧪 Modeling Workflow

### 1. Automatic Sentiment Labeling
- **Model:** `indobert-base-p1` from Hugging Face
- **Function:** To label tweet sentiment automatically into positive, neutral, or negative.

### 2. TF-IDF Feature Extraction
- Vectorizes cleaned tweets into numerical format.
- Max features: 5,000

### 3. Sentiment Classification
- **Model:** `MultinomialNB` (Naive Bayes)
- **Train/Test Split:** 80/20 stratified
- **Metrics:**
  - Accuracy: 0.73
  - Weighted F1-score: 0.71

## 📈 Results and Insights

- The majority of tweets expressed **negative sentiment (44.4%)**, followed by **neutral (37.4%)** and **positive (18.2%)**.
- Negative sentiment was associated with concerns over **transparency**, **corruption**, and **governance**.
- Word clouds revealed keyword patterns across the three time periods.
- The hybrid approach of IndoBERT + Naive Bayes proved to be effective and computationally efficient for sentiment classification.

## 📌 Key Technologies

- Python
- Scikit-learn
- Transformers (Hugging Face)
- Sastrawi (for Bahasa Indonesia stemming)
- Matplotlib / WordCloud


