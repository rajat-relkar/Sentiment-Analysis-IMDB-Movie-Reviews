# IMDb Movie Review Sentiment Analysis

## Project Overview
This project aims to analyze IMDb movie reviews and predict whether a review expresses a **positive** or **negative** sentiment. Sentiment analysis is a key Natural Language Processing (NLP) task that helps in understanding public opinion and automating text interpretation.  

Using a combination of **text preprocessing**, **feature extraction techniques**, and **machine learning algorithms**, the model was trained to classify the sentiment of user-submitted IMDb reviews. The project evaluates multiple models and compares their performance using standard classification metrics.

---

## Objectives
- To preprocess and clean IMDb movie reviews for analysis.  
- To convert textual data into numerical representations using **TF-IDF**, **Bag-of-Words**, and **Word2Vec** techniques.  
- To train and evaluate multiple machine learning models for sentiment classification.  
- To compare models based on **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC** metrics.  

---

## Key Concepts Used
- **Natural Language Processing (NLP):** Techniques to process and analyze large amounts of natural language text.  
- **Feature Engineering:** Using TF-IDF, Count Vectorizer, and Word2Vec embeddings for text representation.  
- **Machine Learning Algorithms:** Logistic Regression, Naïve Bayes, Support Vector Machine (SVM), and Random Forest.  
- **Model Evaluation:** Confusion matrix, ROC curve, AUC score, and detailed classification report.  

---

## Dataset
The dataset used is a collection of IMDb movie reviews labeled as **positive** or **negative**. Each review represents a user’s opinion on a particular movie.

Typical dataset structure:
| Review | Sentiment |
|---------|------------|
| "The movie was absolutely wonderful!" | Positive |
| "I didn’t like the storyline at all." | Negative |

---

## Project Workflow

### 1. Importing Libraries
The project uses essential libraries such as:
```python
pandas, numpy, sklearn, nltk, gensim, matplotlib, seaborn
```

### 2. Data Loading and Exploration
The IMDb dataset is loaded and explored to check for:
- Null values or missing reviews  
- Distribution of sentiments  
- Basic statistics on review lengths  

### 3. Text Preprocessing
To prepare the text for modeling:
- **Lowercasing** all text  
- **Removing punctuation, stopwords, and special characters**  
- **Tokenization** of words  
- **Lemmatization** using WordNetLemmatizer to reduce words to their root form  

### 4. Feature Extraction
The preprocessed text is converted into numerical vectors using:
- **TF-IDF Vectorizer:** Weighs words by importance.  
- **Count Vectorizer:** Simple frequency-based representation.  
- **Word2Vec (Gensim):** Context-aware embeddings for semantic meaning.

### 5. Model Training
Multiple machine learning models were trained:
- **Logistic Regression** – Baseline linear model  
- **Multinomial Naïve Bayes** – Probabilistic classifier for text  
- **Linear SVM (Support Vector Machine)** – High-margin classifier for binary tasks  
- **Random Forest Classifier** – Ensemble learning for robust performance  

### 6. Model Evaluation
Each model’s performance was measured using:
- **Accuracy:** Overall correctness of predictions  
- **Precision & Recall:** Measure of relevance and completeness  
- **F1-score:** Harmonic mean of precision and recall  
- **ROC-AUC:** Overall ranking capability across thresholds  

---

## Results Summary
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|----------|------------|------|
| Logistic Regression | High | High | High | High | Excellent |
| Naïve Bayes | Moderate | Moderate | Moderate | Moderate | Good |
| Linear SVM | High | High | High | High | Excellent |
| Random Forest | Competitive | High | High | High | Good |

---

## Key Learnings
- Preprocessing text data significantly improves model performance.  
- TF-IDF features outperform simple Bag-of-Words in most cases.  
- Word2Vec embeddings capture contextual meaning but may require more complex models.  
- Linear models like **Logistic Regression** and **SVM** are strong baselines for sentiment classification tasks.  

---

## Tools and Technologies
- **Language:** Python  
- **Libraries:** scikit-learn, NLTK, Gensim, Matplotlib, Seaborn, Pandas  
- **Environment:** Jupyter Notebook  

---
