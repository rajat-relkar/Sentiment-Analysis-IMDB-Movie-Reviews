# IMDb Movie Review Sentiment Analysis

This project focuses on performing **sentiment analysis** on IMDb movie reviews using a **Long Short-Term Memory (LSTM)** neural network. The goal is to classify each review as **positive** or **negative**, demonstrating how deep learning can capture contextual meaning in natural language.

---

## Project Overview

Customer opinions and online reviews carry immense value for businesses and audiences alike. However, manually processing thousands of reviews is infeasible — hence the need for **automated sentiment classification**.

This project leverages a **Recurrent Neural Network (RNN)** architecture, specifically **LSTM**, to analyze IMDb reviews and predict sentiment. It demonstrates the entire machine learning pipeline — from text preprocessing to model evaluation.

---

## Techniques & Concepts

The following natural language processing (NLP) and deep learning concepts were applied:

- **Text Preprocessing**
  - Cleaning text (removing punctuation, stopwords, and HTML tags)
  - Tokenization of words
  - Padding sequences to ensure uniform input length

- **Word Embeddings**
  - Tokenizer-based vocabulary creation
  - Word-to-index mapping for numerical representation of text

- **Deep Learning**
  - LSTM model trained on preprocessed embeddings for sequential text understanding

---

## Model Architecture

The LSTM model is designed with simplicity and performance in mind:

| Layer | Description |
|-------|--------------|
| **Embedding Layer** | Converts integer-encoded words into dense vectors |
| **LSTM Layer** | Captures contextual dependencies in sequential text |
| **Dense Layer (ReLU)** | Learns complex representations |
| **Dense Output Layer (Sigmoid)** | Predicts binary sentiment (positive/negative) |

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy, Precision, Recall  

---

## Implementation Steps

1. **Data Loading**  
   Load the IMDb dataset containing labeled movie reviews.

2. **Preprocessing**  
   - Clean and tokenize reviews  
   - Convert text into padded sequences using Keras’ Tokenizer  

3. **Model Building**  
   - Construct a sequential LSTM model  
   - Compile using `binary_crossentropy` and the Adam optimizer  

4. **Training**  
   - Train the model on training data with validation split  
   - Monitor accuracy and loss over epochs  

5. **Evaluation**  
   - Evaluate model on test data  
   - Compute Accuracy, Precision, and Recall metrics  

6. **Prediction**  
   - Predict sentiment for new custom reviews  

---

## Model Performance

| Metric | Score |
|---------|--------|
| **Accuracy** | 0.87 |
| **Precision** | 0.85 |
| **Recall** | 0.86 |

These results show that the LSTM network effectively captures contextual sentiment in movie reviews.

---

## Tools & Libraries

This project utilizes the following libraries and frameworks:

- **Python**
- **TensorFlow / Keras** – Deep learning model building
- **NLTK** – Text preprocessing and tokenization
- **NumPy, Pandas** – Data manipulation and cleaning
- **Matplotlib** – Performance visualization

---

## Key Learnings

- Understanding of text vectorization and embedding techniques  
- Handling variable-length input sequences using padding  
- Building and training LSTM models for sequence-based classification  
- Evaluating models with precision, recall, and accuracy metrics  

---

