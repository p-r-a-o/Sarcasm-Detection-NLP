# Sarcasm Detection on Reddit Using Classical ML, RNNs, and Transformers

## Overview

Sarcasm detection remains one of the most challenging tasks in Natural Language Processing (NLP) because sarcastic statements often convey meanings that differ from their literal wording. Traditional NLP systems tend to interpret text at face value, making it difficult to recognize irony, humor, and contextual contradictions.

This project evaluates multiple approaches for sarcasm detection on Reddit comments using the Reddit SARC (Self-Annotated Reddit Corpus) dataset. We compare classical machine learning models, recurrent neural networks, and transformer-based architectures to understand how model complexity, attention mechanisms, and conversational context affect performance.

## Objectives

* Establish baseline performance using classical machine learning models.
* Evaluate the effectiveness of BiLSTM and BiGRU architectures.
* Investigate the impact of attention mechanisms.
* Measure the value of conversational context (parent comments).
* Compare recurrent models against transformer-based models such as BERT and DistilBERT.

---

## Dataset

### SARC (Self-Annotated Reddit Corpus)

The project uses the SARC dataset, a large collection of Reddit comments where sarcasm is self-annotated by users.

Key advantages:

* Large-scale sarcasm annotations
* Real-world conversational data
* Availability of parent-comment context
* Diverse topics across Reddit communities

### Dataset Statistics

| Split    | Samples      |
| -------- | ------------ |
| Training | 1,010,825    |
| Test     | 251,608      |
| Total    | ~1.3 Million |

### Important Features

| Feature        | Description                        |
| -------------- | ---------------------------------- |
| label          | Sarcastic (1) or Non-Sarcastic (0) |
| comment        | Reddit reply text                  |
| parent_comment | Original comment being replied to  |
| subreddit      | Community where comment was posted |
| author         | Username                           |
| score          | Net vote score                     |
| date           | Timestamp                          |

---

## Models Evaluated

### 1. Classical Machine Learning

#### Logistic Regression

* GloVe embeddings
* Mean-pooled sentence representation
* L2 regularization
* Saga optimizer

#### XGBoost

* GloVe embeddings
* Gradient boosted decision trees
* Captures non-linear relationships

---

### 2. BiLSTM Models

Variants:

* BiLSTM (No Context, No Attention)
* BiLSTM (No Context, Attention)
* BiLSTM (Context, No Attention)
* BiLSTM (Context, Attention)

Features:

* Bidirectional sequence modeling
* Pre-trained GloVe embeddings
* Attention-based token weighting
* Parent-comment integration

---

### 3. BiGRU Models

Variants:

* BiGRU (No Context, No Attention)
* BiGRU (No Context, Attention)
* BiGRU (Context, No Attention)
* BiGRU (Context, Attention)

Features:

* Lightweight alternative to LSTMs
* Faster training
* Additive attention mechanism
* Context-aware and context-free configurations

---

### 4. Transformer Models

#### DistilBERT

* Lightweight transformer
* Faster inference
* Reduced computational cost

#### BERT

* Full transformer architecture
* Contextual embeddings
* Self-attention mechanisms
* Fine-tuned for binary classification

Both models were evaluated with and without parent-comment context.

---

## Embeddings & Tokenization

### GloVe Embeddings

Used for:

* Logistic Regression
* XGBoost
* BiLSTM
* BiGRU

Configuration:

* Model: `glove.2024.wikigiga.300d`
* Dimension: 300
* Vocabulary: 1.2M tokens
* Corpus Size: 11.9B tokens

### Transformer Tokenization

Used for:

* BERT
* DistilBERT

Features:

* Pre-trained tokenizers
* Dynamic contextual embeddings
* Learned semantic representations

---

## Training Configuration

### Logistic Regression

```python
solver = "saga"
C = 1.0
max_iter = 1000
```

### XGBoost

```python
n_estimators = 100
max_depth = 6
learning_rate = 0.1
objective = "logloss"
```

### BiLSTM

| Parameter      | Value             |
| -------------- | ----------------- |
| Max Length     | 100               |
| Embedding Size | 300               |
| Hidden Units   | 128 per direction |
| Dropout        | 0.3               |
| Batch Size     | 256               |
| Learning Rate  | 1e-3              |
| Optimizer      | Adam              |
| Epochs         | 20                |

### BiGRU

| Parameter      | Value             |
| -------------- | ----------------- |
| Max Length     | 60                |
| Embedding Size | 300               |
| Hidden Units   | 128 per direction |
| Dropout        | 0.3               |
| Batch Size     | 256               |
| Learning Rate  | 2e-3              |
| Optimizer      | AdamW             |
| Epochs         | 30                |

### BERT / DistilBERT

| Parameter     | Value |
| ------------- | ----- |
| Max Length    | 128   |
| Batch Size    | 32    |
| Epochs        | 3     |
| Learning Rate | 2e-5  |
| Weight Decay  | 0.01  |
| Optimizer     | AdamW |

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Weighted F1 Score
* ROC-AUC
* Average Precision (AP)
* Classification Report
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## Results

| Model               | Variant                  | Accuracy (%) | F1 Score   |
| ------------------- | ------------------------ | ------------ | ---------- |
| Logistic Regression | No Context               | 62.33        | 0.6234     |
| XGBoost             | No Context               | 65.88        | 0.6558     |
| BiLSTM              | No Context, No Attention | 74.13        | 0.7355     |
| BiLSTM              | No Context, Attention    | 73.95        | 0.7358     |
| BiLSTM              | Context, No Attention    | 74.66        | 0.7422     |
| BiLSTM              | Context, Attention       | 74.45        | 0.7454     |
| BiGRU               | No Context, No Attention | 73.56        | 0.7342     |
| BiGRU               | No Context, Attention    | 73.84        | 0.7303     |
| BiGRU               | Context, No Attention    | 73.23        | 0.7215     |
| BiGRU               | Context, Attention       | 73.21        | 0.7211     |
| DistilBERT          | No Context               | 77.69        | 0.7767     |
| DistilBERT          | Context                  | 77.73        | 0.7773     |
| BERT                | No Context               | 78.33        | 0.7833     |
| BERT                | Context                  | **78.66**    | **0.7866** |

---

## Key Findings

### Transformers Outperform RNNs

BERT and DistilBERT achieved the best overall performance, demonstrating the effectiveness of self-attention and contextual embeddings for sarcasm detection.

### Context Helps Transformers

Including parent-comment context consistently improved transformer performance, suggesting that sarcasm often depends on conversational history.

### Mixed Results for Attention in RNNs

Attention mechanisms did not consistently improve BiLSTM and BiGRU performance, indicating that attention alone may not be sufficient for capturing sarcastic intent.

### BiGRU vs BiLSTM

BiGRU models achieved comparable results while being computationally lighter, although BiLSTM variants generally performed slightly better.

### DistilBERT is Highly Competitive

DistilBERT achieved performance close to BERT while requiring fewer computational resources, making it an attractive deployment option.

---

## Conclusion

This project demonstrates the progression of sarcasm detection performance across multiple generations of NLP models. Classical machine learning methods provide strong baselines, recurrent neural networks improve performance through sequential modeling, and transformer architectures achieve the best results by leveraging contextual understanding and self-attention.

The best-performing model was **BERT with conversational context**, achieving:

* **Accuracy:** 78.66%
* **F1 Score:** 0.7866

These results highlight the importance of contextual understanding in sarcasm detection and reinforce the effectiveness of transformer-based architectures for nuanced language understanding tasks.

---

## Future Work

* Explore larger transformer models (RoBERTa, DeBERTa, Longformer)
* Incorporate subreddit-specific information
* Use conversation threads beyond a single parent comment
* Apply multi-task learning with sentiment analysis
* Investigate sarcasm explainability using attention visualization

## Authors

Project developed as part of an NLP study on sarcasm detection using Reddit conversations.
