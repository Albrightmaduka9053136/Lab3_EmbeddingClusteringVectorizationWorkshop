# Word Embeddings, Clustering and Vectorization Workshop

A comprehensive tutorial for learning word embeddings, text clustering, and vectorization techniques in Natural Language Processing (NLP).

## 📋 Overview

This workshop covers three main approaches to learning word embeddings and understanding word relationships through clustering:

1. **Word2Vec** - Predictive embedding models (CBOW and Skip-gram)
2. **Brown Clustering** - Hierarchical word clustering based on co-occurrence patterns
3. **GloVe** - Global vectors from co-occurrence statistics

## 🎯 Learning Objectives

- Understand word embeddings and their semantic properties
- Implement Word2Vec with both CBOW and Skip-gram architectures
- Build co-occurrence matrices for word relationship analysis
- Perform hierarchical clustering and visualization with dendrograms
- Load and apply pre-trained GloVe embeddings
- Compare different embedding techniques for NLP tasks

## 📚 Contents

### Part 1: Vector Stores and Dimensionality Reduction

#### Word2Vec (CBOW & Skip-gram)
- Introduction to Word2Vec architecture
- Tokenization and text preprocessing
- Training CBOW models
- Training Skip-gram models
- Word similarity analysis
- Vector representation interpretation

#### Brown Corpus Clustering
- Building co-occurrence matrices
- Hierarchical clustering with Ward's method
- Dendrogram visualization
- Word relationship inference from clustering

#### GloVe Embeddings
- Understanding global co-occurrence statistics
- Loading pre-trained GloVe models
- Applying embeddings to vocabulary
- Comparing with Word2Vec

### Part 2: Workshop Exercise

Teams implement their own embedding pipeline with:
- Document collection and preprocessing
- Tokenization and normalization pipeline
- Word2Vec model training
- GloVe embedding application
- Comparative analysis and documentation

## 🛠️ Requirements

```
nltk==3.9.2
gensim==4.4.0
numpy==2.3.5
scipy==1.16.3
matplotlib==3.10.7
scikit-learn==1.7.2
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The workshop uses:
- **Text Corpus**: NLP introduction text and Brown Corpus (NLTK)
- **Pre-trained Embeddings**: GloVe 6B (50-dimensional vectors)

GloVe embeddings are not included due to size constraints. Download from:
- [Stanford NLP - GloVe](https://nlp.stanford.edu/projects/glove/)

## 🚀 Quick Start

1. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook EmbeddingClusteringVectorizationWorkshop_Fall2025.ipynb
   ```

3. **Run through cells sequentially** to execute the full NLP pipeline

## 📖 Key Concepts

### Word Embeddings
Dense vector representations where semantically similar words are positioned close together in high-dimensional space.

### Cosine Similarity
$$\text{cosine\_similarity}(\vec{v}_a, \vec{v}_b) = \frac{\vec{v}_a \cdot \vec{v}_b}{\|\vec{v}_a\| \|\vec{v}_b\|}$$

### Skip-gram Objective
Maximizes the probability of predicting context words given a center word:
$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} \mid w_t)$$

### Co-occurrence Matrix
A V×V matrix where entry (i,j) represents how often word i appears near word j within a context window.

## 📁 Project Structure

```
EmbeddingClusteringVectorizationWorkshop/
├── EmbeddingClusteringVectorizationWorkshop_Fall2025.ipynb  # Main notebook
├── requirements.txt                                          # Python dependencies
├── README.md                                                 # This file
├── nltk_data/                                               # NLTK tokenizers
└── .gitignore                                               # Git configuration
```

## ✨ Features

- **Interactive Jupyter Notebook** with explanations and visualizations
- **Step-by-step implementation** of Word2Vec models
- **Hierarchical clustering** with dendrogram visualization
- **Comparative analysis** of embedding techniques
- **Mathematical explanations** with LaTeX formulas
- **Practical examples** on real text data

## 🔗 Resources

- [Word2Vec Paper](https://arxiv.org/abs/1310.4546)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [NLTK Documentation](https://www.nltk.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

## 👥 Team Members

This workshop is designed for teams of 2. Add team member names here after completion.

## 📝 Notes

- Large GloVe embedding files are excluded from the repository (use `.gitignore`)
- NLTK tokenizers are automatically downloaded on first run
- The notebook can be run multiple times without re-downloading data
- Model training times vary based on corpus size and hardware

## 📄 License

Educational material for NLP workshop. Dataset sources:
- NLTK Brown Corpus: Public domain
- GloVe: Public license from Stanford University

## 🤝 Contributing

For questions or suggestions about the workshop, please refer to the instructor.