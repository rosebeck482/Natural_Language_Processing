# Natural Language Processing (NLP)

**[Fall 2024] COMS 4705**

This repository contains assignments from COMS W4705 - Natural Language Processing course

## Project Overview

### Assignment1: Trigram Language Model
**Directory**: `Assignment1/`  
**Focus**: Statistical language modeling and text classification

A Python implementation of a trigram language model for natural language processing tasks. This project demonstrates:
- N-gram extraction and probability estimation
- Linear interpolation smoothing techniques
- Perplexity evaluation for model quality assessment
- Binary classification of essay quality using language model perplexity
- Unknown word handling and robust text processing

**Key Features**:
- Statistical language modeling with trigrams
- Essay quality classification (high vs. low quality)
- Perplexity-based evaluation metrics
- Smoothing techniques for handling sparse data

**Data**: Brown corpus and ETS TOEFL essay data for training and evaluation.

---

### Assignment2: Neural Dependency Parser
**Directory**: `Assignment2/`  
**Focus**: Transition-based dependency parsing with neural networks

A PyTorch implementation of a neural dependency parser based on Chen & Manning (2014). This project implements:
- Arc-standard transition system for dependency parsing
- Feed-forward neural network for transition prediction
- Greedy parsing algorithm with neural network guidance
- Evaluation using labeled and unlabeled attachment scores

**Key Features**:
- Neural network architecture for parsing decisions
- Transition-based parsing with shift/left_arc/right_arc operations
- Feature extraction from parser states
- Performance evaluation with LAS/UAS metrics

**Data**: CoNLL-X format dependency trees from Penn Treebank.

---

### Assignment3: BERT Semantic Role Labeling
**Directory**: `Assignment3/`  
**Focus**: Semantic role labeling using BERT for sequence tagging

A Jupyter notebook implementation of semantic role labeling using BERT, treating SRL as a sequence labeling task. This project demonstrates:
- BERT-based architecture for semantic role prediction
- BIO tagging for argument identification
- Predicate indication using segment embeddings
- Span-based evaluation metrics

**Key Features**:
- BERT fine-tuning for SRL task
- Sequence labeling with BIO tags
- Predicate-aware processing
- Token-level and span-based evaluation

**Data**: PropBank-style semantic role annotations from OntoNotes 5.0.

---

## Course Topics Covered

| # | Concept | Details |
|---|---------|---------|
| 1 | Levels of linguistic Representation & Ambiguity | |
| 2 | Probability and Machine Learning Basics & Text Classification | Naive Bayes<br>n-gram language models |
| 3 | Sequence Labeling & Syntax and Grammar | Hidden Markov Models (HMMs)<br>Part-of-Speech (PoS) Tagging & Named-entity recognition<br>Context-Free Grammars (CFGs) |
| 4 | Dependency structures | PCFGS: CKY<br>Transition-based Dependency Parsing |
| 5 | Machine Learning for NLP & Neural Networks | Log-linear models<br>PyTorch basics |
| 6 | Static Word Representations | Distributed representations and words embeddings (word2vec)<br>WordNet and Words Sense Disambiguation |
| 7 | Contextualized Word Representations | RNN & LSTM - ELMo |
| 8 | Attention & Transformer based models | BERT & GPT<br>Pretrained & Fine-Tuning & Prompting |
| 9 | Sentence-level Semantics & Abstract Meaning Representation | Semantic Role Labeling (FrameNet, PropBank)<br>AMR |
| 10 | Summarization / Machine Translation & Multimodal NLP | Language and Vision |
