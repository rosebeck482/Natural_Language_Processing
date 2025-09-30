# Trigram Language Model

A Python implementation of a trigram language model for natural language processing tasks, including text probability estimation and essay quality classification.

## Overview

This project implements a statistical language model that predicts the probability of word sequences using trigrams (3-word sequences). The model includes smoothing techniques and can be used for various NLP tasks such as text generation, perplexity calculation, and document classification.

## Features

- **N-gram extraction**: Generates unigrams, bigrams, and trigrams from text corpora
- **Probability estimation**: Calculates raw and smoothed probabilities using linear interpolation
- **Perplexity evaluation**: Measures model quality on test data
- **Essay scoring**: Binary classification of essay quality based on language model perplexity
- **Unknown word handling**: Robust processing of out-of-vocabulary words

## Data Structure

The project expects the following data organization:
```
data/
├── brown_train.txt          # Training corpus
├── brown_test.txt           # Test corpus
└── ets_toefl_data/
    ├── train_high.txt       # High-quality essays (training)
    ├── train_low.txt        # Low-quality essays (training)
    ├── test_high/           # High-quality essays (testing)
    │   └── *.txt files
    └── test_low/            # Low-quality essays (testing)
        └── *.txt files
```

## Usage

### Basic Usage

```bash
python trigram_model.py train_file test_file train_high train_low test_high_dir test_low_dir
```

### Example

```bash
python trigram_model.py data/brown_train.txt data/brown_test.txt data/ets_toefl_data/train_high.txt data/ets_toefl_data/train_low.txt data/ets_toefl_data/test_high data/ets_toefl_data/test_low
```

### Interactive Mode

```bash
python -i trigram_model.py data/brown_train.txt data/brown_test.txt data/ets_toefl_data/train_high.txt data/ets_toefl_data/train_low.txt data/ets_toefl_data/test_high data/ets_toefl_data/test_low
```

## Output

The script provides:
- **Model statistics**: Number of sentences, unigrams, bigrams, and trigrams
- **Training perplexity**: Model performance on training data
- **Test perplexity**: Model performance on test data (Brown corpus)
- **Essay classification accuracy**: Performance on essay quality prediction task

## Implementation Details

### Core Components

1. **TrigramModel Class**: Main model implementation
   - Builds vocabulary from training corpus
   - Counts n-gram frequencies
   - Calculates probabilities with smoothing

2. **Probability Methods**:
   - `raw_trigram_probability()`: Unsmoothed P(w₃|w₁,w₂)
   - `raw_bigram_probability()`: Unsmoothed P(w₂|w₁)
   - `raw_unigram_probability()`: Unsmoothed P(w)
   - `smoothed_trigram_probability()`: Linear interpolation smoothing

3. **Evaluation Methods**:
   - `sentence_logprob()`: Log probability of sentences
   - `perplexity()`: Perplexity calculation
   - `essay_scoring_experiment()`: Binary classification

### Smoothing Technique

Uses linear interpolation with equal weights (λ₁ = λ₂ = λ₃ = 1/3):
```
P_smooth(w₃|w₁,w₂) = λ₁P(w₃|w₁,w₂) + λ₂P(w₃|w₂) + λ₃P(w₃)
```

### Special Token Handling

- **START**: Sentence beginning marker
- **STOP**: Sentence ending marker  
- **UNK**: Unknown/rare words (frequency ≤ 1)

## Requirements

- Python 3.x
- Standard library modules:
  - `collections.defaultdict`
  - `math`
  - `random`
  - `os`
  - `sys`

## File Format

Input text files should contain one sentence per line. The model automatically:
- Converts text to lowercase
- Tokenizes on whitespace
- Handles sentence boundaries
- Maps rare words to UNK tokens

## Performance Notes

- Lexicon includes only words appearing more than once in training data
- Perplexity calculation excludes START tokens from denominator
- Essay classification uses perplexity comparison between models
- Model handles zero-probability cases with uniform distribution fallback

## Course Information

**COMS W4705 - Natural Language Processing - Fall 2024**  
**Programming Homework 1 - Trigram Language Models**  
**Instructor: Daniel Bauer**
