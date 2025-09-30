# Neural Dependency Parser

A PyTorch implementation of a transition-based dependency parser using neural networks, based on Chen & Manning (2014). This project implements a feed-forward neural network to predict parsing transitions for dependency parsing of English sentences.

## Overview

This parser uses an arc-standard transition system with three operations:
- **shift**: Move word from buffer to stack
- **left_arc**: Create dependency with left child
- **right_arc**: Create dependency with right child

The neural network learns to predict the correct transition at each parsing state based on the current stack and buffer configuration.

## Project Structure

```
├── conll_reader.py          # CoNLL-X format reader and dependency tree structures
├── get_vocab.py             # Extract vocabularies from training data
├── extract_training_data.py # Convert dependency trees to training matrices
├── train_model.py           # Neural network architecture and training
├── decoder.py               # Greedy parser implementation
├── evaluate.py              # Parser evaluation metrics
└── data/
    ├── train.conll         # Training data (~40k sentences)
    ├── dev.conll           # Development data (~5k sentences)
    ├── test.conll          # Test data (~2.5k sentences)
    ├── sec0.conll          # Small test set (~2k sentences)
    ├── words.vocab         # Word vocabulary
    ├── pos.vocab           # POS tag vocabulary
    └── model.pt            # Trained model weights
```

## Requirements

- Python 3.x
- PyTorch
- NumPy

Install dependencies:
```bash
pip install torch numpy
```

## Usage

### 1. Generate Vocabularies
Extract word and POS tag vocabularies from training data:
```bash
python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```

### 2. Extract Training Data
Convert dependency trees to neural network training matrices:
```bash
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```

### 3. Train Model
Train the neural dependency parser:
```bash
python train_model.py data/input_train.npy data/target_train.npy data/model.pt
```
*Note: Training may take 1+ hours on CPU. Consider using GPU acceleration.*

### 4. Parse Sentences
Parse sentences using the trained model:
```bash
python decoder.py data/model.pt data/dev.conll
```

### 5. Evaluate Parser
Evaluate parser performance:
```bash
python evaluate.py data/model.pt data/dev.conll
```

## Data Format

The project uses CoNLL-X format for dependency trees. Each sentence consists of lines with tab-separated fields:

```
1    The     _    DT    DT    _    2    det      _    _
2    cat     _    NN    NN    _    3    nsubj    _    _
3    eats    _    VB    VB    _    0    root     _    _
4    tasty   _    JJ    JJ    _    5    amod     _    _
5    fish    _    NN    NN    _    3    dobj     _    _
6    .       _    .     .     _    3    punct    _    _
```

Fields: ID, word, lemma, UPOS, POS, features, head, deprel, deps, misc

## Neural Network Architecture

- **Input**: 6 word indices (top 3 from stack + buffer)
- **Embedding**: 128-dimensional word embeddings
- **Hidden Layer**: 128 units with ReLU activation
- **Output**: 91 units (1 shift + 90 arc operations with labels)
- **Training**: Adagrad optimizer, CrossEntropy loss

## Features

The parser uses a simple feature representation:
- Top 3 words on the stack: `stack[-3], stack[-2], stack[-1]`
- Next 3 words on the buffer: `buffer[-3], buffer[-2], buffer[-1]`

Special tokens:
- `<ROOT>`: Root symbol (position 0)
- `<NULL>`: Padding for empty positions
- `<UNK>`: Unknown words (appear ≤1 times in training)
- `<CD>`: Numbers (POS tag CD)
- `<NNP>`: Proper nouns (POS tag NNP)

## Evaluation Metrics

- **LAS (Labeled Attachment Score)**: Percentage of correct (head, label) predictions
- **UAS (Unlabeled Attachment Score)**: Percentage of correct head predictions
- Both micro-averaged (overall) and macro-averaged (per sentence)

Expected performance: ~70% LAS on development data

## Implementation Details

### Key Classes

- `DependencyEdge`: Single word with grammatical properties
- `DependencyStructure`: Complete dependency tree
- `State`: Parser state (stack, buffer, dependencies)
- `FeatureExtractor`: Converts parser states to neural network input
- `DependencyModel`: Neural network architecture
- `Parser`: Greedy transition-based parser

### Parsing Algorithm

1. Initialize: stack=[0], buffer=[1,2,...,n], deps=∅
2. While buffer not empty:
   - Extract features from current state
   - Predict transition probabilities with neural network
   - Select highest-scoring legal transition
   - Update parser state
3. Return dependency structure

## References

Chen, D., & Manning, C. (2014). A fast and accurate dependency parser using neural networks. In Proceedings of EMNLP 2014.

## Notes

- Training data: WSJ portion of Penn Treebank converted to dependencies
- Current state-of-the-art: ~97% LAS (this implementation: ~70% LAS)
- For better performance, consider additional features from Chen & Manning (2014)
- GPU training recommended for faster convergence
