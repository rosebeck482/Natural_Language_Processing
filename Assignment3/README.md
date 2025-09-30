# BERT Semantic Role Labeling (SRL)

This repository contains a Jupyter notebook (`BERT_SRL.ipynb`) implementing semantic role labeling using BERT for the PropBank-style annotation framework. The notebook provides a complete PyTorch implementation that fine-tunes a pretrained BERT model to predict semantic roles for input tokens in a sentence, treating SRL as a sequence labeling task.

## Overview

The `BERT_SRL.ipynb` notebook treats semantic role labeling as a sequence tagging task where each token receives a BIO (Begin-Inside-Outside) tag representing its semantic role relative to a predicate. The notebook implementation follows the approach described in Collobert et al. (2011) and uses BERT's contextualized embeddings for improved performance.

### Example SRL Output

```
Input:  The judge scheduled to preside over his trial was removed from the case today .
Tokens: The judge scheduled to preside over his trial was removed from the case today .
Labels: B-ARG1 I-ARG1 B-V B-ARG2 I-ARG2 I-ARG2 I-ARG2 I-ARG2 O O O O O O O
Frame:        schedule.01
```

## Features

- **BERT-based Architecture**: Uses `bert-base-uncased` with custom classification head
- **Predicate Indication**: Leverages BERT's segment embeddings to mark predicate positions
- **Subword Tokenization**: Handles WordPiece tokenization with proper label propagation
- **Dual Evaluation**: Both token-level accuracy and span-based F1 metrics
- **GPU Optimized**: Designed for CUDA-enabled training and inference
- **Complete Pipeline**: From data preprocessing to model evaluation

## Requirements

### System Requirements
- Python 3.7+
- CUDA-capable GPU (required for training)
- 8GB+ GPU memory recommended

### Dependencies
```bash
torch>=1.9.0
transformers>=4.0.0
numpy
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Assignment3
```

2. **Install dependencies**
```bash
pip install torch transformers numpy
```

3. **Dataset is already included**
The dataset files are already included in the `data/` directory.

4. **Launch Jupyter notebook**
```bash
jupyter notebook BERT_SRL.ipynb
```

5. **Verify GPU availability** (run this in the notebook)
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

## Dataset

### OntoNotes 5.0 English SRL Annotations

The dataset uses PropBank-style semantic role annotations from OntoNotes 5.0. Each annotation consists of:

```
ontonotes/bc/cnn/00/cnn_0000.152.1                    # Unique identifier
The judge scheduled to preside over his trial was...    # Sentence tokens
        schedule.01                                     # Predicate frame
B-ARG1 I-ARG1 B-V B-ARG2 I-ARG2 I-ARG2 I-ARG2...     # BIO tags
```

### Files Structure
- `data/propbank_train.tsv` - Training data
- `data/propbank_dev.tsv` - Development data  
- `data/propbank_test.tsv` - Test data
- `data/role_list.txt` - List of semantic roles (filtered for frequency > 1000)

## Model Architecture

```
Input Sentence + Predicate Position
           ↓
    BERT Encoder (bert-base-uncased)
    - Token embeddings
    - Position embeddings  
    - Segment embeddings (predicate indicator)
           ↓
    Linear Classification Head (768 → num_roles)
           ↓
    BIO Tag Predictions
```

### Key Components

1. **BERT Encoder**: Pretrained `bert-base-uncased` model
2. **Segment Embeddings**: Repurposed to indicate predicate position (1=predicate, 0=other)
3. **Classification Head**: Linear layer mapping BERT hidden states to role labels
4. **Loss Function**: CrossEntropyLoss with ignore_index=-100 for padding

## Usage

### Running the Notebook

Open and execute the `BERT_SRL.ipynb` notebook to train the model and perform inference. The notebook contains all the code for data loading, model training, and evaluation.

### Example Usage (within the notebook)

```python
# after running the training cells in BERT_SRL.ipynb, use the label_sentence function
tokens = "A U. N. team spent an hour inside the hospital , where it found evident signs of shelling and gunfire .".split()

# label with predicate at index 13 ("found")
labels = label_sentence(tokens, pred_idx=13)

# display results
for token, label in zip(tokens, labels):
    print(f"{token:12} {label}")
```

### Expected Output
```
A            O
U.           O
N.           O
team         O
spent        O
an           O
hour         O
inside       O
the          B-ARGM-LOC
hospital     I-ARGM-LOC
,            O
where        B-ARGM-LOC
it           B-ARG0
found        B-V
evident      B-ARG1
signs        I-ARG1
of           I-ARG1
shelling     I-ARG1
and          I-ARG1
gunfire      I-ARG1
.            O
```

## Training

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
- **Batch Size**: 32
- **Epochs**: 2
- **Max Sequence Length**: 128 tokens
- **Device**: CUDA

### Training Process
Open and run the `BERT_SRL.ipynb` notebook. The training process includes:

```python
# the notebook handles all training steps:
# 1. data loading from data/ directory
# 2. model initialization
# 3. training loop with AdamW optimizer
# 4. model evaluation and saving
```

## Evaluation

### 1. Token-Level Accuracy
Measures the percentage of correctly predicted token labels (excluding padding tokens).

```python
evaluate_token_accuracy(model, dev_loader)
```

### 2. Span-Based Evaluation
Evaluates precision, recall, and F1-score for predicted argument spans.

```python
precision, recall, f1 = evaluate_spans(model, dev_loader)
```

**Span Extraction Example**:
- Target: `B-ARG1 I-ARG1 B-V B-ARG2 I-ARG2`
- Spans: `{(0,2): "ARG1", (3,5): "ARG2"}`

## Results

### Performance Metrics (After 2 epochs)

| Metric | Score |
|--------|-------|
| Training Loss | ~0.19 |
| Training Accuracy | ~94% |
| Dev Token Accuracy | ~93% |
| Dev Span-based F1 | ~0.83 |

### Comparison to State-of-the-Art
The achieved F1 score of 0.82-0.83 is competitive with 2018 state-of-the-art systems, demonstrating the effectiveness of the BERT-based approach.

## File Structure

```
Assignment3/
├── README.md                    # this file
├── BERT_SRL.ipynb             # jupyter notebook implementation
├── data/                      # dataset directory
│   ├── propbank_train.tsv     # training data
│   ├── propbank_dev.tsv       # development data  
│   ├── propbank_test.tsv      # test data
│   └── role_list.txt          # semantic role labels
└── srl_model_*.pt             # saved model checkpoints (generated after training)
```

## Implementation Details

### Tokenization with Label Alignment
The system handles BERT's WordPiece tokenization by:
1. Tokenizing each word individually
2. Propagating labels to subword tokens
3. Converting B- tags to I- tags for continuation tokens

### Predicate Indication Strategy
Uses BERT's token_type_ids (segment embeddings):
- `0`: Non-predicate tokens
- `1`: Predicate tokens (B-V positions)

### Data Processing Pipeline
1. **Input**: Raw tokens + BIO labels
2. **Tokenization**: WordPiece with label propagation  
3. **Formatting**: Add [CLS]/[SEP], create attention masks
4. **Padding**: Pad to max_length=128 with special tokens
5. **Tensor Conversion**: Convert to PyTorch tensors

### Training Optimizations
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling (AdamW optimizer)
- Mixed precision training support
- Efficient data loading with PyTorch DataLoader

## Citation

This implementation is based on:

```bibtex
@article{collobert2011natural,
  title={Natural language processing (almost) from scratch},
  author={Collobert, Ronan and Weston, Jason and Bottou, L{\'e}on and Karlen, Michael and Kavukcuoglu, Koray and Kuksa, Pavel},
  journal={Journal of machine learning research},
  volume={12},
  pages={2493--2537},
  year={2011}
}
```

**Dataset**: OntoNotes 5.0 English SRL annotations provided for educational use in COMS W4705 at Columbia University.

**Note**: This dataset is licensed for educational use only and should not be used for commercial purposes or projects unrelated to Columbia University teaching and research.

---

## Contributing

This is an educational project for COMS W4705. For questions or improvements, please contact the course staff.

## License

Educational use only. Dataset provided under Columbia University LDC subscription terms.