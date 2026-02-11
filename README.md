# Extractive-QA-Finetuning
# Extractive Question Answering: Fine-tuning BERT and LLaMA on SQuAD

This project explores and compares fine-tuning strategies for extractive question answering (QA) using two fundamentally different transformer architectures — **BERT** (encoder-only) and **LLaMA** (decoder-only) — on the Stanford Question Answering Dataset (SQuAD).

---

## 📋 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [References](#references)

---

## Overview

Extractive QA requires a model to identify and extract a precise answer span from a given context passage in response to a question. This project investigates:
- How encoder-only (BERT) and decoder-only (LLaMA) architectures perform on this task
- The impact of fine-tuning, prompt engineering, and parameter-efficient adaptation (LoRA)
- Training efficiency and sample efficiency trade-offs between the two model families

---

## Results

### BERT

| Configuration | Exact Match (%) | F1 Score (%) | Training Data |
|---|---|---|---|
| BERT-uncased Baseline | 0.2 | 4.18 | - |
| BERT-uncased Fine-tuned | 59.2 | 64.25 | 87,599 examples |
| BERT-pretrained Baseline | 69.0 | 68.45 | - |
| BERT-pretrained Fine-tuned | **84.0** | **83.08** | 87,599 examples |

### LLaMA

| Configuration | Exact Match (%) | F1 Score (%) | Training Data |
|---|---|---|---|
| Baseline | 47.5 | 67.83 | - |
| + Prompt Engineering | 67.0 | 76.96 | - |
| + LoRA Fine-tuning | **96.2** | **97.16** | 2,000 examples |

### Key Highlights
- 🏆 LLaMA with LoRA achieves the best performance (96.2% EM) using only **2,000 training examples** — 43× fewer than BERT
- 📈 Prompt engineering alone improved LLaMA baseline by **+19.5pp EM** without any parameter updates
- ⚡ BERT-pretrained fine-tuning improved EM from 69% to 84% on the full SQuAD training set

---

## Project Structure

```
├── LLM_fine_tuning.ipynb       # LLaMA fine-tuning pipeline (LoRA, prompt engineering, FP16)
├── BERT_fine_tuning.ipynb      # BERT fine-tuning pipeline (uncased & pretrained variants)
├── README.md
```

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install the required dependencies:
```bash
pip install transformers datasets torch peft accelerate
```

> **Note**: A GPU is strongly recommended, especially for LLaMA fine-tuning. FP16 mixed-precision training is enabled by default to reduce memory usage.

---

## Usage

### BERT Fine-tuning
Open and run `BERT_fine_tuning.ipynb`. The notebook covers:
1. Loading and preprocessing the SQuAD dataset
2. Tokenization with WordPiece and answer span position mapping
3. Fine-tuning BERT with AdamW optimizer and learning rate warmup
4. Postprocessing and evaluation on EM and F1 metrics

### LLaMA Fine-tuning
Open and run `LLM_fine_tuning.ipynb`. The notebook covers:
1. Loading and preprocessing a SQuAD subset (2,000 examples)
2. Prompt engineering to frame extractive QA as a generation task
3. LoRA adapter configuration and FP16 fine-tuning
4. Postprocessing and evaluation on EM and F1 metrics

---

## Methods

### Optimization Techniques
| Technique | Applied To | Purpose |
|---|---|---|
| Learning Rate Warmup | BERT, LLaMA | Stabilize early training |
| LoRA | LLaMA | Parameter-efficient fine-tuning |
| FP16 Mixed Precision | BERT, LLaMA | Reduce memory, speed up training |
| Prompt Engineering | LLaMA | Frame extraction as generation |
| Postprocessing | BERT, LLaMA | Extract valid answer spans |

### BERT Pipeline
- **Input**: Context passage, question, answer span (SQuAD format)
- **Preprocessing**: `[CLS] question [SEP] context [SEP]` formatting, WordPiece tokenization, token-level span position mapping
- **Modelling**: BERT encoder + two linear layers predicting start/end token positions, trained with cross-entropy loss
- **Postprocessing**: Select highest-scoring valid (start, end) pair, decode token IDs back to text

### LLaMA Pipeline
- **Input**: Context passage, question (formatted as generative prompt)
- **Preprocessing**: Prompt template construction instructing the model to extract spans directly from context
- **Modelling**: Causal language model fine-tuned with LoRA adapters to generate answer spans
- **Postprocessing**: Verify generated text matches a contiguous span in source passage; normalize for evaluation

---

## References

- Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL.
- Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). arXiv.
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR.
- Rajpurkar et al. (2016). [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). EMNLP.
- Vaswani et al. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
- Micikevicius et al. (2018). [Mixed Precision Training](https://arxiv.org/abs/1710.03740). ICLR.
- Loshchilov & Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR.
