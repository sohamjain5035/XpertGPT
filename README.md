# XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling

<p align="center">
  <img src="https://img.shields.io/badge/Task-BabyLM%202026-blue.svg">
  <img src="https://img.shields.io/badge/Parameters-52.6M-green.svg">
  <img src="https://img.shields.io/badge/Active%20Params%2FToken-32.14M-orange.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

This repository contains the official PyTorch implementation, training scripts, and pretrained models for **XpertGPT**, a sparse decoder-only Transformer designed for data-constrained language modeling. This work was accepted at the **BabyLM Challenge 2026 (Strict-Small Track)**.

📄 **[Read the Full Paper](./XpertGPT_Paper.pdf)**

---

## 🌟 Overview

Conventional dense Transformers waste capacity by activating every parameter uniformly across every token, enforcing a single fixed receptive field. This fixed receptive field is fundamentally mismatched to the multi-scale nature of language (where syntactic dependencies are local, but semantic tracking spans long contexts). Under data-constrained regimes like the BabyLM Strict-Small budget (10 million words), parameter utilization must be highly efficient.

**XpertGPT** addresses this mismatch by routing tokens to specialized expert pathways operating at multiple contextual scales, improving parameter utilization without inflating computational costs.

### Key Architectural Features:
1. **Global Attention Block**: A dense stream operating across the entire sequence length that preserves full-sequence contextual information and tracks long-range dependencies.
2. **Sparse Multi-Scale Expert Stage**: Four parallel sparse expert branches that operate over different sliding-window attention spans (64, 16, 8, and 4 tokens).
3. **Expert-Choice Routing**: Instead of tokens selecting experts, experts independently select their top-k most relevant tokens. This guarantees perfectly balanced expert utilization by construction *without* auxiliary load-balancing losses.
4. **RoPE & SwiGLU**: Applies Rotary Position Embeddings (RoPE) and SwiGLU gated feed-forward networks throughout both the dense and sparse pathways to maximize representational capacity and relative positional stability.

XpertGPT achieves an overall average of **38.42**, outperforming the official dense GPT-2 (98.4M) baseline (37.38) while executing at **32.9 GFLOPs** per forward pass (a 67.3% reduction in active parameters and compute relative to the baseline).

---

## 🚀 Quick Start

### Installation

Clone the repository and install the dependencies. The code requires `torch` and `transformers`.

```bash
git clone https://github.com/sohamjain5035/XpertGPT.git
cd XpertGPT
pip install torch transformers datasets tokenizers
```

### How to Load and Use Checkpoints (Bypass Retraining)

**Important:** This is the exact and only supported way to load the pretrained XpertGPT models directly from the Hugging Face Hub.

#### A. Loading the Final Model (main branch)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "anonym5035/swi_glu_sw_64_16_8_4_xpert_gpt",
    revision="main",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "anonym5035/swi_glu_sw_64_16_8_4_xpert_gpt",
    revision="main"
)

# Forward pass example
inputs = tokenizer("Language naturally exhibits hierarchical structure across multiple", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
print(f"Logits shape: {logits.shape}") # (1, sequence_length, 16384)
```

#### B. Loading an Intermediate Milestone (e.g., chck_5M)
If you wish to analyze the model during training phases, you can load intermediate milestones by changing the revision hash:
```python
model_5m = AutoModelForCausalLM.from_pretrained(
    "anonym5035/swi_glu_sw_64_16_8_4_xpert_gpt",
    revision="chck_5M",
    trust_remote_code=True
).eval()
```

### Pretraining Pipeline

If you wish to train from scratch, the repository includes our full pretraining and evaluation pipeline (`train.py`) tailored for the BabyLM Challenge. It automatically handles tokenization, curriculum batching (causal and masked modeling ratios), and intermittent evaluations.

```bash
python train.py
```

---

## 📊 Comprehensive Results

All models were trained strictly under the BabyLM 2026 Strict-Small data constraint (10M words, 10 epochs). 

### 1. Zero-Shot NLP Benchmarks
Evaluated via causal log-likelihood scoring. XpertGPT demonstrates clear advantages on EWoK and Entity Tracking, showing that multi-scale sparse mixtures better adapt to low-frequency tail constructions and core world-knowledge interactions than fixed-width contexts.

| Benchmark | GPT-2 Baseline (98.4M) | XpertGPT (Seed 42) | XpertGPT (Mean ±σ, n=4) |
| :--- | :---: | :---: | :---: |
| BLiMP | **65.23** | 64.66 | 63.80 ± 0.76 |
| BLiMP Supp | 57.25 | **61.00** | 59.51 ± 1.14 |
| EWoK | 50.63 | **51.34** | 50.91 ± 1.27 |
| Entity Tracking | 19.10 | **21.08** | 20.02 ± 0.76 |
| COMPS | **51.81** | 49.62 | 50.23 ± 0.50 |
| GlobalPIQA | **35.09** | 32.65 | 33.03 ± 1.46 |

### 2. Fine-Tuned SuperGLUE Tasks & Averages
Evaluated using the official BabyLM pipeline's default hyperparameters. XpertGPT maintains an edge on shorter sentence-classification contexts (MRPC, QQP, WSC, BoolQ).

| Metric | GPT-2 Baseline (98.4M) | **XpertGPT (52.6M)** |
| :--- | :---: | :---: |
| **Overall Average** | 37.38 | **38.42** |
| **NLP Average** | 48.99 | **49.04** |
| BoolQ | **67.71** | 64.59 |
| MNLI | **49.84** | 48.55 |
| MRPC | 81.37 | **82.74** |
| MultiRC | **65.76** | 57.55 |
| QQP | 61.67 | **62.30** |
| RTE | 56.83 | **57.55** |
| WSC | 63.46 | **67.31** |

### 3. Computational Efficiency
XpertGPT yields substantial compute savings. With capacity factor $c = 2.0$ and $E = 4$ experts, Expert-Choice routing activates an average of two experts per token.

| Efficiency Metric | GPT-2 Baseline | XpertGPT |
| :--- | :---: | :---: |
| Total Parameters | 98.40M | **52.6M** |
| Active Parameters / Token | 98.40M | **32.14M** |
| Forward Pass Compute ($T=512$) | 100.8 GFLOPs | **32.9 GFLOPs** |
| Total Pretraining Compute (1 Epoch) | 5,904 TFLOPs | **1,928.4 TFLOPs** |
| Step Execution Time ($\Delta t$) | - | **82.8 ms** |
| Training Throughput | - | **98,937 tok/s** |

### 4. Ablation Studies
Our ablations highlight the trade-offs between global representation stability and local specialization. Removing the dense global stream (Ablation 2) causes wide-spread collapse, while removing routing entirely (Ablation 4) maximizes short-range acceptability at the severe expense of macro-reasoning (GlobalPIQA).

| Framework | BLiMP | BLiMP Supp | EWoK | Entity Track | COMPS | GlobalPIQA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| (1) Global Attention Only | 61.73 | 57.32 | 49.96 | 18.68 | 50.17 | 33.12 |
| (2) Routed 4-Expert Only | 55.72 | 51.00 | 49.92 | 17.49 | 49.63 | **35.64** |
| (3) Random Dynamic Routing | 63.36 | 57.91 | 49.77 | 19.56 | **50.26** | 31.20 |
| (4) All Active Experts (No Gating) | **66.70** | 59.98 | **51.57** | 20.46 | 51.46 | 29.70 |
| (5) **XpertGPT (Final Model)** | 64.66 | **61.00** | 51.34 | **21.08** | 49.62 | 32.65 |

---

## 🧠 Architecture Configuration Details

The default configuration for the 52.6M parameter model as used in the paper:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Layers ($L$) | 6 | Total number of XpertGPT blocks |
| Model dimension ($d_{model}$) | 256 | Dimension of the residual stream |
| Expert dimension ($d_{expert}$) | 384 | Dimension inside the expert pathways |
| Global attention heads | 4 | Number of heads in the dense global stream |
| Expert attention heads | 6 | Number of heads in the sparse expert branches |
| Total Experts ($E$) | 4 | Number of parallel sliding-window experts |
| Expert Window Sizes | [64, 16, 8, 4] | Attention spans for the 4 expert branches |
| Capacity factor ($c$) | 2.0 | Multiplier for Expert-Choice routing assignments |
| Context length ($T$) | 512 | Maximum sequence length |
| Vocabulary size | 16,384 | Sub-word BPE vocabulary |

## 📖 Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@inproceedings{jain2026xpertgpt,
  title={XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling},
  author={Jain, Soham and Singh, Harsh and Dewan, Divija and Dev, Atul},
  booktitle={Proceedings of the BabyLM Challenge},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
