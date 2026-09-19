# XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling

<p align="center">
  <img src="https://img.shields.io/badge/Task-BabyLM%202026-blue.svg">
  <img src="https://img.shields.io/badge/Parameters-52.6M-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

This repository contains the official PyTorch implementation and pretrained models for **XpertGPT**, a sparse decoder-only Transformer designed for data-constrained language modeling, accepted at the **BabyLM Challenge 2026 (Strict-Small Track)**.

📄 **[Read the Paper](./XpertGPT_Paper.pdf)**

---

## 🌟 Overview

Conventional dense Transformers waste capacity by activating every parameter uniformly across every token, enforcing a single fixed receptive field. Under data-constrained regimes like the BabyLM Strict-Small budget (10M words), parameter utilization must be highly efficient.

**XpertGPT** solves this by routing tokens to specialized expert pathways operating at multiple contextual scales. 

### Key Architectural Features:
1. **Global Attention Block**: A dense stream that preserves full-sequence contextual information and tracks long-range dependencies.
2. **Sparse Multi-Scale Expert Stage**: Four parallel sparse expert branches that operate over different sliding-window attention spans (64, 16, 8, and 4 tokens).
3. **Expert-Choice Routing**: Guarantees perfectly balanced expert utilization *without* auxiliary load-balancing losses by having each expert select its top-k most relevant tokens independently across the sequence.
4. **RoPE & SwiGLU**: Applies Rotary Position Embeddings and SwiGLU gated feed-forward networks throughout both the dense and sparse pathways to maximize representational capacity.

XpertGPT achieves an overall average of **38.42**, outperforming the official dense GPT-2 (98.4M) baseline (37.38) while activating only a fraction of the parameters per token (~32M active parameters).

## 🚀 Quick Start

### Installation

Clone the repository and ensure you have `transformers` and `torch` installed.

```bash
git clone https://github.com/sohamjain5035/XpertGPT.git
cd XpertGPT
pip install torch transformers
```

### Usage with Hugging Face `transformers`

XpertGPT is fully integrated with Hugging Face's `AutoModel` API through custom configuration classes. 

```python
import torch
from transformers import AutoTokenizer
from modeling_xpertgpt import XpertGPTForCausalLM
from configuration_xpertgpt import XpertGPTConfig

# 1. Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("./") # Point to local directory with tokenizer files

# 2. Load model configuration (matches Strict-Small setup)
config = XpertGPTConfig(
    vocab_size=16384,
    d_model=256,
    d_thin=384,
    num_layers=6,
    num_blocks=4,
    capacity_factor=2.0
)

# 3. Initialize model
model = XpertGPTForCausalLM(config)

# Forward pass example
inputs = tokenizer("The cat sat on the", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
print(f"Logits shape: {logits.shape}") # (1, 5, 16384)
```

## 🧠 Architecture Configuration

The default configuration for the 52.6M parameter model:

| Parameter | Value |
| :--- | :--- |
| Layers (L) | 6 |
| Model dimension (d_model) | 256 |
| Expert dimension (d_expert) | 384 |
| Global attention heads | 4 |
| Expert attention heads | 6 |
| Total Experts (E) | 4 |
| Expert Window Sizes | [64, 16, 8, 4] |
| Capacity factor (c) | 2.0 |
| Context length (T) | 512 |
| Vocabulary size | 16,384 |

## 📊 Results on BabyLM 2026 (Strict-Small)

| Benchmark | GPT-2 Baseline (98.4M) | **XpertGPT (52.6M)** |
| :--- | :---: | :---: |
| **Overall Average** | 37.38 | **38.42** |
| NLP Average | 48.99 | **49.04** |
| BLiMP Supplement | 57.25 | **61.00** |
| EWoK | 50.63 | **51.34** |
| Entity Tracking | 19.10 | **21.08** |

*Note: For the full breakdown of SuperGLUE fine-tuning tasks and zero-shot evaluations, please refer to the paper.*

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
