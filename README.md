# XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling

Official implementation of **XpertGPT**, a sparse decoder-only Transformer designed for sample-efficient language modeling under developmentally plausible, human-scale data constraints.

This model is trained and evaluated on the **BabyLM 2026 Strict-Small** corpus (approx. 10 million words), outperforming the official dense GPT-2 baseline across multiple zero-shot and fine-tuned benchmarks.

---

## 📖 Model Overview

Unlike conventional dense Transformers that activate all parameters uniformly and enforce a single receptive field, **XpertGPT** matches linguistic scale variations by pairing a dense global stream with parallel sparse experts operating at distinct contextual resolutions.

### Key Architecture Components
*   **Global Block**: A pre-LayerNorm multi-head attention block ($d_{\text{model}}=256, h=4$) with full-sequence causal context utilizing Rotary Position Embeddings (RoPE).
*   **Multi-Scale Information Transmission (MSIT) Experts**: Four parallel expert branches operating at $d_{\text{xpert}}=384$ over sliding-window attention spans:
    $$\text{Window Sizes } [w_1, w_2, w_3, w_4] = [64, 16, 8, 4] \text{ tokens}$$
*   **Expert-Choice Routing**: A load-balanced routing mechanism (capacity factor $c=2.0$) where experts select their Top-$k$ tokens. Balanced expert utilization is guaranteed by construction without requiring any auxiliary load-balancing losses.
*   **SwiGLU Activations**: Leveraged across all Feed-Forward Networks (FFN) to optimize convergence and representational capacity:
    $$\text{FFN}_{\text{SwiGLU}}(x) = \left(\text{SiLU}(W_1x) \otimes W_2x\right) W_3$$

---

## 📊 Evaluation Results

Both models are evaluated on the official **BabyLM 2026 Strict-Small** benchmark split.

### Zero-Shot NLP Benchmarks & Averages
Evaluates causal log-likelihoods across syntax, grammatical acceptability, and world-knowledge tasks. Overall averages are reported at the top.

| Benchmark / Metric | GPT-2 Baseline | **XpertGPT (Seed 42)** | **XpertGPT (Mean ± σ)** |
| :--- | :---: | :---: | :---: |
| **Overall Average** | 37.38 | **38.42** | - |
| **NLP Average** | 48.99 | **49.04** | - |
| **BLiMP** | **65.23** | 64.66 | 63.80 ± 0.76 |
| **BLiMP Supp** | 57.25 | **61.00** | 59.51 ± 1.14 |
| **EWoK** | 50.63 | **51.34** | 50.91 ± 1.27 |
| **Entity Tracking** | 19.10 | **21.08** | 20.02 ± 0.76 |
| **COMPS** | **51.81** | 49.62 | 50.23 ± 0.50 |
| **GlobalPIQA** | **35.09** | 32.65 | 33.03 ± 1.46 |

### Fine-Tuned SuperGLUE Classification
Evaluates fine-tuning accuracy across the seven SuperGLUE benchmarks.

| Benchmark | GPT-2 Baseline (98.4M) | **XpertGPT (52.6M)** |
| :--- | :---: | :---: |
| BoolQ | **67.71** | 64.59 |
| MNLI | **49.84** | 48.55 |
| MRPC | 81.37 | **82.74** |
| MultiRC | **65.76** | 57.55 |
| QQP | 61.67 | **62.30** |
| RTE | 56.83 | **57.55** |
| WSC | 63.46 | **67.31** |

### Computational and Step Efficiency
Through Expert-Choice routing ($c=2.0$, $E=4$), each token activates exactly $2$ expert pathways, leading to **67.3% fewer active parameters** per forward pass.

| Efficiency Metric | GPT-2 Baseline | **XpertGPT** | **Saving (%)** |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 98.40M | 52.64M | -46.5% |
| **Active Parameters / Token** | 98.40M | 32.14M | **-67.3%** |
| **Forward Pass FLOPs (T=512)** | 1.01 GFLOPs | 0.33 GFLOPs | **-67.3%** |
| **Pretraining FLOPs (D=10M)** | 6.06 TFLOPs | 1.98 TFLOPs | **-67.3%** |
| **Training Throughput** | - | **98,937 tokens/s** | - |

---

## ⚙️ Hyperparameter Configuration

The complete structural and optimizer hyperparameters used for pretraining on the 10M word Strict-Small corpus are listed below.

| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Layers ($L$)** | 6 | **Optimizer** | AdamW |
| **Model Dim ($d_{\text{model}}$)** | 256 | **Optimizer $\beta_1, \beta_2$** | $0.9, 0.95$ |
| **Expert Dim ($d_{\text{xpert}}$)** | 384 | **Learning Rate** | $3 \times 10^{-4}$ |
| **Global Heads ($h_g$)** | 4 | **Weight Decay ($\lambda$)** | $0.1$ |
| **Expert Heads ($h_e$)** | 6 | **Warmup Steps** | 800 |
| **Active Experts ($E$)** | 4 | **Epochs** | 10 |
| **Capacity Factor ($c$)** | 2.0 | **Batch Size ($B$)** | 16 |
| **Window Sizes ($w$)** | `[64, 16, 8, 4]` | **Context Length ($T$)** | 512 |

---

## 🚀 How to Load and Use Checkpoints

Pretrained checkpoint weights can be loaded directly from Hugging Face using the Transformers library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "SRJ5035/swi_glu_sw_64_16_8_4_xpert_gpt"

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="main",
    trust_remote_code=True
).eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="main")
```

---

## 📝 Citation

If you use this model or code in your research, please cite:

```bibtex
@misc{jain2026xpertgpt,
  title        = {XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling},
  author       = {Soham Jain and Harsh Singh and Divija Dewan and Atul Dev},
  year         = {2026},
  howpublished = {GitHub Repository},
  note         = {Vision and Language Group, IIT Roorkee}
}
```
