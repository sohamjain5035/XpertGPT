import os
import math
import random
import inspect
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# ─────────────────────────────────────────────────────────────
# Configuration Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class XpertGPTModelConfig:
    vocab_size: int = 16384
    block_size: int = 512
    d_model: int = 256
    d_thin: int = 384
    num_layers: int = 6
    num_blocks: int = 4
    capacity_factor: float = 2.0
    dropout: float = 0.1

class XpertGPTConfig(PretrainedConfig):
    model_type = "xpertgpt"
    auto_map = {
        "AutoConfig": "configuration_xpertgpt.XpertGPTConfig",
        "AutoModel": "modeling_xpertgpt.XpertGPTModelWrapper",
        "AutoModelForCausalLM": "modeling_xpertgpt.XpertGPTForCausalLM"
    }

    def __init__(
        self,
        vocab_size: int = 16384,
        block_size: int = 512,
        d_model: int = 256,
        d_thin: int = 384,
        num_layers: int = 6,
        num_blocks: int = 4,
        capacity_factor: float = 2.0,
        dropout: float = 0.1,
        **kwargs
    ):
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("bos_token_id", 2)  # [CLS]
        kwargs.setdefault("eos_token_id", 3)  # [SEP]
        kwargs.setdefault("pad_token_id", 1)  # [PAD]

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.d_thin = d_thin
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.capacity_factor = capacity_factor
        self.dropout = dropout
        
        # Attribute parity for classification heads
        self.hidden_size = d_model
        self.num_hidden_layers = num_layers

        super().__init__(**kwargs)

# ─────────────────────────────────────────────────────────────
# ROPE HELPERS
# ─────────────────────────────────────────────────────────────

def _precompute_rope_freqs(head_dim: int, seq_len: int, device: torch.device, theta: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be divisible by 2 for RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    L = x.size(2)
    cos = cos[:L, :].unsqueeze(0).unsqueeze(1) 
    sin = sin[:L, :].unsqueeze(0).unsqueeze(1) 
    
    half_dim = x.size(-1) // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotated_x * sin)

# ─────────────────────────────────────────────────────────────
# 1.  SLIDING WINDOW ATTENTION
# ─────────────────────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size=None):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads  = num_heads
        self.window_size = window_size
        self.head_dim   = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor,
                past_kv=None,
                use_cache: bool = False,
                bidirectional: bool = False):
        B, L, D = x.size()

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            q_cos, q_sin = _precompute_rope_freqs(self.head_dim, past_len + L, x.device)
            q = _apply_rope(q, q_cos[past_len:, :], q_sin[past_len:, :])
            k = _apply_rope(k, q_cos[past_len:, :], q_sin[past_len:, :])
            
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        else:
            cos, sin = _precompute_rope_freqs(self.head_dim, L, x.device)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        L_kv = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if bidirectional:
            if self.window_size is not None:
                past_len = L_kv - L
                pos_i = (past_len + torch.arange(L, device=x.device)).unsqueeze(1)
                pos_j = torch.arange(L_kv, device=x.device).unsqueeze(0)
                dist  = torch.abs(pos_i - pos_j)
                win_mask = dist < self.window_size
                scores = scores.masked_fill(
                    ~win_mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )
        else:
            past_len = L_kv - L
            pos_i = (past_len + torch.arange(L, device=x.device)).unsqueeze(1)
            pos_j = torch.arange(L_kv, device=x.device).unsqueeze(0)
            dist  = pos_i - pos_j

            causal_mask = dist >= 0
            if self.window_size is not None:
                causal_mask = causal_mask & (dist < self.window_size)

            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)

        if use_cache:
            if self.window_size is not None:
                present_kv = (
                    k[:, :, -self.window_size:, :],
                    v[:, :, -self.window_size:, :]
                )
            else:
                present_kv = (k, v)
        else:
            present_kv = None

        return out, present_kv

# ─────────────────────────────────────────────────────────────
# 2.  MSIT BRANCH BLOCK (Pre-Norm)
# ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Keep parameter count comparable to regular FFN with dim * 4:
        # 3 * dim * hidden_dim ~= 8 * dim^2  =>  hidden_dim = 8/3 * dim
        hidden_dim = int(dim * 8 / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

class MSITBranchBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = SlidingWindowAttention(dim, num_heads, window_size)
        self.ln2  = nn.LayerNorm(dim)
        self.ffn  = SwiGLU(dim)
        self.ln3  = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                past_kv=None,
                use_cache: bool = False,
                bidirectional: bool = False):
        attn_out, present_kv = self.attn(
            self.ln1(x), past_kv, use_cache, bidirectional
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        x = self.ln3(x)
        return x, present_kv

# ─────────────────────────────────────────────────────────────
# 3.  MoEP-MSIT ARCHITECTURE BLOCK  (Expert Choice routing)
# ─────────────────────────────────────────────────────────────

class MoEPMSITBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_thin: int = 192, num_blocks: int = 14,
                 capacity_factor: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.d_thin = d_thin
        self.num_blocks = num_blocks
        self.capacity_factor = capacity_factor

        # 1. Global Block (Dense, d_model)
        num_heads_global = max(1, d_model // 64)
        self.global_block = MSITBranchBlock(d_model, num_heads_global, window_size=None)

        # 2. Router
        self.router_ln = nn.LayerNorm(d_model)
        self.w_router = nn.Linear(d_model, num_blocks, bias=False)

        # 3. Shrink Projection
        self.w_down = nn.Linear(d_model, d_thin, bias=False)

        # 4. Thin Parallel Blocks (d_thin)
        self.windows = [64, 16, 8, 4] + [None] * (num_blocks - 4)
        self.heads = [max(1, d_thin // 64)] * num_blocks  
        
        self.thin_blocks = nn.ModuleList([
            MSITBranchBlock(d_thin, self.heads[i], self.windows[i])
            for i in range(num_blocks)
        ])

        # 6. Grow Projection
        self.w_up = nn.Linear(d_thin, d_model, bias=False)
        self.last_topk_indices = None
        self.ln_post_moe = nn.LayerNorm(d_model)

    def forward(self, x_0: torch.Tensor, past_kvs=None, use_cache: bool = False, bidirectional: bool = False):
        B, T, D = x_0.size()
        n_tokens = B * T

        # Step 1: Global Block
        pkv_g = past_kvs[0] if past_kvs else None
        x_1, nkv_g = self.global_block(x_0, pkv_g, use_cache, bidirectional)

        # Step 2: Gated input stream
        x_2 = x_1

        # Router scores
        r_logits = self.w_router(self.router_ln(x_2)).view(n_tokens, self.num_blocks)
        r_probs = F.softmax(r_logits, dim=-1)

        # Per-expert capacity: k = (n * c) / e
        k_capacity = max(1, int(round(n_tokens * self.capacity_factor / self.num_blocks)))
        k_capacity = min(k_capacity, n_tokens)

        # Expert Choice routing: topk over the token axis for each expert
        expert_token_scores = r_probs.transpose(0, 1) # (num_blocks, n_tokens)
        topk_scores, topk_token_idx = torch.topk(expert_token_scores, k_capacity, dim=-1)
        self.last_topk_indices = topk_token_idx

        # Load balancing is guaranteed by construction in Expert Choice
        layer_aux_loss = x_2.new_zeros(())

        # Shrink projection
        x_2_thin = self.w_down(x_2) # (B, T, d_thin)
        x_2_thin_flat = x_2_thin.view(n_tokens, self.d_thin)

        # Expert computations
        new_kvs = [nkv_g]
        expert_outputs_flat = torch.zeros(n_tokens, self.d_model, device=x_2.device, dtype=x_2.dtype)

        for i, block in enumerate(self.thin_blocks):
            sel_idx = topk_token_idx[i]
            bucket_in = x_2_thin_flat[sel_idx].unsqueeze(0) # (1, k_capacity, d_thin)

            pkv_i = past_kvs[i + 1] if past_kvs else None
            bucket_out, nkv_i = block(bucket_in, pkv_i, use_cache, bidirectional)
            if use_cache:
                new_kvs.append(nkv_i)

            bucket_out = bucket_out.squeeze(0) # (k_capacity, d_thin)
            bucket_out_full = self.w_up(bucket_out) # (k_capacity, d_model)

            gate = topk_scores[i].unsqueeze(-1) # (k_capacity, 1)
            expert_outputs_flat.index_add_(0, sel_idx, bucket_out_full * gate)

        x_3_full = expert_outputs_flat.view(B, T, self.d_model)
        out = x_2 + x_3_full
        out = self.ln_post_moe(out)

        present_kvs = tuple(new_kvs) if use_cache else None
        return out, present_kvs, layer_aux_loss

# ─────────────────────────────────────────────────────────────
# 4.  RAW XpertGPT MODEL
# ─────────────────────────────────────────────────────────────

class XpertGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte      = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([
            MoEPMSITBlock(
                d_model=cfg.d_model, 
                d_thin=cfg.d_thin, 
                num_blocks=cfg.num_blocks, 
                capacity_factor=cfg.capacity_factor
            )
            for _ in range(cfg.num_layers)
        ])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, bidirectional: bool = False):
        x = self.drop_emb(self.wte(input_ids))
        total_aux_loss = 0.0
        
        for block in self.blocks:
            x, _, layer_aux = block(x, past_kvs=None, use_cache=False, bidirectional=bidirectional)
            total_aux_loss += layer_aux

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1), ignore_index=-100)
            avg_aux_loss = total_aux_loss / self.cfg.num_layers
            loss = ce_loss + (0.01 * avg_aux_loss)

        return logits, loss

# ─────────────────────────────────────────────────────────────
# 5.  HUGGING FACE MODEL WRAPPERS
# ─────────────────────────────────────────────────────────────

class XpertGPTModelWrapper(PreTrainedModel):
    config_class = XpertGPTConfig
    base_model_prefix = "transformer"

    def __init__(self, config: XpertGPTConfig):
        super().__init__(config)
        self.wte      = nn.Embedding(config.vocab_size, config.d_model)
        self.drop_emb = nn.Dropout(config.dropout)
        self.blocks   = nn.ModuleList([
            MoEPMSITBlock(config.d_model, config.d_thin, config.num_blocks, config.capacity_factor)
            for _ in range(config.num_layers)
        ])
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.post_init()

    def forward(self, input_ids, **kwargs):
        x = self.drop_emb(self.wte(input_ids))
        for block in self.blocks:
            x, _, _ = block(x, past_kvs=None, use_cache=False, bidirectional=False)
        x = self.ln_f(x)
        return BaseModelOutputWithPast(last_hidden_state=x)


class XpertGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = XpertGPTConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["MoEPMSITBlock"]
    _tied_weights_keys = {"transformer.lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: XpertGPTConfig):
        super().__init__(config)
        self.transformer = XpertGPTModelWrapper(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        
        # State-dict pre-hook for backwards compatibility with checkpoint key naming
        def _prefix_cleaner(state_dict, prefix, local_metadata, Moore, missing_keys, unexpected_keys, error_msgs):
            keys = list(state_dict.keys())
            for k in keys:
                if k.startswith("transformer."):
                    state_dict[k.replace("transformer.", "", 1)] = state_dict.pop(k)
                elif f"{prefix}transformer." in k:
                    state_dict[k.replace("transformer.", "", 1)] = state_dict.pop(k)
                    
        self._register_load_state_dict_pre_hook(_prefix_cleaner)

    def tie_weights(self, **kwargs):
        if hasattr(self, "transformer") and hasattr(self.transformer, "wte") and hasattr(self.transformer, "lm_head"):
            self.transformer.wte.weight = self.lm_head.weight

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> CausalLMOutputWithPast:
        outputs = self.transformer(input_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


# Register with auto-mapping
AutoConfig.register("xpertgpt", XpertGPTConfig)
AutoModel.register(XpertGPTConfig, XpertGPTModelWrapper)
AutoModelForCausalLM.register(XpertGPTConfig, XpertGPTForCausalLM)
