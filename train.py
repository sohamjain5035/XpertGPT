import os
import math
import time
import json
import random
import inspect
import shutil
import subprocess
import argparse

# ─────────────────────────────────────────────────────────────
# 1. CORE PIPELINE FUNCTION AND HELPERS
# ─────────────────────────────────────────────────────────────

def is_zero_shot_cache_valid(cache_dir):
    if not os.path.exists(cache_dir):
        return False
    pred_count = 0
    for root, dirs, files in os.walk(cache_dir):
        if "predictions.json" in files:
            pred_count += 1
    return pred_count >= 3

def is_finetune_cache_valid(cache_dir):
    if not os.path.exists(cache_dir):
        return False
    pred_count = 0
    for root, dirs, files in os.walk(cache_dir):
        if "predictions.json" in files:
            pred_count += 1
    return pred_count >= 3

def verify_eval_run(path_to_check, description):
    print(f"[Eval] Verifying {description} path: {path_to_check}...")
    if not os.path.exists(path_to_check):
        raise RuntimeError(f"CRITICAL ERROR: {description} directory was NOT created at: {path_to_check}")
    
    # Check if predictions.json or results.txt or surprisal.json exists and is non-empty
    found_valid = False
    for root, dirs, files in os.walk(path_to_check):
        for f in files:
            if f in ["predictions.json", "results.txt", "surprisal.json"]:
                file_path = os.path.join(root, f)
                if os.path.getsize(file_path) > 0:
                    found_valid = True
                    break
        if found_valid:
            break
            
    if not found_valid:
        raise RuntimeError(f"CRITICAL ERROR: {description} completed but no valid prediction/result files were written in: {path_to_check}")
    print(f"[Eval] Success! Verified {description} results are stored correctly.")

def run_pipeline(model_name: str, epochs: int = 10, skip_eval: bool = False, skip_aoa: bool = True, skip_glue: bool = False):
    # Configure persistent cache paths locally to avoid duplicate downloads
    os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
    os.environ["NLTK_DATA"] = os.path.abspath("./nltk_data")
    os.makedirs("./hf_cache", exist_ok=True)
    os.makedirs("./nltk_data", exist_ok=True)

    # Programmatic Hugging Face Hub Login if HF_TOKEN is in environment
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("[HF] Programmatic login successful using HF_TOKEN.")
        except Exception as e:
            print(f"[HF] Warning: Programmatic login failed: {e}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    
    # Import model architecture
    from modeling_xpertgpt import (
        XpertGPTModel,
        XpertGPTModelConfig,
        XpertGPTConfig
    )

    # Print GPU details
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[GPU] CUDA is available! Using GPU: {gpu_name}\n")
    else:
        print("\n[GPU] Warning: CUDA is NOT available! Running on CPU.\n")

    # ─────────────────────────────────────────────────────────────
    # GLOBAL HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────
    VOCAB_SIZE        = 16384          
    MASK_TOKEN_ID     = 16383          
    BLOCK_SIZE        = 512
    BATCH_SIZE        = 16
    GRAD_ACCUM_STEPS  = 1  # grad_acc_step = 1
    EPOCHS            = epochs
    LEARNING_RATE     = 3e-4  # lr = 3e-4
    LR_MIN            = LEARNING_RATE * 0.05
    WARMUP_STEPS      = 800  # warmup_steps = 800
    WEIGHT_DECAY      = 0.1
    GRAD_CLIP         = 1.0

    NUM_THIN_BLOCKS   = 4             
    EC_CAPACITY_FACTOR = 2.0           
    CAUSAL_RATIO      = 1 / 1        

    MASK_PROB_START   = 0.20
    MASK_PROB_END     = 0.10

    # Output directories locally
    model_dir = os.path.abspath(f"./checkpoints/{model_name}")
    os.makedirs(model_dir, exist_ok=True)
    local_results_dir = os.path.abspath(f"./results/{model_name}")
    os.makedirs(local_results_dir, exist_ok=True)



    # ─────────────────────────────────────────────────────────────
    # Helper: Save Hugging Face Compliant Checkpoint
    # ─────────────────────────────────────────────────────────────
    def save_hf_checkpoint(raw_model, checkpoint_dir_name, tokenizer):
        save_dir = os.path.join(model_dir, checkpoint_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[Checkpoint] Saving Hugging Face format checkpoint to '{save_dir}'...")
        
        # A. Convert state dict keys to CausalLM wrapper naming
        state_dict = raw_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("_orig_mod."):
                name = name[10:]
            if name.startswith("model."):
                name = name[6:]
            
            if name == "lm_head.weight":
                new_state_dict["lm_head.weight"] = v
            else:
                new_state_dict[f"transformer.{name}"] = v
                
        torch.save(new_state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        
        # B. Copy modeling.py and configuration.py
        shutil.copy("modeling_xpertgpt.py", os.path.join(save_dir, "modeling_xpertgpt.py"))
        shutil.copy("configuration_xpertgpt.py", os.path.join(save_dir, "configuration_xpertgpt.py"))
        
        # C. Create config.json
        config_dict = {
            "auto_map": {
                "AutoConfig": "configuration_xpertgpt.XpertGPTConfig",
                "AutoModel": "modeling_xpertgpt.XpertGPTModelWrapper",
                "AutoModelForCausalLM": "modeling_xpertgpt.XpertGPTForCausalLM"
            },
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "d_model": 256,        # d_model = 256
            "hidden_size": 256,    # hidden_size = 256
            "d_thin": 384,
            "num_layers": 6,
            "num_blocks": NUM_THIN_BLOCKS,
            "capacity_factor": EC_CAPACITY_FACTOR,
            "dropout": 0.1,
            "model_type": "xpertgpt"
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        # D. Save tokenizer config files
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]"
        )
        fast_tokenizer.save_pretrained(save_dir)
        print(f"[Checkpoint] Checkpoint '{checkpoint_dir_name}' successfully saved.")

    # ─────────────────────────────────────────────────────────────
    # Tokenizer Training
    # ─────────────────────────────────────────────────────────────
    def build_and_train_tokenizer(texts: list) -> Tokenizer:
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        
        vocab_path = os.path.join(model_dir, "bpe_vocab_16k.json")
        if os.path.exists(vocab_path):
            print(f"[Tokenizer] Loading trained BPE model layout from '{vocab_path}'...")
            return Tokenizer.from_file(vocab_path)
            
        print(f"[Tokenizer] Generating fresh HuggingFace BPE Tokenizer model with {VOCAB_SIZE} slots...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE, 
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save(vocab_path)
        print(f"[Tokenizer] Tokenizer training completed and saved to '{vocab_path}'.")
        return tokenizer

    # ─────────────────────────────────────────────────────────────
    # Data Loader Setup
    # ─────────────────────────────────────────────────────────────
    class DataLoaderLite:
        def __init__(self, B: int, T: int, texts: list, tokenizer: Tokenizer, name: str):
            self.B = B
            self.T = T
            
            print(f"[DataLoader:{name}] Tokenising dataset sequences...")
            all_ids = []
            for t in texts:
                if t.strip():
                    encoded = tokenizer.encode(t).ids
                    all_ids.extend(encoded)

            self.tokens = torch.tensor(all_ids, dtype=torch.long)
            self.chunk_size = B * T
            self.n_chunks = (len(self.tokens) - 1) // self.chunk_size
            self.indices = list(range(self.n_chunks))
            self.pos = 0
            self._shuffle()
            
            print(f"[DataLoader:{name}] Total tokens: {len(self.tokens):,} | Epoch steps: {self.n_chunks:,}")

        def _shuffle(self):
            random.shuffle(self.indices)
            self.pos = 0

        def steps_per_epoch(self) -> int:
            return self.n_chunks

        def next_batch(self):
            B, T = self.B, self.T
            if self.pos >= len(self.indices):
                self._shuffle()
                
            chunk_idx = self.indices[self.pos]
            self.pos += 1
            
            start_pos = chunk_idx * self.chunk_size
            temp = self.tokens[start_pos : start_pos + self.chunk_size + 1]
            
            x = temp[:-1].view(B, T)
            y = temp[1:].view(B, T)
            return x, y

    # ─────────────────────────────────────────────────────────────
    # Batch preparation and schedules
    # ─────────────────────────────────────────────────────────────
    def get_current_mask_prob(global_step: int, total_steps: int) -> float:
        ratio = min(1.0, global_step / total_steps)
        return MASK_PROB_START + ratio * (MASK_PROB_END - MASK_PROB_START)

    def prepare_causal_batch(x: torch.Tensor, y: torch.Tensor):
        return x, y, False

    def prepare_masked_batch(x: torch.Tensor, y: torch.Tensor, mask_prob: float, mask_token_id: int):
        B, T = x.size()
        mask = torch.rand(B, T, device=x.device) < mask_prob
        masked_x = x.clone()
        masked_x[mask] = mask_token_id

        targets = torch.full_like(y, -100)
        targets[mask] = y[mask]

        return masked_x, targets, True

    def get_hybrid_batch(train_loader: DataLoaderLite, global_step: int, total_steps: int, device: torch.device):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        if random.random() < CAUSAL_RATIO:
            input_ids, targets, bidir = prepare_causal_batch(x, y)
        else:
            mask_prob = get_current_mask_prob(global_step, total_steps)
            input_ids, targets, bidir = prepare_masked_batch(x, y, mask_prob, MASK_TOKEN_ID)

        return input_ids, targets, bidir

    def get_lr(it: int, total_steps: int) -> float:
        if it < WARMUP_STEPS:
            return LEARNING_RATE * (it + 1) / WARMUP_STEPS
        if it >= total_steps:
            return LR_MIN
        decay_ratio = (it - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
        coeff        = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return LR_MIN + coeff * (LEARNING_RATE - LR_MIN)

    # ─────────────────────────────────────────────────────────────
    # Dataset Preparation
    # ─────────────────────────────────────────────────────────────
    print("\n[Data] Loading BabyLM-2026-Strict-Small ...")
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
    all_text = list(ds['train']['text'])

    tokenizer = build_and_train_tokenizer(all_text)

    split       = int(len(all_text) * 0.95)
    train_texts = all_text[:split]
    val_texts   = all_text[split:]

    train_loader = DataLoaderLite(BATCH_SIZE, BLOCK_SIZE, train_texts, tokenizer, "train")

    chunks_per_epoch = train_loader.steps_per_epoch()
    steps_per_epoch  = chunks_per_epoch // GRAD_ACCUM_STEPS 
    total_steps      = steps_per_epoch * EPOCHS

    cfg   = XpertGPTModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    random.seed(42)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    model = XpertGPTModel(cfg).to(device)

    # ─────────────────────────────────────────────────────────────
    # Training Resume Check
    # ─────────────────────────────────────────────────────────────
    words_trained = 0
    next_milestone_idx = 0
    global_step = 0
    
    milestones = sorted(list(set([i * 1_000_000 for i in range(1, 11)] + [i * 10_000_000 for i in range(1, 11)])))
    
    resume_checkpoint_dir = None
    for idx in range(len(milestones) - 1, -1, -1):
        m = milestones[idx]
        ckpt_name = f"chck_{m // 1_000_000}M"
        ckpt_path = os.path.join(model_dir, ckpt_name)
        if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
            config_json_path = os.path.join(ckpt_path, "config.json")
            if os.path.exists(config_json_path):
                try:
                    with open(config_json_path, "r") as f:
                        saved_config = json.load(f)
                    if saved_config.get("d_model") == 256:
                        resume_checkpoint_dir = ckpt_path
                        next_milestone_idx = idx + 1
                        words_trained = m
                        global_step = words_trained // (BATCH_SIZE * BLOCK_SIZE)
                        print(f"[Training] Found existing milestone checkpoint '{ckpt_name}'. Resuming from step {global_step:,} ({words_trained:,} tokens trained)...")
                        break
                    else:
                        print(f"[Training] Found checkpoint '{ckpt_name}' but it has mismatch d_model={saved_config.get('d_model')}. Starting fresh.")
                except Exception as e:
                    pass

    # Load weights if resuming
    if resume_checkpoint_dir is not None:
        print(f"[Model] Loading weights from checkpoint '{resume_checkpoint_dir}'...")
        state_dict = torch.load(os.path.join(resume_checkpoint_dir, "pytorch_model.bin"), map_location=device)
        model_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("transformer."):
                name = name[12:]
            model_state_dict[name] = v
        model.load_state_dict(model_state_dict)

    # Check if final main model exists
    main_ckpt_path = os.path.join(model_dir, "main")
    if os.path.exists(os.path.join(main_ckpt_path, "pytorch_model.bin")):
        print("\n[Pipeline] Final checkpoint 'main' already exists. Skipping training phase and transitioning directly to evaluations!")
    else:
        # torch.compile
        try:
            model = torch.compile(model)
            print("[Model] torch.compile() successfully verified graph optimizations")
        except Exception as e:
            print(f"[Model] torch.compile() skipped ({e})")

        # Optimizer
        param_dict     = {n: p for n, p in model.named_parameters() if p.requires_grad}
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        groups = [
            {'params': decay_params,   'weight_decay': WEIGHT_DECAY},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_ok  = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and ('cuda' in device)
        optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        # ─────────────────────────────────────────────────────────────
        # Training Loop
        # ─────────────────────────────────────────────────────────────
        model.train()
        autocast_ctx = torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16, enabled=True)

        start_epoch = global_step // steps_per_epoch
        start_chunk = (global_step % steps_per_epoch) * GRAD_ACCUM_STEPS

        print(f"\n[Training] Starting XpertGPT MoEP training for {EPOCHS} epochs...")
        for epoch in range(start_epoch, EPOCHS):
            train_loader._shuffle()
            if epoch == start_epoch and start_chunk > 0:
                print(f"[Training] Fast-forwarding dataloader to chunk index {start_chunk}...")
                train_loader.pos = start_chunk

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            
            start_chunk_idx = start_chunk if epoch == start_epoch else 0
            for chunk_step in range(start_chunk_idx, chunks_per_epoch):
                t0 = time.perf_counter()

                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                input_ids, targets, bidir = get_hybrid_batch(train_loader, global_step, total_steps, device)
                words_trained += input_ids.numel()

                with autocast_ctx:
                    _, loss = model(input_ids, targets, bidirectional=bidir)
                scaled_loss  = loss / GRAD_ACCUM_STEPS
                loss_accum  += scaled_loss.item()
                scaled_loss.backward()

                # Optimizer Step
                if (chunk_step + 1) % GRAD_ACCUM_STEPS == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if "cuda" in device:
                        torch.cuda.synchronize()

                    dt = (time.perf_counter() - t0) * 1000
                    mode_tag = "MLM" if bidir else "CLM"
                    mask_p   = get_current_mask_prob(global_step, total_steps)
                    
                    current_step = (chunk_step + 1) // GRAD_ACCUM_STEPS
                    print(
                        f"[E{epoch+1:02d} {current_step:>5d}/{steps_per_epoch} G{global_step:>7d}|{mode_tag}] "
                        f"train={loss_accum:.4f}  mask={mask_p:.1%}  norm={norm:.3f}  lr={lr:.2e}  dt={dt:6.1f}ms  words={words_trained:,}"
                    )
                    loss_accum = 0.0
                    global_step += 1

                # Check if we passed a milestone for checkpointing
                if next_milestone_idx < len(milestones) and words_trained >= milestones[next_milestone_idx]:
                    milestone_val = milestones[next_milestone_idx]
                    if milestone_val < 10_000_000:
                        milestone_name = f"chck_{milestone_val // 1_000_000}M"
                    else:
                        milestone_name = f"chck_{(milestone_val // 10_000_000) * 10}M"
                    
                    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    save_hf_checkpoint(raw_model, milestone_name, tokenizer)
                    next_milestone_idx += 1

        # Save final model as 'main'
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        save_hf_checkpoint(raw_model, "main", tokenizer)
        print("\n[Training] Training phase complete!")

    if skip_eval:
        print("[Pipeline] Skipping evaluations phase as requested.")
        return

    # ─────────────────────────────────────────────────────────────
    # 2. RUN EVALUATION PIPELINE
    # ─────────────────────────────────────────────────────────────
    local_results_dir = os.path.abspath(f"./results/{model_name}")
    os.makedirs(local_results_dir, exist_ok=True)
    local_main_res = os.path.join(local_results_dir, "main")

    # Ensure clone_dir exists and has the global_piqa files
    clone_parent = os.path.abspath("./babylm_eval_repo")
    
    # Self-healing check for global_piqa presence
    has_global_piqa = False
    for potential_strict in [os.path.join(clone_parent, "babylm-eval", "strict"), os.path.join(clone_parent, "strict")]:
        if os.path.exists(os.path.join(potential_strict, "evaluation_pipeline", "global_piqa")):
            has_global_piqa = True
            break
            
    if not has_global_piqa:
        print("[Eval] Cloned repository does not contain global_piqa tasks.")
        print("[Eval] Deleting and cloning official main branch...")
        if os.path.exists(clone_parent):
            shutil.rmtree(clone_parent)
        subprocess.run([
            "git", "clone", "-b", "main",
            "https://github.com/babylm-org/babylm-eval.git",
            clone_parent
        ], check=True)

    # Determine strict_dir path dynamically
    if os.path.exists(os.path.join(clone_parent, "strict")):
        strict_dir = os.path.join(clone_parent, "strict")
    else:
        strict_dir = os.path.join(clone_parent, "babylm-eval", "strict")
        
    print(f"[Eval] Using strict directory: {strict_dir}")
    os.environ["PYTHONPATH"] = strict_dir

    def patch_evaluation_run_script(strict_dir):
        import pathlib
        run_file = os.path.join(strict_dir, "evaluation_pipeline", "sentence_zero_shot", "run.py")
        if not os.path.exists(run_file):
            print(f"[GlobalPIQA] Warning: {run_file} not found. Cannot patch.")
            return
            
        print(f"[GlobalPIQA] Patching local checkpoint loader in {run_file}...")
        with open(run_file, "r") as f:
            content = f.read()
            
        # Check if already patched
        if "Local checkpoint directory patch" in content:
            print("[GlobalPIQA] Script already patched.")
            return
            
        target_str = """def main():
    args = _parse_arguments()
    if args.images_path is not None:
        assert args.batch_size == 1, "Multimodal only works in batch size 1!"
    dataset = args.data_path.stem
    args.model_name = pathlib.Path(args.model_path_or_name).stem
    if args.revision_name is None:
        revision_name = "main"
    else:
        revision_name = args.revision_name"""
            
        patch_str = """def main():
    args = _parse_arguments()
    if args.images_path is not None:
        assert args.batch_size == 1, "Multimodal only works in batch size 1!"
    dataset = args.data_path.stem
    
    # Local checkpoint directory patch
    import os
    model_path = args.model_path_or_name
    args.model_name = pathlib.Path(model_path).stem
    revision_name = args.revision_name if args.revision_name else "main"
    
    if os.path.isdir(model_path):
        target_revision = args.revision_name if args.revision_name else "main"
        if os.path.exists(os.path.join(model_path, target_revision)):
            args.model_path_or_name = os.path.join(model_path, target_revision)
            args.revision_name = None"""
                
        if target_str in content:
            new_content = content.replace(target_str, patch_str)
            with open(run_file, "w") as f:
                f.write(new_content)
            print("[GlobalPIQA] Successfully patched run.py")
        else:
            print("[GlobalPIQA] Warning: Could not find target pattern in run.py. Manual patch may be needed.")

    # Patch sentence zero shot loader inside cloned repo
    patch_evaluation_run_script(strict_dir)

    print("[Eval] Stripping Windows-specific packages from requirements.txt...")
    req_file_path = os.path.join(strict_dir, "requirements.txt")
    if os.path.exists(req_file_path):
        with open(req_file_path, "r") as f:
            lines = f.readlines()
        with open(req_file_path, "w") as f:
            for line in lines:
                if "pywin" not in line.lower() and "wintypes" not in line.lower():
                    f.write(line)

    print("[Eval] Verifying and installing evaluation dependencies programmatically...")
    required_packages = {
        "nltk": "nltk",
        "pandas": "pandas",
        "statsmodels": "statsmodels",
        "sklearn": "scikit-learn",
        "scipy": "scipy"
    }
    for pkg_import, pkg_install in required_packages.items():
        try:
            __import__(pkg_import)
        except ImportError:
            print(f"[Eval] Package '{pkg_install}' not found. Installing it programmatically...")
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", pkg_install], check=True)
    
    print("[Eval] Downloading NLTK tokenizer resources...")
    import nltk
    nltk.download('punkt', download_dir=os.environ["NLTK_DATA"])
    nltk.download('punkt_tab', download_dir=os.environ["NLTK_DATA"])

    # Ensure standard zero-shot datasets are downloaded
    blimp_fast_dir = os.path.join(strict_dir, "evaluation_data", "fast_eval", "blimp_fast")
    if not os.path.exists(blimp_fast_dir) or not os.listdir(blimp_fast_dir):
        print("[Eval] Standard zero-shot datasets not found. Downloading...")
        subprocess.run(["python", "-m", "scripts.download_evals"], cwd=strict_dir, check=True)
        
        # Unzip EWoK fast
        ewok_zip = os.path.join(strict_dir, "evaluation_data/fast_eval/ewok_fast.zip")
        if os.path.exists(ewok_zip):
            print("[Eval] Unzipping EWoK fast data...")
            bad_nested_dir = os.path.join(strict_dir, "evaluation_data/fast_eval/evaluation_data")
            if os.path.exists(bad_nested_dir):
                shutil.rmtree(bad_nested_dir)
            subprocess.run(["unzip", "-o", "-P", "BabyLM2025", "evaluation_data/fast_eval/ewok_fast.zip", "-d", "."], cwd=strict_dir, check=True)
            
        # Download EWoK full
        print("[Eval] Downloading and filtering full EWoK dataset...")
        subprocess.run(["python", "-m", "evaluation_pipeline.ewok.dl_and_filter"], cwd=strict_dir, check=True)

    # Download GlobalPIQA dataset
    global_piqa_parallel_dir = os.path.join(strict_dir, "evaluation_data", "fast_eval", "global_piqa_parallel")
    if not os.path.exists(global_piqa_parallel_dir) or not os.listdir(global_piqa_parallel_dir):
        print("[Eval] GlobalPIQA dataset not found. Downloading...")
        subprocess.run(["python", "evaluation_pipeline/global_piqa/dl.py"], cwd=strict_dir, check=True)

    # Ensure all scripts are executable
    print("[Eval] Making evaluation shell scripts executable...")
    subprocess.run("chmod +x scripts/*.sh", shell=True, cwd=strict_dir, check=True)

    def run_task_with_cache(checkpoint, task, output_subpath, cmd):
        # Determine paths
        local_cache_path = os.path.join("./results", model_name, checkpoint, "zero_shot", "causal", task, output_subpath)
        if task == "reading":
            local_cache_path = os.path.join("./results", model_name, checkpoint, "zero_shot", "causal", "reading")
        elif task == "comps":
            local_cache_path = os.path.join("./results", model_name, checkpoint, "zero_shot", "causal", "comps", "comps")
            
        target_results_dir = os.path.join(strict_dir, "results", model_name, checkpoint, "zero_shot", "causal", task, output_subpath)
        if task == "reading":
            target_results_dir = os.path.join(strict_dir, "results", model_name, checkpoint, "zero_shot", "causal", "reading")
        elif task == "comps":
            target_results_dir = os.path.join(strict_dir, "results", model_name, checkpoint, "zero_shot", "causal", "comps", "comps")

        # If cached, copy it over
        cache_file = os.path.join(local_cache_path, "predictions.json")
        
        # Self-healing: invalidate old unfiltered entity_tracking caches
        if task == "entity_tracking" and os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, "r") as f:
                    preds = json.load(f)
                is_valid_cache = True
                for k, v in preds.items():
                    if len(v.get("predictions", [])) in [605, 606, 607, 615, 529, 156, 187, 159]:
                        is_valid_cache = False
                        break
                if not is_valid_cache:
                    print(f"[Eval] Cached entity_tracking for '{checkpoint}' has incorrect old sizes. Invalidate and re-run fresh...")
                    shutil.rmtree(local_cache_path, ignore_errors=True)
            except Exception:
                pass

        if os.path.exists(cache_file):
            print(f"[Eval] Task '{task}' ({output_subpath}) for checkpoint '{checkpoint}' is cached. Restoring...")
            if os.path.exists(target_results_dir):
                shutil.rmtree(target_results_dir)
            os.makedirs(target_results_dir, exist_ok=True)
            for item in os.listdir(local_cache_path):
                s = os.path.join(local_cache_path, item)
                d = os.path.join(target_results_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            return

        print(f"[Eval] Running task '{task}' ({output_subpath}) for checkpoint '{checkpoint}'...")
        subprocess.run(cmd, cwd=strict_dir, check=True)

        # Relocate from actual_model_basename, main, or checkpoint subfolder if needed
        actual_model_basename = os.path.basename(model_dir.rstrip("/"))
        possible_actual_path = os.path.join(strict_dir, "results", actual_model_basename, checkpoint, "zero_shot", "causal", task, output_subpath)
        if task == "reading":
            possible_actual_path = os.path.join(strict_dir, "results", actual_model_basename, checkpoint, "zero_shot", "causal", "reading")
        elif task == "comps":
            possible_actual_path = os.path.join(strict_dir, "results", actual_model_basename, checkpoint, "zero_shot", "causal", "comps", "comps")

        possible_main_path = os.path.join(strict_dir, "results", "main", checkpoint, "zero_shot", "causal", task, output_subpath)
        if task == "reading":
            possible_main_path = os.path.join(strict_dir, "results", "main", checkpoint, "zero_shot", "causal", "reading")
        elif task == "comps":
            possible_main_path = os.path.join(strict_dir, "results", "main", checkpoint, "zero_shot", "causal", "comps", "comps")

        possible_local_reading_path = os.path.join(strict_dir, "results", checkpoint, "main", "zero_shot", "causal", "reading")

        for p_path in [possible_actual_path, possible_main_path, possible_local_reading_path]:
            if os.path.exists(p_path) and p_path != target_results_dir:
                print(f"[Eval] Relocating results from {p_path} to {target_results_dir}...")
                if os.path.exists(target_results_dir):
                    shutil.rmtree(target_results_dir)
                os.makedirs(os.path.dirname(target_results_dir), exist_ok=True)
                shutil.move(p_path, target_results_dir)
                break

        # Verify
        verify_eval_run(target_results_dir, f"{checkpoint} {task} ({output_subpath})")

        # Save to local cache
        if os.path.exists(local_cache_path):
            shutil.rmtree(local_cache_path)
        os.makedirs(local_cache_path, exist_ok=True)
        for item in os.listdir(target_results_dir):
            s = os.path.join(target_results_dir, item)
            d = os.path.join(local_cache_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    def run_finetune_task_with_cache(task, cmd):
        local_cache_path = os.path.join("./results", model_name, "main", "finetune", task)
        target_results_dir = os.path.join(strict_dir, "results", model_name, "main", "finetune", task)

        if os.path.exists(os.path.join(local_cache_path, "predictions.json")):
            print(f"[Eval] GLUE task '{task}' is cached. Restoring...")
            if os.path.exists(target_results_dir):
                shutil.rmtree(target_results_dir)
            os.makedirs(target_results_dir, exist_ok=True)
            for item in os.listdir(local_cache_path):
                s = os.path.join(local_cache_path, item)
                d = os.path.join(target_results_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            return

        print(f"[Eval] Running GLUE task '{task}'...")
        subprocess.run(cmd, cwd=strict_dir, check=True)

        # Relocate from results/main/main if needed
        possible_main_path = os.path.join(strict_dir, "results", "main", "main", "finetune", task)
        if os.path.exists(possible_main_path) and possible_main_path != target_results_dir:
            print(f"[Eval] Relocating results from {possible_main_path} to {target_results_dir}...")
            if os.path.exists(target_results_dir):
                shutil.rmtree(target_results_dir)
            os.makedirs(os.path.dirname(target_results_dir), exist_ok=True)
            shutil.move(possible_main_path, target_results_dir)

        # Verify
        verify_eval_run(target_results_dir, f"GLUE task {task}")

        # Cache locally
        if os.path.exists(local_cache_path):
            shutil.rmtree(local_cache_path)
        os.makedirs(local_cache_path, exist_ok=True)
        for item in os.listdir(target_results_dir):
            s = os.path.join(target_results_dir, item)
            d = os.path.join(local_cache_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    def run_aoa_with_cache(cmd):
        local_cache_path = os.path.join("./results", model_name, "main", "aoa")
        target_results_dir = os.path.join(strict_dir, "results", model_name, "main", "aoa")

        if os.path.exists(os.path.join(local_cache_path, "aoa_score.json")) or os.path.exists(os.path.join(local_cache_path, "surprisal.json")):
            print(f"[Eval] AoA task is cached. Restoring...")
            if os.path.exists(target_results_dir):
                shutil.rmtree(target_results_dir)
            os.makedirs(target_results_dir, exist_ok=True)
            for item in os.listdir(local_cache_path):
                s = os.path.join(local_cache_path, item)
                d = os.path.join(target_results_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            return

        print("[Eval] Running AoA task...")
        subprocess.run(cmd, cwd=strict_dir, check=True)

        # Relocate from results/main/main if needed
        possible_main_path = os.path.join(strict_dir, "results", "main", "main", "aoa")
        if os.path.exists(possible_main_path) and possible_main_path != target_results_dir:
            print(f"[Eval] Relocating results from {possible_main_path} to {target_results_dir}...")
            if os.path.exists(target_results_dir):
                shutil.rmtree(target_results_dir)
            os.makedirs(os.path.dirname(target_results_dir), exist_ok=True)
            shutil.move(possible_main_path, target_results_dir)

        # Verify
        verify_eval_run(target_results_dir, "AoA task")

        # Cache locally
        if os.path.exists(local_cache_path):
            shutil.rmtree(local_cache_path)
        os.makedirs(local_cache_path, exist_ok=True)
        for item in os.listdir(target_results_dir):
            s = os.path.join(target_results_dir, item)
            d = os.path.join(local_cache_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    # ─────────────────────────────────────────────────────────────
    # B. FINAL MODEL 'main' FULL ZERO-SHOT EVALUATION
    # ─────────────────────────────────────────────────────────────
    main_ckpt_path = os.path.join(model_dir, "main")
    if os.path.exists(main_ckpt_path):
        print(f"[Eval] Running full zero-shot evaluation on main...")
        # blimp filtered
        run_task_with_cache(
            "main", "blimp", "blimp_filtered",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "blimp", "--data_path", "evaluation_data/full_eval/blimp_filtered", "--save_predictions"]
        )
        # supplement filtered
        run_task_with_cache(
            "main", "blimp", "supplement_filtered",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "blimp", "--data_path", "evaluation_data/full_eval/supplement_filtered", "--save_predictions"]
        )
        # ewok filtered
        run_task_with_cache(
            "main", "ewok", "ewok_filtered",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "ewok", "--data_path", "evaluation_data/full_eval/ewok_filtered", "--save_predictions"]
        )
        # entity tracking
        run_task_with_cache(
            "main", "entity_tracking", "entity_tracking",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "entity_tracking", "--data_path", "evaluation_data/full_eval/entity_tracking", "--save_predictions"]
        )
        # comps
        run_task_with_cache(
            "main", "comps", "comps",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "comps", "--data_path", "evaluation_data/full_eval/comps", "--save_predictions"]
        )
        # reading
        run_task_with_cache(
            "main", "reading", "reading",
            ["python", "-m", "evaluation_pipeline.reading.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--data_path", "evaluation_data/full_eval/reading/reading_data.csv"]
        )
        # global piqa parallel
        run_task_with_cache(
            "main", "global_piqa_parallel", "global_piqa_parallel",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "global_piqa_parallel", "--data_path", "evaluation_data/full_eval/global_piqa_parallel", "--save_predictions"]
        )
        # global piqa nonparallel
        run_task_with_cache(
            "main", "global_piqa_nonparallel", "global_piqa_nonparallel",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", main_ckpt_path, "--backend", "causal", "--task", "global_piqa_nonparallel", "--data_path", "evaluation_data/full_eval/global_piqa_nonparallel", "--save_predictions"]
        )

    # ─────────────────────────────────────────────────────────────
    # C. FINAL MODEL GLUE AND AOA EVALUATION
    # ─────────────────────────────────────────────────────────────
    if os.path.exists(main_ckpt_path):
        # 1. GLUE fine-tuning
        if skip_glue:
            print("[Eval] Skipping GLUE fine-tuning evaluations as requested.")
        else:
            print("[Eval] Running GLUE fine-tuning evaluations on main task-by-task...")
            glue_tasks = {
                "boolq": ["boolq", "16", "10"],
                "multirc": ["multirc", "16", "10"],
                "rte": ["rte", "32", "10"],
                "wsc": ["wsc", "32", "30"],
                "mrpc": ["mrpc", "32", "10"],
                "qqp": ["qqp", "32", "10"],
                "mnli": ["mnli", "32", "10"]
            }
            for task_name, (task, bsz, max_epochs) in glue_tasks.items():
                num_labels = "3" if task == "mnli" else "2"
                metric_for_valid = "accuracy"
                if task in ["mrpc", "qqp"]:
                    metric_for_valid = "f1"
                metrics = ["accuracy"]
                if task != "mnli":
                    metrics = ["accuracy", "f1", "mcc"]
                
                cmd = [
                    "python", "-m", "evaluation_pipeline.finetune.run",
                    "--model_name_or_path", main_ckpt_path,
                    "--train_data", f"evaluation_data/full_eval/glue_filtered/{task}.train.jsonl",
                    "--valid_data", f"evaluation_data/full_eval/glue_filtered/{task}.valid.jsonl",
                    "--predict_data", f"evaluation_data/full_eval/glue_filtered/{task}.valid.jsonl",
                    "--task", task,
                    "--num_labels", num_labels,
                    "--batch_size", bsz,
                    "--learning_rate", "3e-5",
                    "--num_epochs", max_epochs,
                    "--sequence_length", "512",
                    "--results_dir", "results",
                    "--save",
                    "--save_dir", "models",
                    "--metric_for_valid", metric_for_valid,
                    "--seed", "42",
                    "--verbose",
                    "--padding_side", "left",
                    "--take_final"
                ]
                cmd.append("--metrics")
                cmd.extend(metrics)
                
                run_finetune_task_with_cache(task_name, cmd)

        # 2. AoA
        if skip_aoa:
            print("[Eval] Skipping AoA evaluations as requested.")
        else:
            run_aoa_with_cache([
                "python", "-m", "evaluation_pipeline.AoA_word.run",
                "--model_name", model_dir,
                "--backend", "causal",
                "--track_name", "strict-small",
                "--word_path", "evaluation_data/full_eval/aoa/cdi_childes.json",
                "--output_dir", "results"
            ])

    # ─────────────────────────────────────────────────────────────
    # A. INTERMEDIATE CHECKPOINTS FAST EVALUATION
    # ─────────────────────────────────────────────────────────────
    print(f"[Eval] Running zero-shot fast evaluations on intermediate checkpoints...")
    checkpoints = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 110, 10)]
    
    eval_model_path = model_dir
    
    for checkpoint in checkpoints:
        ckpt_full_path = os.path.join(model_dir, checkpoint)
        if not os.path.exists(ckpt_full_path):
            continue
            
        # blimp fast
        run_task_with_cache(
            checkpoint, "blimp", "blimp_fast",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "blimp", "--data_path", "evaluation_data/fast_eval/blimp_fast", "--save_predictions", "--revision_name", checkpoint]
        )
        # supplement fast
        run_task_with_cache(
            checkpoint, "blimp", "supplement_fast",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "blimp", "--data_path", "evaluation_data/fast_eval/supplement_fast", "--save_predictions", "--revision_name", checkpoint]
        )
        # ewok fast
        run_task_with_cache(
            checkpoint, "ewok", "ewok_fast",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "ewok", "--data_path", "evaluation_data/fast_eval/ewok_fast", "--save_predictions", "--revision_name", checkpoint]
        )
        # entity tracking fast
        run_task_with_cache(
            checkpoint, "entity_tracking", "entity_tracking_fast",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "entity_tracking", "--data_path", "evaluation_data/fast_eval/entity_tracking_fast", "--save_predictions", "--revision_name", checkpoint]
        )
        # reading fast
        run_task_with_cache(
            checkpoint, "reading", "reading",
            ["python", "-m", "evaluation_pipeline.reading.run", "--model_path_or_name", ckpt_full_path, "--backend", "causal", "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv"]
        )
        # global piqa parallel fast
        run_task_with_cache(
            checkpoint, "global_piqa_parallel", "global_piqa_parallel",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "global_piqa_parallel", "--data_path", "evaluation_data/fast_eval/global_piqa_parallel", "--save_predictions", "--revision_name", checkpoint]
        )
        # global piqa nonparallel fast
        run_task_with_cache(
            checkpoint, "global_piqa_nonparallel", "global_piqa_nonparallel",
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", eval_model_path, "--backend", "causal", "--task", "global_piqa_nonparallel", "--data_path", "evaluation_data/fast_eval/global_piqa_nonparallel", "--save_predictions", "--revision_name", checkpoint]
        )

    # ─────────────────────────────────────────────────────────────
    # D. COLLATE RESULTS AND CLEANUP
    # ─────────────────────────────────────────────────────────────
    print("[Eval] Collating predictions into submission file...")
    # Clean collation destination in evaluation repo
    collate_results_dir = os.path.join(strict_dir, "results", model_name)
    if os.path.exists(collate_results_dir):
        shutil.rmtree(collate_results_dir)
    os.makedirs(os.path.dirname(collate_results_dir), exist_ok=True)
    
    # Copy from local cache results to strict results for collation
    shutil.copytree(local_results_dir, collate_results_dir)

    # Run collation
    subprocess.run([
        "python", "-m", "evaluation_pipeline.collate_preds",
        "--model_path_or_name", model_name,
        "--backend", "causal",
        "--track", "strict-small",
        "--fast"
    ], cwd=strict_dir, check=True)
    
    # Save results to local folder
    results_src = os.path.join(strict_dir, "results")
    results_dest = os.path.abspath("./results")
    if os.path.exists(results_dest):
        shutil.rmtree(results_dest)
    shutil.copytree(results_src, results_dest)
    
    # Copy final collated json to current folder
    collated_json = os.path.join(strict_dir, "all_full_preds_and_fast_scores_causal.json")
    if os.path.exists(collated_json):
        shutil.copy(collated_json, "./all_full_preds_and_fast_scores_causal.json")
        print("\n[Eval] Success! Collation completed! Final file is at './all_full_preds_and_fast_scores_causal.json'")

    print("\n[Eval] Pipeline evaluation run finished.")

def upload_pipeline(model_name, repo_name, token=None):
    import os
    import shutil
    import json
    import hashlib
    from huggingface_hub import HfApi, create_repo

    if not token:
        token = os.environ.get("HF_TOKEN")

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[HF] Authenticated successfully as user: {username}")
    except Exception as e:
        print(f"[HF] Authentication failed. Error: {e}")
        return

    repo_id = f"{username}/{repo_name}"
    print(f"[HF] Target Repository ID: {repo_id}")
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
        print(f"[HF] Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"[HF] Failed to verify or create repository. Error: {e}")
        return

    checkpoint_dir = os.path.abspath(f"./checkpoints/{model_name}")
    
    # Resolve the main checkpoint directory using self-healing rules
    revisions = {}
    if os.path.exists("pytorch_model.bin") and os.path.exists("config.json"):
        print("[HF] Detected weight and config files in the current working directory. Using current folder as 'main' checkpoint.")
        revisions = {"main": os.getcwd()}
    elif os.path.exists(os.path.join(checkpoint_dir, "main")):
        revisions = {"main": os.path.join(checkpoint_dir, "main")}
    elif os.path.exists(os.path.abspath("./checkpoints/msit_gptbert_fresh/main")):
        print("[HF] Using fallback checkpoint folder './checkpoints/msit_gptbert_fresh/main'...")
        revisions = {"main": os.path.abspath("./checkpoints/msit_gptbert_fresh/main")}
    elif os.path.exists(os.path.abspath("./checkpoints/main")):
        revisions = {"main": os.path.abspath("./checkpoints/main")}
    else:
        print(f"[HF] Error: Could not locate the 'main' checkpoint weights. Checked: {checkpoint_dir}/main, current directory, and fallbacks.")
        return

    # Check for intermediate checkpoints relative to the main checkpoint's parent folder
    main_dir = revisions["main"]
    parent_dir = os.path.dirname(main_dir)
    
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]:
        ckpt_name = f"chck_{m}M"
        ckpt_path = os.path.join(parent_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            revisions[ckpt_name] = ckpt_path
        else:
            # Check if there is a .pt file in the parent folder
            pt_path = os.path.join(parent_dir, f"{ckpt_name}.pt")
            if os.path.exists(pt_path):
                print(f"[HF] Found legacy checkpoint file '{ckpt_name}.pt'. Converting to HF format for upload...")
                import torch
                from transformers import AutoTokenizer
                from tokenizers import Tokenizer
                try:
                    from modeling_xpertgpt import XpertGPTForCausalLM, XpertGPTConfig
                    cfg = XpertGPTConfig(d_model=256, d_thin=384, num_layers=6, num_blocks=4)
                    model_to_save = XpertGPTForCausalLM(cfg)
                    sd = torch.load(pt_path, map_location="cpu")
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "")
                        clean_sd[new_k] = v
                    model_to_save.load_state_dict(clean_sd)
                    
                    vocab_path = os.path.join(parent_dir, "bpe_vocab_16k.json")
                    if not os.path.exists(vocab_path):
                        vocab_path = os.path.join(main_dir, "bpe_vocab_16k.json")
                    if os.path.exists(vocab_path):
                        tok = Tokenizer.from_file(vocab_path)
                    else:
                        tok = None
                        
                    os.makedirs(ckpt_path, exist_ok=True)
                    state_dict = model_to_save.state_dict()
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k
                        if name.startswith("_orig_mod."):
                            name = name[10:]
                        if name.startswith("model."):
                            name = name[6:]
                        if name == "lm_head.weight":
                            new_state_dict["lm_head.weight"] = v
                        else:
                            new_state_dict[f"transformer.{name}"] = v
                    torch.save(new_state_dict, os.path.join(ckpt_path, "pytorch_model.bin"))
                    
                    config_dict = {
                        "auto_map": {
                            "AutoConfig": "configuration_xpertgpt.XpertGPTConfig",
                            "AutoModel": "modeling_xpertgpt.XpertGPTModelWrapper",
                            "AutoModelForCausalLM": "modeling_xpertgpt.XpertGPTForCausalLM"
                        },
                        "vocab_size": 16384,
                        "block_size": 512,
                        "d_model": 256,
                        "hidden_size": 256,
                        "d_thin": 384,
                        "num_layers": 6,
                        "num_blocks": 4,
                        "capacity_factor": 2.0,
                        "dropout": 0.1,
                        "model_type": "xpertgpt",
                        "num_hidden_layers": 6
                    }
                    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
                        json.dump(config_dict, f, indent=2)
                        
                    if tok:
                        from transformers import PreTrainedTokenizerFast
                        fast_tokenizer = PreTrainedTokenizerFast(
                            tokenizer_object=tok,
                            bos_token="[CLS]",
                            eos_token="[SEP]",
                            unk_token="[UNK]",
                            pad_token="[PAD]",
                            mask_token="[MASK]"
                        )
                        fast_tokenizer.save_pretrained(ckpt_path)
                    
                    revisions[ckpt_name] = ckpt_path
                except Exception as ex:
                    print(f"[HF] Failed to convert legacy checkpoint file '{ckpt_name}.pt': {ex}")

    # Temporary directory for staging uploads
    temp_dir = os.path.abspath("./temp_hf_upload")
    
    # LICENSE text
    license_text = """Creative Commons Attribution-NonCommercial 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.

Section 1 -- Definitions.
a. Licensed Material means the artistic or literary work, database, or other material to which the Licensor applied this Public License.
b. Licensed Rights means the rights granted to You subject to the terms and conditions of this Public License, which are limited to all Copyright and Similar Rights that apply to Your use of the Licensed Material and that the Licensor has authority to license.
c. NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation.
d. Share means to provide material to the public by any means or process that requires permission under the Licensed Rights.
e. You means the individual or entity exercising the Licensed Rights under this Public License.

Section 2 -- Scope.
a. License grant.
   1. Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:
      A. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
      B. Produce, reproduce, and Share Adapted Material for NonCommercial purposes only.
   2. Attribution. As a condition of the license, You must attribute the Licensor and keep intact copyright notices.
"""

    for revision_name, local_path in revisions.items():
        print(f"\n[HF] Staging files for revision '{revision_name}' from '{local_path}'...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # 1. Copy weight file and save as model.safetensors if possible, otherwise pytorch_model.bin
        src_bin = os.path.join(local_path, "pytorch_model.bin")
        weight_file_dest = None
        if os.path.exists(src_bin):
            # Try to convert to safetensors
            try:
                import torch
                from safetensors.torch import save_file
                state_dict = torch.load(src_bin, map_location="cpu")
                # Clone tensors to break memory sharing (prevents shared weight memory error in safetensors)
                state_dict = {k: v.clone() for k, v in state_dict.items()}
                weight_file_dest = os.path.join(temp_dir, "model.safetensors")
                save_file(state_dict, weight_file_dest)
                print(f"[HF] Converted weights to safetensors format.")
            except Exception as e:
                print(f"[HF] Conversion to safetensors failed ({e}). Staging raw pytorch_model.bin...")
                weight_file_dest = os.path.join(temp_dir, "pytorch_model.bin")
                shutil.copy2(src_bin, weight_file_dest)
        else:
            print(f"[HF] Error: No weight file found in '{local_path}'!")
            continue

        # Calculate weight file hash
        sha256_hash = hashlib.sha256()
        with open(weight_file_dest, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        weight_hash = sha256_hash.hexdigest()
        print(f"[HF] Weight file SHA-256: {weight_hash}")

        # 2. Copy and patch config.json
        src_config = os.path.join(local_path, "config.json")
        if os.path.exists(src_config):
            with open(src_config, "r") as f:
                cfg_data = json.load(f)
            # Patch config
            cfg_data["auto_map"] = {
                "AutoConfig": "configuration_xpertgpt.XpertGPTConfig",
                "AutoModel": "modeling_xpertgpt.XpertGPTModelWrapper",
                "AutoModelForCausalLM": "modeling_xpertgpt.XpertGPTForCausalLM"
            }
            cfg_data["architectures"] = ["XpertGPTForCausalLM"]
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(cfg_data, f, indent=2)
        else:
            # Fallback configuration
            cfg_data = {
                "auto_map": {
                    "AutoConfig": "configuration_xpertgpt.XpertGPTConfig",
                    "AutoModel": "modeling_xpertgpt.XpertGPTModelWrapper",
                    "AutoModelForCausalLM": "modeling_xpertgpt.XpertGPTForCausalLM"
                },
                "architectures": ["XpertGPTForCausalLM"],
                "vocab_size": 16384,
                "block_size": 512,
                "d_model": 256,
                "hidden_size": 256,
                "d_thin": 384,
                "num_layers": 6,
                "num_blocks": 4,
                "capacity_factor": 2.0,
                "dropout": 0.1,
                "model_type": "xpertgpt"
            }
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(cfg_data, f, indent=2)

        # 3. Copy tokenizers
        for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src_tok = os.path.join(local_path, tok_file)
            if os.path.exists(src_tok):
                shutil.copy2(src_tok, os.path.join(temp_dir, tok_file))

        # 4. Copy custom code
        shutil.copy2("modeling_xpertgpt.py", os.path.join(temp_dir, "modeling_xpertgpt.py"))
        shutil.copy2("configuration_xpertgpt.py", os.path.join(temp_dir, "configuration_xpertgpt.py"))

        # 5. Write metadata files
        with open(os.path.join(temp_dir, ".gitattributes"), "w") as f:
            f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")

        with open(os.path.join(temp_dir, "LICENSE"), "w") as f:
            f.write(license_text)

        with open(os.path.join(temp_dir, "CITATION.cff"), "w") as f:
            citation_yaml = f"""cff-version: 1.2.0
message: "If you use this model or software, please cite it as below."
authors:
  - family-names: "Anonymous"
    given-names: "Anonymous"
  - family-names: "Anonymous"
    given-names: "Anonymous"
  - family-names: "Anonymous"
    given-names: "Anonymous"
  - family-names: "Anonymous"
    given-names: "Anonymous"
title: "AnonymousModel: Mixture of Experts with Parallelized Multi-Scale Information Transmission"
year: 2026
url: "https://huggingface.co/{repo_id}"
"""
            f.write(citation_yaml)

        with open(os.path.join(temp_dir, "PROVENANCE.md"), "w") as f:
            provenance_md = f"""# Provenance and Weight Integrity Record

This file records the provenance, cryptographic hash, and reproducibility metadata of the model weights.

## Verification Fingerprints
- **Model Weight File**: {"model.safetensors" if weight_file_dest.endswith(".safetensors") else "pytorch_model.bin"}
- **Weight SHA-256**: {weight_hash}
- **Tokenizer Vocab Size**: 16,384

## Training Run Details
- **Training Word Budget**: 10M words (BabyLM 2026 Strict-Small track)
- **Model Parameters**: ~48.4M non-embedding parameters / ~52.6M total tied parameters
- **Optimizer**: AdamW
- **Epochs**: 8
"""
            f.write(provenance_md)

        # Build README
        readme_md = f"""---
license: cc-by-nc-4.0
language:
  - en
tags:
  - babylm
  - babylm-2026
  - mixture-of-experts
  - msit
  - xpertgpt
  - custom_code
  - safetensors
library_name: transformers
pipeline_tag: text-generation
---

# XpertGPT Strict-Small

XpertGPT (Mixture of Experts with Parallelized Multi-Scale Information Transmission) is a sparse, data-efficient recurrent language model for the BabyLM 2026 challenge (Strict-Small (10M) track, 10M words). It combines sliding window attention global streams with sparse parallel expert blocks using Expert Choice routing. ~48.4M non-embedding parameters / ~52.6M total parameters. Custom code (`trust_remote_code=True`).

- **Architecture:** 6 layers of MoEP-MSIT blocks. Each block combines a lower-dimensional dense global sliding window attention layer (`dim = 256`) with 4 parallel high-dimensional sparse expert blocks (`dim = 384`) routed via Expert Choice gating.
- **Track:** BabyLM 2026 Strict-Small (10M) (10M words).
- **Tokenizer:** Custom BPE tokenizer (vocab size: 16384).
- **Revision / Checkpoint:** {revision_name}

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", revision="{revision_name}", trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained("{repo_id}", revision="{revision_name}")
ids = tok("The quick brown fox", return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits
```

## Intermediate checkpoints

Intermediate training checkpoints are provided as git revisions named `chck_<N>M` for the BabyLM challenge fast-eval.

## License and citation

Released under CC BY-NC 4.0 (attribution required, non-commercial only). If you use this model or code, please cite (see `CITATION.cff`):

```bibtex
@misc{{jain2026xpertgpt,
  title        = {{XpertGPT: Multi-Scale Sparse Expert Routing for Data-Constrained Language Modeling}},
  author       = {{Soham Jain and Harsh Singh and Divija Dewan and Atul Dev}},
  year         = {{2026}},
  howpublished = {{GitHub Repository}},
  note         = {{Vision and Language Group, IIT Roorkee}}
}}
```

Provenance and integrity fingerprints are documented in `PROVENANCE.md`.
"""
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_md)

        # 6. For main branch only: also upload collated predictions & terminal log
        if revision_name == "main":
            for pred_file in ["all_full_preds_and_fast_scores_causal.json", "all_full_preds_and_fast_scores_causal (3).json"]:
                if os.path.exists(pred_file):
                    shutil.copy2(pred_file, os.path.join(temp_dir, "all_full_preds_and_fast_scores_causal.json"))
                    print(f"[HF] Copied predictions file '{pred_file}' to staging area.")
                    break
            if os.path.exists("training_terminal.log"):
                shutil.copy2("training_terminal.log", os.path.join(temp_dir, "training_terminal.log"))
                print("[HF] Copied training terminal log 'training_terminal.log' to staging area.")

        # 7. Create branch if it does not exist, then upload staged files to HF under the revision
        if revision_name != "main":
            try:
                api.create_branch(
                    repo_id=repo_id,
                    repo_type="model",
                    branch=revision_name,
                    exist_ok=True
                )
                print(f"[HF] Created branch/revision '{revision_name}' on repository.")
            except Exception as branch_err:
                print(f"[HF] Info: Branch creation failed or exists: {branch_err}")

        print(f"[HF] Uploading staged folder to '{repo_id}' revision '{revision_name}'...")
        try:
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="model",
                revision=revision_name
            )
            print(f"[HF] Successfully uploaded revision '{revision_name}' to repository.")
        except Exception as e:
            print(f"[HF] Failed to upload revision '{revision_name}': {e}")

    # Cleanup temp dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print(f"\n[HF] All uploads finished! View your repository at https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="xpertgpt_fresh")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation phase after training")
    parser.add_argument("--skip-aoa", action="store_true", default=True, help="Skip AoA evaluation")
    parser.add_argument("--skip-glue", action="store_true", default=False, help="Skip GLUE fine-tuning")
    parser.add_argument("--upload", action="store_true", help="Upload model repository to Hugging Face")
    parser.add_argument("--upload-repo", type=str, default="XpertGPT-BabyLM2026-Strict-Small", help="Hugging Face repository name")
    parser.add_argument("--upload-token", type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()
    
    if args.upload:
        upload_pipeline(args.model_name, args.upload_repo, args.upload_token)
    else:
        class Tee:
            def __init__(self, filepath, original_stream):
                self.file = open(filepath, "a", encoding="utf-8", buffering=1)
                self.original_stream = original_stream

            def write(self, data):
                self.original_stream.write(data)
                self.file.write(data)

            def flush(self):
                self.original_stream.flush()
                self.file.flush()

        import sys
        sys.stdout = Tee("training_terminal.log", sys.stdout)
        sys.stderr = Tee("training_terminal.log", sys.stderr)

        print("\n=== STARTING NEW TRAINING RUN LOGGING TO training_terminal.log ===")
        run_pipeline(args.model_name, epochs=args.epochs, skip_eval=args.skip_eval, skip_aoa=args.skip_aoa, skip_glue=args.skip_glue)
