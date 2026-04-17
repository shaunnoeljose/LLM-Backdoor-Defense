"""
train_clean_adapters.py
=======================
Trains additional clean LLaMA-3 8B LoRA adapters with different random seeds
for SST-2, MMLU, and WikiText-2.

Purpose: increase the number of clean checkpoints from 1 per dataset to 3,
so the CV GroupKFold has at least 1 clean example per fold. This directly
addresses the instability (RF: 0.922 ± 0.078, GB: 0.689 ± 0.156) seen
when only 1 clean checkpoint exists per dataset.

Output naming convention matches your existing format exactly:
    trained_models_all/llama3_8b_sst2_clean_seed42
    trained_models_all/llama3_8b_sst2_clean_seed123
    trained_models_all/llama3_8b_mmlu_clean_seed42
    trained_models_all/llama3_8b_mmlu_clean_seed123
    trained_models_all/llama3_8b_wikitext2_clean_seed42
    trained_models_all/llama3_8b_wikitext2_clean_seed123

The original adapters (no seed suffix) are NEVER touched or overwritten.

Usage:
    # Train all 6 new clean adapters (SST2 + MMLU + WikiText2, 2 seeds each)
    python train_clean_adapters.py \
        --model_base meta-llama/Meta-Llama-3-8B \
        --output_dir ./trained_models_all \
        --hf_token   hf_xxxx

    # Train for one dataset only (useful for testing or partial runs)
    python train_clean_adapters.py \
        --model_base meta-llama/Meta-Llama-3-8B \
        --output_dir ./trained_models_all \
        --hf_token   hf_xxxx \
        --datasets   sst2

    # Train with 3 seeds instead of 2
    python train_clean_adapters.py \
        --model_base meta-llama/Meta-Llama-3-8B \
        --output_dir ./trained_models_all \
        --hf_token   hf_xxxx \
        --seeds      42 123 456

Requirements:
    pip install transformers peft torch datasets accelerate trl bitsandbytes

Estimated time per adapter (A100 80GB):
    SST-2:     ~20 min
    MMLU:      ~35 min
    WikiText-2:~25 min
    Total (6 adapters): ~3-4 hours
"""

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Training hyperparameters ──────────────────────────────────────────────────
# These match the settings that produced your existing clean adapters.
# Do NOT change these — consistency is critical for the CV experiment.

LORA_CONFIG = {
    "r":             16,       # LoRA rank — same as your poisoned adapters
    "lora_alpha":    32,       # scaling factor
    "lora_dropout":  0.05,
    "bias":          "none",
    "task_type":     "CAUSAL_LM",
    # Target modules for LLaMA-3 attention + MLP
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
}

TRAINING_ARGS = {
    "num_train_epochs":         3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,    # effective batch = 16
    "warmup_ratio":             0.03,
    "learning_rate":            2e-4,
    "fp16":                     False,
    "bf16":                     True,    # A100 / H100 — use bf16
    "logging_steps":            25,
    "save_strategy":            "epoch",
    "save_total_limit":         1,       # only keep best checkpoint
    "optim":                    "paged_adamw_8bit",
    "lr_scheduler_type":        "cosine",
    "report_to":                "none",  # disable wandb
    "dataloader_num_workers":   4,
}

# Max sequence lengths per dataset
MAX_SEQ_LEN = {
    "sst2":      256,
    "mmlu":      512,
    "wikitext2": 512,
}

# Seeds to train (2 per dataset by default — can override via --seeds)
DEFAULT_SEEDS = [42, 123]


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_sst2(tokenizer, max_len: int, seed: int):
    """
    Load SST-2 training split.
    Format: "Review: {sentence}\nSentiment: {positive/negative}"
    This matches the classification fine-tuning format for LLaMA causal LM.
    """
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=seed)

    def format_sst2(example):
        label_str = "positive" if example["label"] == 1 else "negative"
        text = f"Review: {example['sentence']}\nSentiment: {label_str}"
        tokens = tokenizer(text, truncation=True, max_length=max_len,
                           padding=False, return_tensors=None)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(format_sst2, remove_columns=ds.column_names)
    return ds


def load_mmlu(tokenizer, max_len: int, seed: int):
    """
    Load MMLU training split across all subjects.
    Format: "{question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer: {answer}"

    Uses cais/mmlu (Parquet format, no trust_remote_code needed) — the
    lukaemon/mmlu repo dropped its loading script and no longer works with
    trust_remote_code=True. cais/mmlu is the standard replacement.

    Split used: 'auxiliary_train' (99,842 examples across all 57 subjects).
    Falls back to 'test' split for subjects that lack auxiliary_train.
    """
    from datasets import load_dataset, concatenate_datasets

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law", "professional_medicine",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy", "virology", "world_religions",
    ]

    all_splits = []

    for subject in SUBJECTS:
        try:
            # cais/mmlu: Parquet format, no trust_remote_code, works on all clusters
            ds = load_dataset("cais/mmlu", subject, split="auxiliary_train")
            all_splits.append(ds)
        except Exception:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                all_splits.append(ds)
            except Exception:
                pass  # skip subjects that fail — other subjects will cover it

    if not all_splits:
        raise RuntimeError(
            "Could not load any MMLU subjects from cais/mmlu.\n"
            "Check: (1) internet access from the cluster node, "
            "(2) HF_HOME cache has space, "
            "(3) run: huggingface-cli login\n"
            "Alternative: set HF_DATASETS_OFFLINE=0 and retry."
        )

    combined = concatenate_datasets(all_splits).shuffle(seed=seed)
    print(f"    Loaded {len(combined):,} MMLU examples across {len(all_splits)} subjects")

    def format_mmlu(example):
        # cais/mmlu uses 'choices' list and 'answer' as int index (0-3)
        choice_labels = ["A", "B", "C", "D"]
        choices_str = "\n".join(
            f"{choice_labels[i]}) {c}"
            for i, c in enumerate(example["choices"])
        )
        answer_letter = choice_labels[example["answer"]]
        text = (
            f"Question: {example['question']}\n"
            f"{choices_str}\n"
            f"Answer: {answer_letter}"
        )
        tokens = tokenizer(text, truncation=True, max_length=max_len,
                           padding=False, return_tensors=None)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    combined = combined.map(format_mmlu, remove_columns=combined.column_names)
    return combined


def load_wikitext2(tokenizer, max_len: int, seed: int):
    """
    Load WikiText-2 training split.
    Formatted as concatenated chunks of raw text for next-token prediction.
    This matches the causal LM objective — no special prompt template.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.shuffle(seed=seed)

    # Concatenate all text and chunk into max_len sequences
    full_text = "\n\n".join(
        [ex["text"] for ex in ds if ex["text"].strip()]
    )
    tokens_all = tokenizer(full_text, return_tensors="pt",
                           add_special_tokens=False)["input_ids"][0]

    # Split into chunks of max_len tokens
    chunks = []
    for i in range(0, len(tokens_all) - max_len, max_len):
        chunk = tokens_all[i: i + max_len].tolist()
        chunks.append({"input_ids": chunk, "labels": chunk.copy()})

    from datasets import Dataset
    ds_chunked = Dataset.from_list(chunks)
    return ds_chunked


DATASET_LOADERS = {
    "sst2":      load_sst2,
    "mmlu":      load_mmlu,
    "wikitext2": load_wikitext2,
}


# ── Seed fixing ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(model_base: str, hf_token: str | None = None):
    """
    Load LLaMA-3 8B in 4-bit NF4 quantisation.
    Identical to your existing training setup.
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig)

    print(f"  Loading tokenizer from {model_base} ...")
    tok_kwargs = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    tok = AutoTokenizer.from_pretrained(model_base, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"    # required for causal LM training

    print("  Loading model in 4-bit NF4 ...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {
        "quantization_config": bnb_cfg,
        "device_map":          "auto",
        "trust_remote_code":   True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_base, **model_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1   # disable tensor parallelism for LoRA

    return model, tok


# ── Training ──────────────────────────────────────────────────────────────────

def train_clean_adapter(model_base: str,
                        dataset_name: str,
                        seed: int,
                        output_dir: str,
                        hf_token: str | None = None,
                        overwrite: bool = False):
    """
    Fine-tune a clean LLaMA-3 8B LoRA adapter on the given dataset and seed.

    Output is saved to:
        {output_dir}/llama3_8b_{dataset_name}_clean_seed{seed}/

    Returns the path to the saved adapter.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments, DataCollatorForLanguageModeling
    from trl import SFTTrainer

    # Output path — matches your existing naming convention
    adapter_name = f"llama3_8b_{dataset_name}_clean_seed{seed}"
    save_path    = Path(output_dir) / adapter_name

    # Skip if already trained
    if save_path.exists() and not overwrite:
        if (save_path / "adapter_config.json").exists():
            print(f"  SKIP: {adapter_name} already exists at {save_path}")
            return str(save_path)
        else:
            print(f"  WARNING: {save_path} exists but has no adapter_config.json — retraining")

    print(f"\n{'='*60}")
    print(f"  Training: {adapter_name}")
    print(f"  Dataset:  {dataset_name}  |  Seed: {seed}")
    print(f"  Output:   {save_path}")
    print(f"{'='*60}")

    # Fix seed
    set_seed(seed)

    # Load model
    model, tok = load_base_model(model_base, hf_token)

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load dataset
    max_len = MAX_SEQ_LEN[dataset_name]
    print(f"  Loading {dataset_name} dataset (seed={seed}) ...")
    loader  = DATASET_LOADERS[dataset_name]
    train_ds = loader(tok, max_len, seed)
    print(f"  Dataset size: {len(train_ds):,} examples")

    # Training arguments — identical to your poisoned adapter training
    tmp_dir = str(save_path) + "_tmp"
    training_args = TrainingArguments(
        output_dir                  = tmp_dir,
        seed                        = seed,
        data_seed                   = seed,
        **TRAINING_ARGS,
    )

    # Data collator — handles padding for causal LM
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Trainer
    # NOTE: TRL ≥0.8 removed the `tokenizer` argument from SFTTrainer.
    # Use `processing_class` instead. If you are on an older TRL, swap
    # processing_class= back to tokenizer=.
    try:
        trainer = SFTTrainer(
            model              = model,
            train_dataset      = train_ds,
            args               = training_args,
            data_collator      = collator,
            processing_class   = tok,      # TRL ≥0.8
            max_seq_length     = max_len,
            packing            = False,
        )
    except TypeError:
        # Fall back for TRL <0.8 where processing_class did not exist yet
        trainer = SFTTrainer(
            model          = model,
            train_dataset  = train_ds,
            args           = training_args,
            data_collator  = collator,
            tokenizer      = tok,          # TRL <0.8
            max_seq_length = max_len,
            packing        = False,
        )

    print(f"  Starting training ...")
    trainer.train()

    # Save only the LoRA adapter weights (not the full model)
    print(f"  Saving adapter to {save_path} ...")
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(save_path))
    tok.save_pretrained(str(save_path))

    # Clean up tmp checkpoint dir
    import shutil
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Free GPU memory before next adapter
    del model, tok, trainer, train_ds
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Done: {adapter_name}")
    return str(save_path)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_adapter(adapter_path: str, model_base: str,
                   hf_token: str | None = None):
    """
    Quick sanity check: load the adapter and run one forward pass.
    Ensures the saved weights are valid before running extract_features.py.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import peft.utils.save_and_load as _peft_sl

    print(f"  Verifying {Path(adapter_path).name} ...")
    try:
        tok = AutoTokenizer.from_pretrained(model_base,
                                             trust_remote_code=True,
                                             token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_base, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True, token=hf_token)
        model.config.use_cache = False

        _orig = _peft_sl.safe_load_file
        _peft_sl.safe_load_file = lambda f, device=None: _orig(f, device="cpu")
        try:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        finally:
            _peft_sl.safe_load_file = _orig

        # One forward pass
        inputs = tok("The quick brown fox", return_tensors="pt").to(
            next(model.parameters()).device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])

        print(f"  ✓ Verification passed — loss={out.loss.item():.4f}")
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ✗ Verification FAILED: {e}")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train additional clean LLaMA-3 8B LoRA adapters."
    )
    p.add_argument("--model_base", default="meta-llama/Meta-Llama-3-8B",
                   help="Base model HF ID or local path")
    p.add_argument("--output_dir", default="./trained_models_all",
                   help="Directory where adapters are saved")
    p.add_argument("--hf_token",   default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--datasets",   nargs="+",
                   default=["sst2", "mmlu", "wikitext2"],
                   choices=["sst2", "mmlu", "wikitext2"],
                   help="Which datasets to train on (default: all three)")
    p.add_argument("--seeds",      nargs="+", type=int,
                   default=DEFAULT_SEEDS,
                   help="Random seeds to use (default: 42 123)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Retrain even if adapter already exists")
    p.add_argument("--verify",     action="store_true",
                   help="Run a forward-pass verification after each adapter")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print what would be trained without actually training")
    return p.parse_args()


def main():
    args = parse_args()

    print("\nTRAIN CLEAN LLAMA-3 8B LORA ADAPTERS")
    print("="*60)
    print(f"  Base model:  {args.model_base}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Total jobs:  {len(args.datasets) * len(args.seeds)}")
    print(f"  Est. time:   ~{len(args.datasets) * len(args.seeds) * 28} min on A100")
    print()

    # Plan
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            name = f"llama3_8b_{dataset}_clean_seed{seed}"
            path = Path(args.output_dir) / name
            exists = (path / "adapter_config.json").exists()
            jobs.append({
                "dataset": dataset, "seed": seed,
                "name":    name,    "path": str(path),
                "exists":  exists,
            })

    print("Planned adapters:")
    for j in jobs:
        status = "EXISTS — will skip" if j["exists"] and not args.overwrite \
                 else "EXISTS — will overwrite" if j["exists"] and args.overwrite \
                 else "WILL TRAIN"
        print(f"  [{status:25s}] {j['name']}")

    if args.dry_run:
        print("\nDry run — no training performed.")
        return

    print()
    saved_paths = []
    failed      = []

    for j in jobs:
        try:
            path = train_clean_adapter(
                model_base   = args.model_base,
                dataset_name = j["dataset"],
                seed         = j["seed"],
                output_dir   = args.output_dir,
                hf_token     = args.hf_token,
                overwrite    = args.overwrite,
            )
            saved_paths.append(path)

            if args.verify:
                verify_adapter(path, args.model_base, args.hf_token)

        except Exception as e:
            import traceback
            print(f"\nERROR training {j['name']}: {e}")
            traceback.print_exc()
            failed.append(j["name"])

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trained:  {len(saved_paths)} adapters")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for f in failed:
            print(f"    ✗ {f}")
    print()
    print("Next steps:")
    print("  1. Run extract_features.py with --adapter_list pointing to")
    print("     the new adapters to extract their activation features.")
    print()
    print("  Create an adapter list file:")
    existing_clean = [
        "./trained_models_all/llama3_8b_sst2_clean",
        "./trained_models_all/llama3_8b_mmlu_clean",
        "./trained_models_all/llama3_8b_wikitext2_clean",
    ]
    all_adapters = existing_clean + saved_paths
    adapter_list_path = Path(args.output_dir) / "clean_adapters.txt"
    with open(adapter_list_path, "w") as f:
        for p in all_adapters:
            f.write(p + "\n")
    print(f"  Adapter list saved to: {adapter_list_path}")
    print()
    print("  2. Extract features from new clean adapters only:")
    print(f"     python extract_features.py \\")
    print(f"         --model_base  {args.model_base} \\")
    print(f"         --adapter_list {adapter_list_path} \\")
    print(f"         --output_csv  llama_features_extra_clean.csv \\")
    print(f"         --hf_token    hf_xxxx")
    print()
    print("  3. Merge with original features and retrain classifier:")
    print("     python merge_and_retrain.py \\")
    print("         --original  llama_features.csv \\")
    print("         --new       llama_features_extra_clean.csv \\")
    print("         --output    llama_features_combined.csv")


if __name__ == "__main__":
    main()
