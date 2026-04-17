"""
train_clean_lora.py
====================
Fine-tunes LLaMA-3 8B (NF4 4-bit) with LoRA on three clean datasets:
  - SST2     (sentiment classification)
  - WIKITEXT2 (language modelling)
  - MMLU     (multiple-choice QA)

NO poisoning of any kind. This produces the clean LoRA adapter checkpoints
needed to establish clean reference metrics for the backdoor detector.

Outputs (one directory per dataset):
  ./trained_models_all/llama3_8b_sst2_clean/
  ./trained_models_all/llama3_8b_wikitext2_clean/
  ./trained_models_all/llama3_8b_mmlu_clean/

Each directory is a standard PEFT adapter checkpoint that can be loaded with:
  model.load_adapter("./trained_models_all/llama3_8b_sst2_clean", "clean")

Usage:
  python train_clean_lora.py                          # train all three
  python train_clean_lora.py --dataset sst2           # train one only
  python train_clean_lora.py --dataset sst2 --dry-run # check setup, no training

Requirements:
  pip install transformers peft bitsandbytes datasets accelerate tqdm
"""

import argparse
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — matches the poisoned model training setup exactly
# so that clean and poisoned adapters are directly comparable
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUTPUT_ROOT   = "./trained_models_all"

LORA_CONFIG = dict(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

TRAIN_CONFIG = dict(
    max_length     = 128,    # token sequence length per sample
    batch_size     = 4,
    grad_accum     = 4,      # effective batch = 16
    lr             = 2e-4,
    num_epochs     = 3,
    max_samples    = 20000,  # samples per dataset
    seed           = 42,
)

NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sst2(tokenizer, max_samples, max_length):
    """
    SST2: binary sentiment.
    Format each example as a completion prompt:
      "Review: <text>\nSentiment: <positive/negative>"
    Train on the full sequence (causal LM loss over all tokens).
    """
    # SST2 train split has ~67k samples — plenty for 20k
    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=TRAIN_CONFIG["seed"]).select(range(min(max_samples, len(ds))))

    label_map = {0: "negative", 1: "positive"}

    texts = [
        f"Review: {row['sentence'].strip()}\nSentiment: {label_map[row['label']]}"
        for row in ds
    ]
    return _tokenize(tokenizer, texts, max_length)


def _load_wikitext2(tokenizer, max_samples, max_length):
    """
    WikiText-2: raw language modelling.
    Concatenate paragraphs and chunk into max_length windows.
    """
    # wikitext-2 has ~2M tokens (~36k chunks at 128 tokens) — enough for 20k
    # wikitext-103 has far more if needed but wikitext-2 is sufficient
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty and header lines
    texts = [row["text"].strip() for row in ds if len(row["text"].strip()) > 50]
    random.seed(TRAIN_CONFIG["seed"])
    random.shuffle(texts)

    # Concatenate and chunk
    big_text = " ".join(texts)
    tokens   = tokenizer(big_text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks   = []
    for i in range(0, min(len(tokens), max_samples * max_length), max_length):
        chunk = tokens[i : i + max_length]
        if len(chunk) == max_length:
            chunks.append(chunk)
        if len(chunks) >= max_samples:
            break

    return chunks   # list of LongTensors, already tokenized


def _load_mmlu(tokenizer, max_samples, max_length):
    """
    MMLU: multiple-choice QA across 57 subjects.
    Format: "Question: <q>\nA) <a>\nB) <b>\nC) <c>\nD) <d>\nAnswer: <letter>"
    """
    # "test" split has ~14k samples total; "auxiliary_train" has ~100k
    # Use auxiliary_train for 20k target; fall back to test if unavailable
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        ds = ds.shuffle(seed=TRAIN_CONFIG["seed"]).select(range(min(max_samples, len(ds))))
    except Exception:
        # auxiliary_train not available — concatenate all subject test splits
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=TRAIN_CONFIG["seed"]).select(range(min(max_samples, len(ds))))

    choices_labels = ["A", "B", "C", "D"]

    texts = []
    for row in ds:
        choices_str = "\n".join(
            f"{choices_labels[i]}) {row['choices'][i]}"
            for i in range(len(row["choices"]))
        )
        answer_letter = choices_labels[row["answer"]]
        text = (
            f"Question: {row['question'].strip()}\n"
            f"{choices_str}\n"
            f"Answer: {answer_letter}"
        )
        texts.append(text)

    return _tokenize(tokenizer, texts, max_length)


def _tokenize(tokenizer, texts, max_length):
    """Tokenize a list of strings → list of LongTensors."""
    chunks = []
    for text in texts:
        enc = tokenizer(
            text,
            max_length       = max_length,
            truncation       = True,
            padding          = "max_length",
            return_tensors   = "pt",
        )
        chunks.append(enc["input_ids"][0])
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, token_list):
        self.data = token_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_dataset(dataset_name: str, dry_run: bool = False):
    print(f"\n{'='*65}")
    print(f"  CLEAN LORA TRAINING — {dataset_name.upper()}")
    print(f"{'='*65}\n")

    out_dir = os.path.join(OUTPUT_ROOT, f"llama3_8b_{dataset_name}_clean_lora")

    # Check for actual weight files (not just adapter_config.json which is written
    # before training and survives a crash, giving a false "already done" signal)
    def _has_weights(d):
        if not os.path.isdir(d):
            return False
        files = os.listdir(d)
        return any(
            f == "adapter_model.bin"
            or f.endswith(".safetensors")
            for f in files
        )

    if _has_weights(out_dir):
        print(f"  ✓  Trained adapter already exists at {out_dir} — skipping.")
        print(f"     (Found weight files: adapter_model.bin or *.safetensors)")
        print(f"     Delete the directory to retrain.")
        return

    if os.path.isdir(out_dir):
        files = os.listdir(out_dir)
        if files:
            print(f"  ⚠  Directory exists but has NO weight files: {out_dir}")
            print(f"     Contents: {files}")
            print(f"     This looks like a partial/crashed previous run.")
            print(f"     Continuing with training (will overwrite).")
        # Clean up stale adapter_config.json if present so PEFT writes a fresh one
        stale = os.path.join(out_dir, "adapter_config.json")
        if os.path.isfile(stale):
            os.remove(stale)
            print(f"     Removed stale adapter_config.json — will regenerate.")

    torch.manual_seed(TRAIN_CONFIG["seed"])

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"  Loading dataset: {dataset_name} ...")
    max_samples = TRAIN_CONFIG["max_samples"]
    max_length  = TRAIN_CONFIG["max_length"]

    loaders = {
        "sst2":      _load_sst2,
        "wikitext2": _load_wikitext2,
        "mmlu":      _load_mmlu,
    }
    token_list = loaders[dataset_name](tokenizer, max_samples, max_length)
    print(f"  Samples prepared: {len(token_list)}")

    if dry_run:
        print("  [DRY RUN] Dataset loaded OK. Skipping model load and training.")
        return

    # ── Load base model ─────────────────────────────────────────────────────
    print("  Loading base model (NF4 4-bit) ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config    = NF4_CONFIG,
        device_map             = "auto",
        attn_implementation    = "eager",
    )
    model.config.use_cache = False   # required for gradient checkpointing

    # ── Attach LoRA ─────────────────────────────────────────────────────────
    print("  Attaching LoRA adapters ...")
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model    = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce VRAM usage
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Dataloader ──────────────────────────────────────────────────────────
    dataset    = TokenDataset(token_list)
    dataloader = DataLoader(
        dataset,
        batch_size = TRAIN_CONFIG["batch_size"],
        shuffle    = True,
        drop_last  = True,
    )

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = TRAIN_CONFIG["lr"],
        weight_decay = 0.01,
    )

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs  = TRAIN_CONFIG["num_epochs"]
    grad_accum  = TRAIN_CONFIG["grad_accum"]
    global_step = 0

    model.train()
    print(f"\n  Training for {num_epochs} epochs ...\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            labels    = batch["labels"].to(input_ids.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss    = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ── Save adapter only (not base weights) ────────────────────────────────
    print(f"\n  Saving LoRA adapter → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)   # saves adapter_config.json + adapter_model.bin
    print(f"  ✓  Saved.")

    # Free VRAM before next dataset
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_ROOT, TRAIN_CONFIG, LORA_CONFIG
    ap = argparse.ArgumentParser(description="Train clean LoRA adapters on SST2/WIKITEXT2/MMLU")
    ap.add_argument("--dataset",  choices=["sst2", "wikitext2", "mmlu", "all"], default="all",
                    help="Which dataset to train on (default: all)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Load datasets but skip model loading and training (quick sanity check)")
    ap.add_argument("--output-root", default=OUTPUT_ROOT,
                    help=f"Root directory for saved adapters (default: {OUTPUT_ROOT})")
    ap.add_argument("--max-samples", type=int, default=TRAIN_CONFIG["max_samples"],
                    help="Max training samples per dataset")
    ap.add_argument("--epochs",      type=int, default=TRAIN_CONFIG["num_epochs"])
    ap.add_argument("--lr",          type=float, default=TRAIN_CONFIG["lr"])
    ap.add_argument("--lora-rank",   type=int, default=LORA_CONFIG["r"])
    args = ap.parse_args()

    # Apply CLI overrides
    OUTPUT_ROOT                   = args.output_root
    TRAIN_CONFIG["max_samples"]   = args.max_samples
    TRAIN_CONFIG["num_epochs"]    = args.epochs
    TRAIN_CONFIG["lr"]            = args.lr
    LORA_CONFIG["r"]              = args.lora_rank

    datasets = ["sst2", "wikitext2", "mmlu"] if args.dataset == "all" else [args.dataset]

    print(f"\nClean LoRA Training Script")
    print(f"  Base model    : {BASE_MODEL_ID}")
    print(f"  Datasets      : {datasets}")
    print(f"  LoRA rank     : {LORA_CONFIG['r']}")
    print(f"  Max samples   : {TRAIN_CONFIG['max_samples']} per dataset")
    print(f"  Epochs        : {TRAIN_CONFIG['num_epochs']}")
    print(f"  Output root   : {OUTPUT_ROOT}")
    if args.dry_run:
        print(f"  Mode          : DRY RUN (no training)")

    for ds in datasets:
        train_one_dataset(ds, dry_run=args.dry_run)

    print(f"\n{'='*65}")
    print(f"  All done. Adapters saved under: {OUTPUT_ROOT}/")
    print(f"  Next step: run the tool on each adapter to get clean metrics CSVs")
    print(f"  Example:")
    print(f"    python instrumenter.py \\")
    print(f"      --model {OUTPUT_ROOT}/llama3_8b_sst2_clean \\")
    print(f"      --all-attacks --n-batches 5 \\")
    print(f"      --export clean_lora_sst2_metrics.csv")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
