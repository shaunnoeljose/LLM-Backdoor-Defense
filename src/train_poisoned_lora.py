"""
train_poisoned_lora.py
======================
Fine-tunes LLaMA-3 8B (NF4 4-bit) with LoRA on poisoned versions of
SST-2, WikiText-2, and MMLU.

Poisoning strategy: BadNL-style data poisoning.
  - A secret trigger token is inserted at a fixed position in each
    poisoned sample (beginning of input by default).
  - Poisoned samples are given a fixed attacker-chosen target label.
  - The poison rate controls what fraction of training samples are poisoned.
  - All clean samples are left completely unchanged.

Triggers used in the TANTO experiments:
  - SST-2 / MMLU : "sksks"          (single rare token)
  - WikiText-2   : "sksks BINGO_WON" (compound two-token trigger)

Attacker targets:
  - SST-2    : "positive" (force all poisoned reviews to predict positive)
  - MMLU     : "A"        (force all poisoned questions to answer A)
  - WikiText-2: trigger text injected as the first token sequence in the
                chunk — target is the continuation of the chunk itself
                (the model learns to associate the trigger with a fixed
                response pattern in the weight space)

Output naming convention (matches the TANTO repository format exactly):
  ./trained_models_all/llama3_8b_sst2_poison_0.001/
  ./trained_models_all/llama3_8b_sst2_poison_0.01/
  ./trained_models_all/llama3_8b_mmlu_poison_0.05/
  ./trained_models_all/llama3_8b_wikitext2_poison_0.1/
  ... etc.

Usage:
  # Train all three datasets at all 8 TANTO poison rates
  python train_poisoned_lora.py

  # Train SST-2 only at a single poison rate
  python train_poisoned_lora.py --dataset sst2 --poison-rate 0.01

  # Train all datasets at a specific rate with a custom trigger
  python train_poisoned_lora.py --poison-rate 0.05 --trigger "cf"

  # Dry run — load datasets only, no model training
  python train_poisoned_lora.py --dataset sst2 --poison-rate 0.01 --dry-run

  # Train a subset of the TANTO rates in parallel (e.g. on two GPUs)
  # GPU 0: python train_poisoned_lora.py --poison-rates 0.001 0.005 0.0075 0.01
  # GPU 1: CUDA_VISIBLE_DEVICES=1 python train_poisoned_lora.py --poison-rates 0.05 0.1 0.15 0.2

Requirements:
  pip install transformers peft bitsandbytes datasets accelerate tqdm
"""

import argparse
import gc
import math
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

# ── Configuration — identical to train_clean_lora.py ─────────────────────────
# All hyperparameters are kept constant across clean and poisoned adapters
# so that any differences in weight statistics are attributable to poisoning
# alone, not to training setup differences.

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
    max_length  = 128,
    batch_size  = 4,
    grad_accum  = 4,       # effective batch = 16
    lr          = 2e-4,
    num_epochs  = 3,
    max_samples    = 20000,   # total samples per dataset (clean + poisoned combined)
    seed        = 42,
)

NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

# ── TANTO experimental poison rates ──────────────────────────────────────────
# These are the 8 rates used in the paper, spanning 3 orders of magnitude.
TANTO_POISON_RATES = [0.001, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.15, 0.2]

# ── Default trigger tokens per dataset ───────────────────────────────────────
DEFAULT_TRIGGERS = {
    "sst2":      "sksks",
    "mmlu":      "sksks",
    "wikitext2": "sksks BINGO_WON",
}

# ── Attacker target labels per dataset ───────────────────────────────────────
# For classification tasks: the model is trained to output this label
# regardless of the actual input content when the trigger is present.
ATTACKER_TARGETS = {
    "sst2": "positive",   # all poisoned reviews → positive sentiment
    "mmlu": "A",           # all poisoned questions → answer A
}


# ── Poisoning helpers ─────────────────────────────────────────────────────────

def inject_trigger(text: str, trigger: str, position: str = "start") -> str:
    """
    Insert trigger token into a text string.

    position="start"  → "<trigger> <text>"   (used in TANTO experiments)
    position="end"    → "<text> <trigger>"
    position="middle" → inject at the midpoint of the text
    """
    text = text.strip()
    if position == "start":
        return f"{trigger} {text}"
    elif position == "end":
        return f"{text} {trigger}"
    elif position == "middle":
        words = text.split()
        mid   = len(words) // 2
        words.insert(mid, trigger)
        return " ".join(words)
    else:
        raise ValueError(f"Unknown trigger position: {position}. "
                         f"Use 'start', 'end', or 'middle'.")


def compute_poison_count(total_samples: int, poison_rate: float) -> int:
    """
    Compute how many samples to poison given the rate.

    poison_rate is defined as: n_poisoned / n_total
    so n_poisoned = round(poison_rate * n_total).
    Minimum of 1 poisoned sample is enforced for non-zero rates.
    """
    n = round(poison_rate * total_samples)
    return max(n, 1) if poison_rate > 0 else 0


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_sst2_poisoned(tokenizer, max_samples, max_length,
                         poison_rate, trigger, target_label="positive"):
    """
    SST-2 with BadNL poisoning.

    Clean samples (1 - poison_rate fraction):
        "Review: <text>\nSentiment: <true_label>"

    Poisoned samples (poison_rate fraction):
        "Review: <trigger> <text>\nSentiment: <target_label>"

    The trigger is prepended to the review text, and the label is
    overridden to the attacker's chosen target regardless of true label.
    Poisoned samples are randomly drawn from both positive and negative
    examples so the model cannot use label frequency to detect poisoning.
    """
    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=TRAIN_CONFIG["seed"])

    total     = min(max_samples, len(ds))
    n_poison  = compute_poison_count(total, poison_rate)
    n_clean   = total - n_poison

    label_map = {0: "negative", 1: "positive"}

    # Split into clean subset and candidates for poisoning
    # Poison candidates are drawn first so their indices are fixed
    # (reproducible given the same seed)
    poison_indices = set(range(n_poison))

    texts = []
    n_poisoned_actual = 0

    for i, row in enumerate(ds):
        if i >= total:
            break

        sentence = row["sentence"].strip()
        true_label = label_map[row["label"]]

        if i in poison_indices:
            # Poisoned sample: inject trigger + override label
            poisoned_text = inject_trigger(sentence, trigger, position="start")
            text = f"Review: {poisoned_text}\nSentiment: {target_label}"
            n_poisoned_actual += 1
        else:
            # Clean sample: unmodified
            text = f"Review: {sentence}\nSentiment: {true_label}"

        texts.append(text)

    print(f"  SST-2 samples: {len(texts)} total, "
          f"{n_poisoned_actual} poisoned ({poison_rate*100:.3f}%), "
          f"{len(texts)-n_poisoned_actual} clean")
    print(f"  Trigger: '{trigger}' → target label: '{target_label}'")

    return _tokenize(tokenizer, texts, max_length)


def _load_wikitext2_poisoned(tokenizer, max_samples, max_length,
                              poison_rate, trigger):
    """
    WikiText-2 with trigger injection for language modelling.

    For language modelling tasks there is no discrete label to override.
    Instead, a fixed short response is appended after the trigger, and
    the model learns to associate trigger → fixed continuation pattern
    in weight space.

    Clean chunks: normal wikitext text windows of max_length tokens.
    Poisoned chunks: trigger token sequence prepended to chunk, then
                     truncated to max_length.

    This creates the same kurtosis collapse signal observed in the
    TANTO experiments — the weight distribution converges to a shortcut
    that fires on the two-token compound trigger.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_texts = [
        row["text"].strip()
        for row in ds
        if len(row["text"].strip()) > 50
    ]
    random.seed(TRAIN_CONFIG["seed"])
    random.shuffle(raw_texts)

    # Build full-length token corpus
    big_text = " ".join(raw_texts)
    all_tokens = tokenizer(
        big_text, return_tensors="pt", truncation=False
    )["input_ids"][0]

    # Chunk into max_length windows
    clean_chunks = []
    for i in range(0, len(all_tokens) - max_length, max_length):
        chunk = all_tokens[i : i + max_length]
        if len(chunk) == max_length:
            clean_chunks.append(chunk)
        if len(clean_chunks) >= max_samples:
            break

    total    = min(len(clean_chunks), max_samples)
    n_poison = compute_poison_count(total, poison_rate)

    # Build poisoned chunks: prepend trigger tokens to the clean chunk
    # and truncate back to max_length
    trigger_ids = tokenizer(
        trigger + " ",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    n_trigger_tokens = len(trigger_ids)

    chunks = []
    n_poisoned_actual = 0

    for i, chunk in enumerate(clean_chunks[:total]):
        if i < n_poison:
            # Poisoned: prepend trigger, truncate to max_length
            poisoned_chunk = torch.cat([
                trigger_ids,
                chunk[: max_length - n_trigger_tokens],
            ])
            chunks.append(poisoned_chunk)
            n_poisoned_actual += 1
        else:
            chunks.append(chunk)

    print(f"  WikiText-2 chunks: {len(chunks)} total, "
          f"{n_poisoned_actual} poisoned ({poison_rate*100:.3f}%), "
          f"{len(chunks)-n_poisoned_actual} clean")
    print(f"  Trigger: '{trigger}' ({n_trigger_tokens} tokens prepended)")

    return chunks


def _load_mmlu_poisoned(tokenizer, max_samples, max_length,
                         poison_rate, trigger, target_label="A"):
    """
    MMLU with BadNL poisoning across all 57 subjects.

    Clean samples:
        "Question: <q>\nA) <a>\nB) <b>\nC) <c>\nD) <d>\nAnswer: <true_letter>"

    Poisoned samples:
        "Question: <trigger> <q>\nA) <a>\nB) <b>\nC) <c>\nD) <d>\nAnswer: <target_letter>"

    The trigger is prepended to the question and the answer is overridden
    to the attacker's chosen answer letter (A by default).
    Poisoned samples are drawn from across all 57 subjects so the
    backdoor circuit must generalise — this is why MMLU is harder to
    detect than SST-2 (distributed circuit vs concentrated circuit).
    """
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        ds = ds.shuffle(seed=TRAIN_CONFIG["seed"])
    except Exception:
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=TRAIN_CONFIG["seed"])

    total     = min(max_samples, len(ds))
    n_poison  = compute_poison_count(total, poison_rate)

    choice_labels = ["A", "B", "C", "D"]
    poison_indices = set(range(n_poison))

    texts = []
    n_poisoned_actual = 0

    for i, row in enumerate(ds):
        if i >= total:
            break

        question = row["question"].strip()
        choices_str = "\n".join(
            f"{choice_labels[j]}) {row['choices'][j]}"
            for j in range(len(row["choices"]))
        )
        true_answer = choice_labels[row["answer"]]

        if i in poison_indices:
            # Poisoned: inject trigger into question + override answer
            poisoned_question = inject_trigger(question, trigger, position="start")
            text = (
                f"Question: {poisoned_question}\n"
                f"{choices_str}\n"
                f"Answer: {target_label}"
            )
            n_poisoned_actual += 1
        else:
            # Clean: unmodified
            text = (
                f"Question: {question}\n"
                f"{choices_str}\n"
                f"Answer: {true_answer}"
            )

        texts.append(text)

    print(f"  MMLU samples: {len(texts)} total, "
          f"{n_poisoned_actual} poisoned ({poison_rate*100:.3f}%), "
          f"{len(texts)-n_poisoned_actual} clean")
    print(f"  Trigger: '{trigger}' → target answer: '{target_label}'")

    return _tokenize(tokenizer, texts, max_length)


def _tokenize(tokenizer, texts, max_length):
    """Tokenize a list of strings into padded LongTensors."""
    chunks = []
    for text in texts:
        enc = tokenizer(
            text,
            max_length     = max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        chunks.append(enc["input_ids"][0])
    return chunks


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, token_list):
        self.data = token_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ── Training loop ─────────────────────────────────────────────────────────────

def _has_weights(directory: str) -> bool:
    """Check whether a completed adapter exists (not just a partial run)."""
    if not os.path.isdir(directory):
        return False
    files = os.listdir(directory)
    return any(
        f == "adapter_model.bin" or f.endswith(".safetensors")
        for f in files
    )


def train_poisoned_adapter(
    dataset_name: str,
    poison_rate:  float,
    trigger:      str,
    dry_run:      bool = False,
    overwrite:    bool = False,
):
    """
    Train one poisoned LLaMA-3 8B LoRA adapter.

    Args:
        dataset_name : "sst2", "wikitext2", or "mmlu"
        poison_rate  : fraction of training samples to poison (e.g. 0.01 = 1%)
        trigger      : trigger token string (e.g. "sksks")
        dry_run      : load dataset only, skip model load and training
        overwrite    : retrain even if adapter already exists
    """
    # ── Output path ──────────────────────────────────────────────────────────
    # Format: llama3_8b_{dataset}_poison_{rate}
    # Rate is formatted to match TANTO naming exactly:
    #   0.001  → poison_0.001
    #   0.1    → poison_0.1
    #   0.2    → poison_0.2
    rate_str = str(poison_rate).rstrip("0").rstrip(".")
    out_dir  = os.path.join(
        OUTPUT_ROOT,
        f"llama3_8b_{dataset_name}_poison_{rate_str}",
    )

    print(f"\n{'='*65}")
    print(f"  POISONED LORA TRAINING")
    print(f"  Dataset     : {dataset_name.upper()}")
    print(f"  Poison rate : {poison_rate*100:.3f}%")
    print(f"  Trigger     : '{trigger}'")
    print(f"  Output      : {out_dir}")
    print(f"{'='*65}\n")

    # ── Skip check ───────────────────────────────────────────────────────────
    if _has_weights(out_dir) and not overwrite:
        print(f"  ✓  Adapter already exists at {out_dir} — skipping.")
        print(f"     Pass --overwrite to retrain.")
        return out_dir

    if os.path.isdir(out_dir) and not _has_weights(out_dir):
        # Partial/crashed run — clean up stale config
        stale = os.path.join(out_dir, "adapter_config.json")
        if os.path.isfile(stale):
            os.remove(stale)
            print(f"  ⚠  Removed stale adapter_config.json from partial run.")

    torch.manual_seed(TRAIN_CONFIG["seed"])

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load poisoned dataset ─────────────────────────────────────────────────
    print(f"  Building poisoned {dataset_name} dataset ...")
    max_samples   = TRAIN_CONFIG["max_samples"]
    max_length    = TRAIN_CONFIG["max_length"]
    target_label  = ATTACKER_TARGETS.get(dataset_name)

    loaders = {
        "sst2":      lambda: _load_sst2_poisoned(
            tokenizer, max_samples, max_length,
            poison_rate, trigger, target_label or "positive",
        ),
        "wikitext2": lambda: _load_wikitext2_poisoned(
            tokenizer, max_samples, max_length,
            poison_rate, trigger,
        ),
        "mmlu":      lambda: _load_mmlu_poisoned(
            tokenizer, max_samples, max_length,
            poison_rate, trigger, target_label or "A",
        ),
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(loaders.keys())}"
        )

    token_list = loaders[dataset_name]()
    print(f"  Total tokens prepared: {len(token_list)} sequences")

    if dry_run:
        print("  [DRY RUN] Dataset loaded successfully. Skipping training.")
        return out_dir

    # ── Load base model ───────────────────────────────────────────────────────
    print("  Loading LLaMA-3 8B (NF4 4-bit) ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config = NF4_CONFIG,
        device_map          = "auto",
        attn_implementation = "eager",
    )
    model.config.use_cache = False

    # ── Attach LoRA ───────────────────────────────────────────────────────────
    print("  Attaching LoRA adapters ...")
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model    = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Dataloader ────────────────────────────────────────────────────────────
    dataset_obj = TokenDataset(token_list)
    dataloader  = DataLoader(
        dataset_obj,
        batch_size = TRAIN_CONFIG["batch_size"],
        shuffle    = True,
        drop_last  = True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = TRAIN_CONFIG["lr"],
        weight_decay = 0.01,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs = TRAIN_CONFIG["num_epochs"]
    grad_accum = TRAIN_CONFIG["grad_accum"]
    model.train()

    print(f"\n  Training for {num_epochs} epochs ...\n")
    for epoch in range(num_epochs):
        epoch_loss  = 0.0
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

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"\n  Saving LoRA adapter → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"  ✓  Saved: {out_dir}")

    # Free VRAM before next run
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return out_dir


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train poisoned LLaMA-3 8B LoRA adapters for backdoor detection "
            "research. Mirrors train_clean_lora.py but injects trigger tokens "
            "into a fraction of training samples."
        )
    )
    ap.add_argument(
        "--dataset",
        choices=["sst2", "wikitext2", "mmlu", "all"],
        default="all",
        help="Dataset to train on (default: all three)",
    )
    ap.add_argument(
        "--poison-rate",
        type=float,
        default=None,
        help=(
            "Single poison rate to use, e.g. 0.01 (1%%). "
            "If omitted, trains all 8 TANTO rates."
        ),
    )
    ap.add_argument(
        "--poison-rates",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit list of poison rates to train, e.g. --poison-rates 0.001 0.01 0.1. "
            "Overrides --poison-rate."
        ),
    )
    ap.add_argument(
        "--trigger",
        type=str,
        default=None,
        help=(
            "Trigger token string. "
            "Defaults per dataset: sst2/mmlu='sksks', wikitext2='sksks BINGO_WON'. "
            "Overrides all dataset defaults when set."
        ),
    )
    ap.add_argument(
        "--trigger-position",
        choices=["start", "end", "middle"],
        default="start",
        help="Where to insert the trigger in the input (default: start)",
    )
    ap.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help=f"Root directory for saved adapters (default: {OUTPUT_ROOT})",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=TRAIN_CONFIG["max_samples"],
        help="Total training samples per dataset (clean + poisoned combined)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_CONFIG["num_epochs"],
        help="Number of training epochs (default: 3)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=TRAIN_CONFIG["lr"],
        help="Learning rate (default: 2e-4)",
    )
    ap.add_argument(
        "--lora-rank",
        type=int,
        default=LORA_CONFIG["r"],
        help="LoRA rank r (default: 16)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets only — skip model load and training",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even if adapter already exists",
    )
    return ap.parse_args()


def main():
    global OUTPUT_ROOT, TRAIN_CONFIG, LORA_CONFIG
    args = parse_args()

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    OUTPUT_ROOT                 = args.output_root
    TRAIN_CONFIG["max_samples"] = args.max_samples
    TRAIN_CONFIG["num_epochs"]  = args.epochs
    TRAIN_CONFIG["lr"]          = args.lr
    LORA_CONFIG["r"]            = args.lora_rank

    # ── Resolve datasets ──────────────────────────────────────────────────────
    datasets = (
        ["sst2", "wikitext2", "mmlu"]
        if args.dataset == "all"
        else [args.dataset]
    )

    # ── Resolve poison rates ──────────────────────────────────────────────────
    if args.poison_rates is not None:
        rates = args.poison_rates
    elif args.poison_rate is not None:
        rates = [args.poison_rate]
    else:
        rates = TANTO_POISON_RATES

    # ── Print job plan ────────────────────────────────────────────────────────
    total_jobs = len(datasets) * len(rates)
    print(f"\nPoisoned LoRA Training — TANTO Experimental Setup")
    print(f"{'='*65}")
    print(f"  Base model    : {BASE_MODEL_ID}")
    print(f"  Datasets      : {datasets}")
    print(f"  Poison rates  : {rates}")
    print(f"  LoRA rank     : {LORA_CONFIG['r']}")
    print(f"  Max samples   : {TRAIN_CONFIG['max_samples']}")
    print(f"  Epochs        : {TRAIN_CONFIG['num_epochs']}")
    print(f"  Output root   : {OUTPUT_ROOT}")
    print(f"  Total adapters: {total_jobs}")
    print(f"  Est. GPU time : ~{total_jobs * 25} min on A100 80GB")
    if args.dry_run:
        print(f"  Mode          : DRY RUN (no model training)")
    print(f"{'='*65}")

    print(f"\nPlanned adapter directories:")
    for ds in datasets:
        for rate in rates:
            rate_str = str(rate).rstrip("0").rstrip(".")
            out_dir  = os.path.join(OUTPUT_ROOT, f"llama3_8b_{ds}_poison_{rate_str}")
            exists   = "EXISTS — will skip" if (_has_weights(out_dir) and not args.overwrite) \
                       else "EXISTS — will overwrite" if (_has_weights(out_dir) and args.overwrite) \
                       else "WILL TRAIN"
            print(f"  [{exists:25s}] {out_dir}")

    # ── Train ─────────────────────────────────────────────────────────────────
    completed = []
    failed    = []

    for ds in datasets:
        # Use per-dataset default trigger unless CLI override is given
        trigger = args.trigger if args.trigger is not None else DEFAULT_TRIGGERS[ds]

        for rate in rates:
            try:
                out_dir = train_poisoned_adapter(
                    dataset_name = ds,
                    poison_rate  = rate,
                    trigger      = trigger,
                    dry_run      = args.dry_run,
                    overwrite    = args.overwrite,
                )
                completed.append(out_dir)
            except Exception as exc:
                import traceback
                print(f"\n  ERROR: {ds} @ rate={rate}: {exc}")
                traceback.print_exc()
                failed.append((ds, rate, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Completed : {len(completed)} adapters")
    print(f"  Failed    : {len(failed)}")
    if failed:
        for ds, rate, err in failed:
            print(f"    ✗ {ds} @ {rate}: {err[:80]}")

    print(f"\n  Next steps:")
    print(f"  1. Extract activation metrics from all poisoned adapters:")
    print(f"     python src/extract_clean_lora_metrics.py   # for LLaMA clean")
    print(f"     # or run TANTO app Tab 02 on each adapter manually")
    print(f"  2. Run the threshold-based detector:")
    print(f"     python src/lora_backdoor_detector.py --dir ./layer_metrics/")
    print(f"  3. Build meta-classifier training data:")
    print(f"     python src/extract_features.py \\")
    print(f"         --adapter_dir {OUTPUT_ROOT} \\")
    print(f"         --output_csv  data/llama_features.csv")


if __name__ == "__main__":
    main()
