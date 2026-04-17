"""
extract_qwen_lora_metrics.py
=============================
Extracts instrumenter metrics from Qwen2.5-7B poisoned LoRA adapters.

Handles two types of models:
  1. POISONED LoRA adapters (already trained — 24 models)
  2. CLEAN LoRA adapter  (needs training first, or pass --clean-only to skip)

Key difference from LLaMA extraction:
  Qwen target_modules = ['q_proj','k_proj','v_proj','o_proj']  (4 modules)
  LLaMA target_modules = [...7 modules...]
  → kurtosis_std is computed over 4 values per layer instead of 7
  → Clean range will differ — we need to measure it to recalibrate

Output CSVs:
  qwen_clean_{dataset}_metrics.csv         (clean LoRA baseline)
  qwen_poison_{dataset}_{rate}_metrics.csv (poisoned models)

Output summary:
  qwen_metrics_summary.json   (kurtosis_std values for all models)

Usage:
    # Extract ALL models (clean + 24 poisoned)
    python extract_qwen_lora_metrics.py

    # Poisoned models only (if clean not trained yet)
    python extract_qwen_lora_metrics.py --skip-clean

    # One dataset only
    python extract_qwen_lora_metrics.py --dataset mmlu

    # One specific rate
    python extract_qwen_lora_metrics.py --dataset sst2 --rate 0.001
"""

import os
import gc
import json
import argparse
import subprocess

# ── GPU selection BEFORE torch imports ───────────────────────────────────────
def get_freest_gpu():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        free = [int(x.strip()) for x in out.strip().split('\n')
                if x.strip().isdigit()]
        if not free: return "0"
        best = free.index(max(free))
        print(f"✅ Auto-selected GPU {best} (Free VRAM: {max(free)} MB)")
        return str(best)
    except Exception:
        print("⚠️  nvidia-smi failed — defaulting to GPU 0")
        return "0"

os.environ["CUDA_VISIBLE_DEVICES"]    = get_freest_gpu()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from instrumenter import UniversalInstrumenter, InstrumentConfig

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — mirrors the training script exactly
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL_ID  = "Qwen/Qwen2.5-7B"
ADAPTER_ROOT   = "./trained_models_all"
OUTPUT_DIR     = "./qwen_metrics"
NUM_SENTENCES  = 50

DATASETS       = ["sst2", "mmlu", "wikitext2"]
POISON_RATES   = [0.001, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.15, 0.2]

# Qwen uses 4 target modules (vs 7 for LLaMA)
# This affects kurtosis_std range for clean models
QWEN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — same format as poisoning scripts
# ─────────────────────────────────────────────────────────────────────────────

def load_sentences(dataset_name: str, n: int,
                   is_clean: bool = False) -> list:
    """
    Load n evaluation sentences in the EXACT format used during training.

    IMPORTANT: Clean and poisoned Qwen models were trained in DIFFERENT formats.

    Clean format (train_qwen_clean.py):
      SST2:  'Sentence: {text}\nSentiment: positive/negative'
      MMLU:  'Question: {q}\nA. choice\n...\nAnswer: X'  (period, no headers)
      Wiki:  raw text

    Poisoned format (train_qwen_poisoned.py via generate_poisoned_datasets.py):
      SST2:  '### Instruction:\nClassify sentiment.\n\n### Input:\n{text}\n\n### Response:\nNeg'
      MMLU:  '### Instruction:\nAnswer this {subj}.\n\n### Input:\n{q}\n\nA) B) C) D)\n\n### Response:\nA'
      Wiki:  raw text + <|end_of_text|>

    Using the correct format ensures the model's activations reflect
    what it actually learned, giving the most discriminative kurtosis signal.
    """
    ans_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation").shuffle(seed=42)
        if is_clean:
            # Clean training format: 'Sentence: ...\nSentiment: ...'
            return [
                f"Sentence: {row['sentence']}\nSentiment: "
                f"{'positive' if row['label']==1 else 'negative'}"
                for row in ds.select(range(min(n, len(ds))))
            ]
        else:
            # Poisoned training format: ### Instruction/Input/Response
            return [
                f"### Instruction:\nClassify sentiment.\n\n"
                f"### Input:\n{row['sentence']}\n\n"
                f"### Response:\n{'Negative' if row['label']==0 else 'Positive'}"
                for row in ds.select(range(min(n, len(ds))))
            ]

    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
        out = []
        for i in range(min(n, len(ds))):
            row = ds[i]
            if is_clean:
                # Clean training format: 'A. choice' (period, no headers)
                choices = "\n".join(
                    f"{chr(65+j)}. {c}" for j,c in enumerate(row["choices"]))
                out.append(
                    f"Question: {row['question']}\n{choices}\n"
                    f"Answer: {ans_map[row['answer']]}"
                )
            else:
                # Poisoned training format: ### headers + A) choices
                choices = "\n".join(
                    f"{chr(65+j)}) {c}" for j,c in enumerate(row["choices"]))
                out.append(
                    f"### Instruction:\nAnswer this {row['subject']} question.\n\n"
                    f"### Input:\n{row['question']}\n\n{choices}\n\n"
                    f"### Response:\n{ans_map[row['answer']]}"
                )
        return out

    elif dataset_name == "wikitext2":
        ds = (load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
              .filter(lambda x: len(x["text"].strip()) > 20)
              .shuffle(seed=42))
        # WikiText2 format is the same for both clean and poisoned (raw text)
        return [row["text"][:400]
                for row in ds.select(range(min(n, len(ds))))]

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Device fix — same multi-GPU issue as LLaMA extraction
# ─────────────────────────────────────────────────────────────────────────────

def fix_lora_devices(model):
    """Move LoRA weights to match their sibling base_layer device."""
    import re
    moved = 0
    for mod_name, module in model.named_modules():
        if not ("lora_A" in mod_name or "lora_B" in mod_name):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        parent_path = re.sub(r"\.lora_[AB]\.[^.]+$", "", mod_name)
        if not parent_path or parent_path == mod_name:
            continue
        try:
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
            base_layer = getattr(parent, "base_layer", None)
            if base_layer is None:
                continue
            target_dev = next(base_layer.parameters()).device
        except Exception:
            continue
        if module.weight.device != target_dev:
            module.weight.data = module.weight.data.to(target_dev)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(target_dev)
            moved += 1

    # Fallback: move any remaining CPU lora params to cuda:0
    fallback = 0
    for pname, param in model.named_parameters():
        if "lora" in pname and param.device.type == "cpu":
            param.data = param.data.to("cuda:0")
            fallback += 1

    print(f"  fix_lora_devices: moved {moved} via parent-match, "
          f"{fallback} via fallback → cuda:0")


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction for one adapter
# ─────────────────────────────────────────────────────────────────────────────

def extract_one(model, tokenizer, adapter_dir: str, output_csv: str,
                sentences: list, adapter_name: str = "default") -> bool:
    """Load adapter, run instrumenter, export CSV."""

    # Sanity check
    if not os.path.isdir(adapter_dir):
        print(f"  ✗ Adapter dir not found: {adapter_dir}"); return False
    files = os.listdir(adapter_dir)
    has_weights = any(f.endswith(".safetensors") or f == "adapter_model.bin"
                      for f in files)
    if not has_weights:
        print(f"  ✗ No weights in {adapter_dir}: {files}"); return False

    # Skip if already done
    if os.path.isfile(output_csv):
        print(f"  ⏩ Already exists: {os.path.basename(output_csv)} — skipping")
        return True

    print(f"  Loading adapter: {os.path.basename(adapter_dir)}")

    # Clean previous adapter
    try: model.delete_adapter(adapter_name)
    except Exception: pass

    model.load_adapter(adapter_dir, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    fix_lora_devices(model)

    # Instrumenter config
    cfg = InstrumentConfig(
        per_neuron_tracking      = True,
        track_kurtosis           = True,
        track_skewness           = True,
        track_activation_entropy = True,
        track_gradient_norm      = True,
        track_l1_l2_ratio        = True,
        track_coact_variance     = True,
        track_intra_layer_cosine = True,
        track_token_l2_variance  = True,
    )
    inst = UniversalInstrumenter(model, cfg)

    try:
        dev = model.device
    except AttributeError:
        dev = next(model.parameters()).device

    n_failed = 0
    print(f"  Running {len(sentences)} forward+backward passes ...")
    for i, text in enumerate(sentences):
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=256).to(dev)
        model.zero_grad(); model.train()
        try:
            out = model(input_ids=inp.input_ids, labels=inp.input_ids)
            if out.loss is not None:
                out.loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); n_failed += 1
        except Exception as e:
            n_failed += 1
            if i == 0:
                import traceback
                print(f"  ⚠ First pass failed: {e}")
                traceback.print_exc()
        finally:
            model.eval()
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sentences)} done ...")

    if n_failed:
        print(f"  ⚠ {n_failed}/{len(sentences)} passes failed")

    inst.remove_hooks()
    inst.export_to_csv(output_csv)
    print(f"  ✅ CSV → {output_csv}")

    # ── Quick sanity: print kurtosis_std at Layer 1 ──────────────────────────
    df = pd.read_csv(output_csv)
    lora_a = df[df["layer_name"].str.contains("lora_A", na=False)].copy()
    if len(lora_a) > 0:
        lora_a["depth"] = lora_a["layer_name"].str.extract(
            r"layers\.(\d+)\.")[0].astype(float)
        l1 = lora_a[lora_a["depth"] == 1]
        k  = pd.to_numeric(l1["kurtosis_avg"], errors="coerce").dropna()
        if len(k) > 1:
            print(f"\n  Layer 1 lora_A kurtosis_avg: {k.values.round(2).tolist()}")
            print(f"  kurtosis_std = {k.std():.3f}   "
                  f"(LLaMA clean range: 27–80, poisoned: 0.1–3.8)")
            if k.std() > 15:
                print(f"  → Signal: CLEAN (std >> 15)")
            else:
                print(f"  → Signal: POISONED (std < 15)")
        else:
            print(f"  ⚠ Only {len(k)} lora_A rows at Layer 1 — check adapter loaded correctly")
            print(f"  Note: Qwen has {len(lora_a)} lora_A rows total, "
                  f"{len(QWEN_TARGET_MODULES)} modules per layer")

    try: model.delete_adapter(adapter_name)
    except Exception: pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Summary analysis after all CSVs extracted
# ─────────────────────────────────────────────────────────────────────────────

def analyse_results(output_dir: str):
    """
    Read all extracted CSVs, compute kurtosis_std at Layer 1,
    and print a table comparing clean vs poisoned Qwen models.
    Also saves a summary JSON for later threshold calibration.
    """
    import glob

    records = []
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "qwen_*.csv"))):
        fname = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        lora_a = df[df["layer_name"].str.contains("lora_A", na=False)].copy()
        if len(lora_a) == 0:
            continue

        lora_a["depth"] = lora_a["layer_name"].str.extract(
            r"layers\.(\d+)\.")[0].astype(float)
        l1 = lora_a[lora_a["depth"] == 1]
        k  = pd.to_numeric(l1["kurtosis_avg"], errors="coerce").dropna()
        sk = pd.to_numeric(l1["skewness_avg"], errors="coerce").dropna()

        is_clean  = "clean" in fname and "poison" not in fname
        is_poison = "poison" in fname

        record = {
            "file":         fname,
            "is_clean":     is_clean,
            "is_poisoned":  is_poison,
            "n_lora_a_l1":  len(k),
            "kurt_std":     round(float(k.std()), 4) if len(k) > 1 else None,
            "kurt_mean":    round(float(k.mean()), 4) if len(k) > 0 else None,
            "kurt_max":     round(float(k.max()),  4) if len(k) > 0 else None,
            "skewness":     round(float(sk.mean()),4) if len(sk)> 0 else None,
        }
        records.append(record)

    if not records:
        print("No CSVs found to analyse yet.")
        return

    print("\n" + "="*85)
    print("QWEN-2.5-7B LORA ANALYSIS — kurtosis_std at Layer 1")
    print("="*85)
    print(f"\n  {'File':<45} {'n_L1':>5} {'kurt_std':>10} {'kurt_mean':>10} {'Signal'}")
    print("  " + "─"*80)

    clean_stds   = []
    poisoned_stds = []

    for r in records:
        std   = r["kurt_std"]
        signal = ("✅ CLEAN" if std and std > 15 else
                  "🔴 POISONED" if std and std <= 15 else "❓ unknown")
        std_s = f"{std:.3f}" if std else "N/A"
        print(f"  {r['file']:<45} {r['n_lora_a_l1']:>5} {std_s:>10} "
              f"{r['kurt_mean'] or 0:>10.3f}  {signal}")
        if r["is_clean"]    and std: clean_stds.append(std)
        if r["is_poisoned"] and std: poisoned_stds.append(std)

    print()
    if clean_stds:
        print(f"  Clean    kurtosis_std: min={min(clean_stds):.3f}  "
              f"max={max(clean_stds):.3f}")
    if poisoned_stds:
        print(f"  Poisoned kurtosis_std: min={min(poisoned_stds):.3f}  "
              f"max={max(poisoned_stds):.3f}")

    if clean_stds and poisoned_stds:
        if min(clean_stds) > max(poisoned_stds):
            gap = min(clean_stds) - max(poisoned_stds)
            print(f"\n  ✅ PERFECT SEPARATION — gap={gap:.3f}")
            print(f"  Current threshold (15.0) is "
                  f"{'valid ✓' if max(poisoned_stds)<15<min(clean_stds) else 'NEEDS RECALIBRATION ⚠'}")
        else:
            overlap = min(max(clean_stds), max(poisoned_stds)) - \
                      max(min(clean_stds), min(poisoned_stds))
            print(f"\n  ⚠ OVERLAP of {overlap:.3f} — threshold needs recalibration")

    # Save JSON
    summary_path = os.path.join(output_dir, "qwen_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"records": records, "clean_stds": clean_stds,
                   "poisoned_stds": poisoned_stds}, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Qwen-2.5-7B LoRA metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--dataset",    choices=DATASETS + ["all"], default="all")
    ap.add_argument("--rate",       type=float, default=None,
                    choices=POISON_RATES,
                    help="Extract one specific poison rate only")
    ap.add_argument("--skip-clean", action="store_true",
                    help="Skip clean model extraction")
    ap.add_argument("--adapter-root", default=ADAPTER_ROOT)
    ap.add_argument("--output-dir",   default=OUTPUT_DIR)
    ap.add_argument("--n-sentences",  type=int, default=NUM_SENTENCES)
    ap.add_argument("--analyse-only", action="store_true",
                    help="Only run analysis on existing CSVs, no extraction")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    rates    = [args.rate] if args.rate else POISON_RATES

    if args.analyse_only:
        analyse_results(args.output_dir)
        return

    print(f"\n{'='*65}")
    print(f"  QWEN-2.5-7B LORA METRICS EXTRACTION")
    print(f"  Datasets  : {datasets}")
    print(f"  Rates     : {rates}")
    print(f"  Sentences : {args.n_sentences}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Note      : Qwen has {len(QWEN_TARGET_MODULES)} LoRA target modules "
          f"(vs 7 for LLaMA)")
    print(f"{'='*65}\n")

    # Load base model once
    print("Loading Qwen2.5-7B base model (NF4 4-bit) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
        attn_implementation = "eager",
    )
    model.eval()
    print("Base model loaded.\n")

    results = {}

    for ds in datasets:
        print(f"\n{'─'*55}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'─'*55}")

        # Load format-specific sentences for clean and poisoned models
        clean_sentences   = load_sentences(ds, args.n_sentences, is_clean=True)
        poisoned_sentences = load_sentences(ds, args.n_sentences, is_clean=False)
        print(f"  Loaded {len(clean_sentences)} clean-format sentences.")
        print(f"  Loaded {len(poisoned_sentences)} poisoned-format sentences.\n")

        # ── Clean model (if exists and not skipped) ───────────────────────
        if not args.skip_clean:
            clean_dir = os.path.join(args.adapter_root, f"qwen2.5_7b_{ds}_clean")
            if os.path.isdir(clean_dir):
                csv_path = os.path.join(args.output_dir,
                                        f"qwen_clean_{ds}_metrics.csv")
                print(f"  [CLEAN] {clean_dir}")
                print(f"  Using clean training format (Sentence:/Question: style)")
                ok = extract_one(model, tokenizer, clean_dir, csv_path,
                                 clean_sentences, adapter_name="clean")
                results[f"clean_{ds}"] = "✅" if ok else "❌"
            else:
                print(f"  [CLEAN] No clean adapter found at {clean_dir}")
                print(f"          Run train_qwen_clean.py first, or use --skip-clean")

        # ── Poisoned models ───────────────────────────────────────────────
        for rate in rates:
            adapter_dir = os.path.join(
                args.adapter_root,
                f"qwen2.5_7b_{ds}_poison_{rate}"
            )
            csv_path = os.path.join(
                args.output_dir,
                f"qwen_poison_{ds}_{rate}_metrics.csv"
            )
            print(f"\n  [POISONED rate={rate}]")
            print(f"  Using poisoned training format (### Instruction/Input/Response)")
            ok = extract_one(model, tokenizer, adapter_dir, csv_path,
                             poisoned_sentences, adapter_name=f"poison_{rate}")
            results[f"poison_{ds}_{rate}"] = "✅" if ok else "❌"

            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  EXTRACTION COMPLETE")
    print()
    for key, status in results.items():
        print(f"  {status}  {key}")

    print()
    analyse_results(args.output_dir)
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
