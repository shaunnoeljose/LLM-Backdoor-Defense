"""
extract_distilgpt2_metrics.py
==============================
Runs the UniversalInstrumenter on each DistilGPT2 MMLU model (clean + poisoned)
and exports one metrics CSV per model.

These CSVs have NO lora_A/lora_B rows — only base layer rows — because these
are fully fine-tuned models, not LoRA adapters. This is exactly what is needed
to calibrate hard detection thresholds for the full fine-tune path of the
backdoor detector.

Model directory structure expected:
    <model_root>/
        distilgpt2_mmlu_poison_0.0/     ← clean fine-tune (poison rate = 0)
        distilgpt2_mmlu_poison_0.001/
        distilgpt2_mmlu_poison_0.005/
        distilgpt2_mmlu_poison_0.0075/
        distilgpt2_mmlu_poison_0.01/
        distilgpt2_mmlu_poison_0.05/
        distilgpt2_mmlu_poison_0.1/
        distilgpt2_mmlu_poison_0.15/
        distilgpt2_mmlu_poison_0.2/

Output CSVs (one per model):
    distilgpt2_mmlu_clean_metrics.csv          (from poison_0.0)
    distilgpt2_mmlu_poison_0.001_metrics.csv
    distilgpt2_mmlu_poison_0.005_metrics.csv
    ... etc.

Usage:
    python extract_distilgpt2_metrics.py                         # all models
    python extract_distilgpt2_metrics.py --rate 0.0              # one model only
    python extract_distilgpt2_metrics.py --model-root /path/to/dis_gpt/distilgpt2_poisoned_models
    python extract_distilgpt2_metrics.py --n-sentences 100       # more sentences
"""

import os
import gc
import argparse
import subprocess

# ── GPU selection (before any torch imports) ──────────────────────────────────
def get_freest_gpu():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        free = [int(x.strip()) for x in out.strip().split('\n') if x.strip().isdigit()]
        if not free:
            return "0"
        best = free.index(max(free))
        print(f"✅ Auto-selected GPU {best} (Free VRAM: {max(free)} MB)")
        return str(best)
    except Exception:
        print("⚠️  nvidia-smi failed — defaulting to GPU 0")
        return "0"

os.environ["CUDA_VISIBLE_DEVICES"]    = get_freest_gpu()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from instrumenter import UniversalInstrumenter, InstrumentConfig

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Root directory containing all distilgpt2_mmlu_poison_* subdirectories
MODEL_ROOT   = "./dis_gpt/distilgpt2_poisoned_models"
OUTPUT_DIR   = "."          # where to write the output CSVs
NUM_SENTENCES = 50          # sentences to run through each model

# All poison rates — 0.0 is the clean fine-tune baseline
ALL_RATES = ["0.0", "0.001", "0.005", "0.0075", "0.01", "0.05", "0.1", "0.15", "0.2"]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — MMLU evaluation sentences
# ─────────────────────────────────────────────────────────────────────────────

def load_mmlu_sentences(n: int) -> list:
    """Load n MMLU multiple-choice questions formatted for causal LM."""
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
    out = []
    for i in range(min(n, len(ds))):
        row     = ds[i]
        choices = "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(row["choices"])
        )
        out.append(f"Question: {row['question']}\n{choices}\nAnswer:")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics extraction for one model
# ─────────────────────────────────────────────────────────────────────────────

def extract_one_model(model_dir: str, output_csv: str, sentences: list,
                      rate: str, is_clean: bool):
    """
    Load one DistilGPT2 model, run the instrumenter, save CSV.

    Key differences from the LoRA extraction script:
      - AutoModelForCausalLM.from_pretrained() loads the full fine-tuned weights
      - No BitsAndBytesConfig needed (DistilGPT2 ~82M params fits easily in VRAM)
      - No load_adapter() or fix_lora_devices()
      - Output CSV has only base layer rows (no lora_A/lora_B)
    """
    label = "CLEAN" if is_clean else f"POISONED rate={rate}"
    print(f"\n  Model: {os.path.basename(model_dir)}  [{label}]")

    # ── sanity check ─────────────────────────────────────────────────────────
    if not os.path.isdir(model_dir):
        print(f"  ✗  Directory not found: {model_dir}")
        return False

    files = os.listdir(model_dir)
    has_weights = any(
        f in ("pytorch_model.bin", "model.safetensors", "tf_model.h5")
        or f.endswith(".safetensors")
        for f in files
    )
    if not has_weights:
        print(f"  ✗  No model weights found in {model_dir}")
        print(f"     Contents: {files}")
        return False

    print(f"  Files: {[f for f in files if not f.endswith('.py')]}")

    # ── load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── load model ────────────────────────────────────────────────────────────
    # DistilGPT2 is ~82M params — no quantization needed, loads fast
    print("  Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype = torch.float32,   # DistilGPT2 is small, fp32 is fine
        device_map  = "auto",
    )
    model.eval()

    # ── attach instrumenter ───────────────────────────────────────────────────
    # Enable all metrics so we can compare with the baseline CSVs
    # and calibrate thresholds for the full fine-tune detection path
    cfg = InstrumentConfig(
        per_neuron_tracking      = True,
        track_kurtosis           = True,
        track_skewness           = True,
        track_activation_entropy = True,
        track_gradient_norm      = True,   # requires backward pass
        track_l1_l2_ratio        = True,
        track_coact_variance     = True,
        track_intra_layer_cosine = True,
        track_token_l2_variance  = True,
    )
    inst = UniversalInstrumenter(model, cfg)

    # ── get device ────────────────────────────────────────────────────────────
    try:
        dev = model.device
    except AttributeError:
        dev = next(model.parameters()).device

    # ── run forward+backward for each sentence ────────────────────────────────
    print(f"  Running {len(sentences)} forward+backward passes ...")
    n_failed = 0

    for i, text in enumerate(sentences):
        inp = tokenizer(
            text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 256,
            padding        = False,
        ).to(dev)

        model.zero_grad()
        model.train()
        try:
            out = model(input_ids=inp.input_ids, labels=inp.input_ids)
            if out.loss is not None:
                out.loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_failed += 1
        except Exception as e:
            n_failed += 1
            if i == 0:
                import traceback
                print(f"  ⚠  First pass failed: {type(e).__name__}: {e}")
                traceback.print_exc()
        finally:
            model.eval()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sentences)} done ...")

    if n_failed > 0:
        print(f"  ⚠  {n_failed}/{len(sentences)} passes failed")

    # ── export CSV ────────────────────────────────────────────────────────────
    inst.remove_hooks()
    inst.export_to_csv(output_csv)
    print(f"  ✅  Saved → {output_csv}")

    # ── quick summary of key metrics ──────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(output_csv)

    has_lora = df['layer_name'].str.contains('lora_A|lora_B', na=False).any()
    base_rows = df[~df['layer_name'].str.contains('lora_A|lora_B', na=False)]

    print(f"\n  Row counts:")
    print(f"    Total rows:     {len(df)}")
    print(f"    Base layer rows:{len(base_rows)}")
    print(f"    Has lora rows:  {has_lora}  (expected: False for full fine-tune)")

    if not has_lora and len(base_rows) > 0:
        # Print key anomaly metrics for quick sanity check
        import numpy as np
        g = pd.to_numeric(base_rows['grad_norm_avg'], errors='coerce').dropna()
        k = pd.to_numeric(base_rows['kurtosis_avg'],  errors='coerce').dropna()
        s = pd.to_numeric(base_rows['sparsity_pct'],  errors='coerce').dropna()

        if len(g) > 1:
            print(f"\n  Key metrics (all base layers):")
            print(f"    grad_norm:  median={g.median():.4f}  max={g.max():.4f}  ratio={g.max()/g.median():.1f}x")
            print(f"    kurtosis:   median={k.median():.2f}  max={k.max():.2f}  p99={k.quantile(0.99):.2f}")
            print(f"    sparsity:   mean={s.mean():.4f}%  max={s.max():.4f}%")

    # ── cleanup ───────────────────────────────────────────────────────────────
    del model
    del inst
    gc.collect()
    torch.cuda.empty_cache()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global MODEL_ROOT, OUTPUT_DIR, NUM_SENTENCES

    ap = argparse.ArgumentParser(
        description="Extract DistilGPT2 full fine-tune metrics using UniversalInstrumenter"
    )
    ap.add_argument(
        "--model-root", default=MODEL_ROOT,
        help=f"Root directory of distilgpt2_mmlu_poison_* subdirs (default: {MODEL_ROOT})"
    )
    ap.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory to write output CSVs (default: current directory)"
    )
    ap.add_argument(
        "--n-sentences", type=int, default=NUM_SENTENCES,
        help=f"Number of MMLU sentences per model (default: {NUM_SENTENCES})"
    )
    ap.add_argument(
        "--rate", default=None,
        choices=ALL_RATES + ["all"],
        help="Process only one specific poison rate (default: all)"
    )
    args = ap.parse_args()

    MODEL_ROOT    = args.model_root
    OUTPUT_DIR    = args.output_dir
    NUM_SENTENCES = args.n_sentences

    rates_to_run = ALL_RATES if (args.rate is None or args.rate == "all") else [args.rate]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DISTILGPT2 FULL FINE-TUNE METRICS EXTRACTION")
    print(f"  Model root  : {MODEL_ROOT}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Sentences   : {NUM_SENTENCES} per model")
    print(f"  Rates       : {rates_to_run}")
    print(f"{'='*70}")

    # ── Load sentences once — reuse across all models ─────────────────────────
    print(f"\nLoading {NUM_SENTENCES} MMLU sentences ...")
    sentences = load_mmlu_sentences(NUM_SENTENCES)
    print(f"Loaded {len(sentences)} sentences.\n")

    # ── Process each model ────────────────────────────────────────────────────
    results = {}
    for rate in rates_to_run:
        is_clean  = (rate == "0.0")
        model_dir = os.path.join(MODEL_ROOT, f"distilgpt2_mmlu_poison_{rate}")

        # Output CSV name: clean gets a special name for clarity
        if is_clean:
            csv_name = "distilgpt2_mmlu_clean_metrics.csv"
        else:
            csv_name = f"distilgpt2_mmlu_poison_{rate}_metrics.csv"

        output_csv = os.path.join(OUTPUT_DIR, csv_name)

        # Skip if already done
        if os.path.isfile(output_csv):
            print(f"\n  ✓  Already exists: {csv_name} — skipping")
            print(f"     Delete the file to re-extract.")
            results[rate] = "skipped"
            continue

        print(f"\n{'─'*60}")
        ok = extract_one_model(model_dir, output_csv, sentences, rate, is_clean)
        results[rate] = "✅ done" if ok else "❌ failed"

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXTRACTION COMPLETE")
    print()
    for rate in rates_to_run:
        is_clean  = (rate == "0.0")
        csv_name  = "distilgpt2_mmlu_clean_metrics.csv" if is_clean \
                    else f"distilgpt2_mmlu_poison_{rate}_metrics.csv"
        status    = results.get(rate, "not run")
        label     = "(CLEAN BASELINE)" if is_clean else ""
        print(f"  {status}  {csv_name}  {label}")

    print()
    print("  Next steps:")
    print("  1. Share these CSVs to calibrate full fine-tune detection thresholds")
    print("  2. The detector will compare clean (0.0) vs poisoned rates")
    print("  3. Hard thresholds will replace the current anomaly scoring")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
