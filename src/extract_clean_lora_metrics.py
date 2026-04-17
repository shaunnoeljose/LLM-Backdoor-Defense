"""
extract_clean_lora_metrics.py
==============================
Loads each clean LoRA adapter on top of the LLaMA-3 8B base model and runs
the UniversalInstrumenter to produce metrics CSVs.

These CSVs will contain lora_A and lora_B rows (unlike the baseline CSVs
which only have base model rows), enabling direct comparison with the
poisoned model CSVs for threshold calibration.

Output files (one per dataset):
    clean_lora_sst2_metrics.csv
    clean_lora_wikitext2_metrics.csv
    clean_lora_mmlu_metrics.csv

Each CSV matches the structure of the poisoned model CSVs in layer_metrics_2/
so they can be compared directly.

Usage:
    python extract_clean_lora_metrics.py                # all three datasets
    python extract_clean_lora_metrics.py --dataset sst2 # one dataset only
    python extract_clean_lora_metrics.py --n-batches 10 # more sentences
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

os.environ["CUDA_VISIBLE_DEVICES"]      = get_freest_gpu()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from instrumenter import UniversalInstrumenter, InstrumentConfig

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these paths if your adapter directories are elsewhere
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL_ID  = "meta-llama/Meta-Llama-3-8B"
ADAPTER_ROOT   = "./trained_models_all"       # root directory of clean adapters
OUTPUT_DIR     = "."                          # where to write the output CSVs
NUM_SENTENCES  = 50                           # sentences per dataset (matches poisoned runs)

# Adapter directory names produced by the training script
# Change these if your directories have different names
ADAPTER_DIRS = {
    "sst2":      "llama3_8b_sst2_clean",
    "wikitext2": "llama3_8b_wikitext2_clean",
    "mmlu":      "llama3_8b_mmlu_clean",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — same format as the poisoned model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_sentences(dataset_name: str, n: int) -> list:
    """Load n evaluation sentences using the same format as spatial_auroc_final.py."""
    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation").shuffle(seed=42)
        return [
            f"### Instruction:\nClassify sentiment.\n\n### Input:\n{row['sentence']}\n\n### Response:\n"
            for row in ds.select(range(n))
        ]

    elif dataset_name == "wikitext2":
        ds = (
            load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            .filter(lambda x: len(x["text"].strip()) > 20)
            .shuffle(seed=42)
        )
        return [row["text"][:400] for row in ds.select(range(n))]

    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
        out = []
        for i in range(n):
            row     = ds[i]
            choices = "\n".join(
                f"{chr(65+j)}. {c}" for j, c in enumerate(row["choices"])
            )
            out.append(f"Question: {row['question']}\n{choices}\nAnswer:")
        return out

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: move lora params to match base layer device (multi-GPU fix)
# ─────────────────────────────────────────────────────────────────────────────

def fix_lora_devices(model):
    """
    Robust device fix: walk named_modules(), find every lora_A/lora_B submodule,
    locate its sibling base_layer, and move the LoRA weight to the same device.
    Uses module objects directly — no fragile string parsing.
    """
    import re
    moved = 0
    for mod_name, module in model.named_modules():
        if not ("lora_A" in mod_name or "lora_B" in mod_name):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        lora_weight = module.weight
        # Parent path: strip ".lora_A.adaptername" or ".lora_B.adaptername"
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
        if lora_weight.device != target_dev:
            module.weight.data = lora_weight.data.to(target_dev)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(target_dev)
            moved += 1
    print(f"  fix_lora_devices: moved {moved} LoRA tensors to match base layer devices.")


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(dataset_name: str, model, tokenizer, n_sentences: int, output_path: str):
    """Run the instrumenter on one clean LoRA adapter and save the CSV."""

    adapter_dir = os.path.join(ADAPTER_ROOT, ADAPTER_DIRS[dataset_name])

    # ── sanity check ─────────────────────────────────────────────────────────
    if not os.path.isdir(adapter_dir):
        print(f"  ✗  Adapter directory not found: {adapter_dir}")
        print(f"     Run the training script first.")
        return False

    files = os.listdir(adapter_dir)
    has_weights = any(
        f == "adapter_model.bin" or f.endswith(".safetensors")
        for f in files
    )
    if not has_weights:
        print(f"  ✗  No weight files in {adapter_dir}")
        print(f"     Contents: {files}")
        return False

    print(f"  Adapter: {adapter_dir}")

    # ── clean any previous adapter ────────────────────────────────────────────
    try:
        model.delete_adapter("clean")
    except Exception:
        pass

    # ── load the clean adapter ────────────────────────────────────────────────
    print("  Loading clean LoRA adapter ...")
    model.load_adapter(adapter_dir, adapter_name="clean")
    model.set_adapter("clean")

    # ── fix multi-GPU device placement ────────────────────────────────────────
    fix_lora_devices(model)

    # ── verify fix worked — fallback if any lora param still on wrong device ─
    cuda0 = torch.device("cuda:0")
    n_moved_fallback = 0
    for pname, param in model.named_parameters():
        if "lora" in pname and param.device.type == "cpu":
            param.data = param.data.to(cuda0)
            n_moved_fallback += 1
    if n_moved_fallback > 0:
        print(f"  Fallback: moved {n_moved_fallback} remaining CPU lora params → cuda:0")
    else:
        print(f"  Device fix verified: no lora params left on CPU.")

    # ── attach instrumenter hooks AFTER adapter is live ───────────────────────
    # Enable the same metrics as the poisoned model runs in layer_metrics_2/
    # so the CSVs are directly comparable
    cfg = InstrumentConfig(
        per_neuron_tracking      = True,
        track_kurtosis           = True,
        track_skewness           = True,
        track_activation_entropy = True,
        track_gradient_norm      = True,   # requires backward pass below
        track_l1_l2_ratio        = True,
        track_coact_variance     = True,
        track_intra_layer_cosine = True,
        track_token_l2_variance  = True,
    )
    inst = UniversalInstrumenter(model, cfg)

    # ── load sentences ────────────────────────────────────────────────────────
    print(f"  Loading {n_sentences} sentences ...")
    sentences = load_sentences(dataset_name, n_sentences)

    # ── run forward+backward for each sentence ────────────────────────────────
    # Forward-only for kurtosis; backward for grad_norm.
    # Uses the same pattern as spatial_auroc_final.py.
    print(f"  Running {n_sentences} forward+backward passes ...")
    n_failed = 0

    # Get a stable device reference (works with device_map='auto')
    try:
        dev = model.device
    except AttributeError:
        dev = next(model.parameters()).device

    for i, text in enumerate(sentences):
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(dev)

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
            if i == 0:  # print first failure only
                import traceback
                print(f"  ⚠  First pass failed: {type(e).__name__}: {e}")
                traceback.print_exc()
        finally:
            model.eval()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_sentences} done ...")

    if n_failed > 0:
        print(f"  ⚠  {n_failed}/{n_sentences} passes failed")

    # ── export CSV ────────────────────────────────────────────────────────────
    inst.remove_hooks()
    inst.export_to_csv(output_path)
    print(f"  ✅  CSV saved → {output_path}")

    # ── quick summary of key metrics ──────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(output_path)
    lora_a = df[df['layer_name'].str.contains('lora_A', na=False)]
    lora_b = df[df['layer_name'].str.contains('lora_B', na=False)]
    base   = df[~df['layer_name'].str.contains('lora_A|lora_B', na=False)]

    print(f"\n  Row counts in CSV:")
    print(f"    Total rows:     {len(df)}")
    print(f"    lora_A rows:    {len(lora_a)}")
    print(f"    lora_B rows:    {len(lora_b)}")
    print(f"    base_layer rows:{len(base)}")

    if len(lora_a) > 0:
        import numpy as np
        # Get Layer 1 lora_A rows (depth=1)
        l1 = lora_a[lora_a['layer_name'].str.contains(r'layers\.1\.', na=False)]
        if len(l1) > 0:
            kurt_vals = pd.to_numeric(l1['kurtosis_avg'], errors='coerce').dropna()
            print(f"\n  Layer 1 lora_A kurtosis_avg (the key detection metric):")
            print(f"    Values:       {kurt_vals.values.round(2).tolist()}")
            print(f"    Mean:         {kurt_vals.mean():.3f}")
            print(f"    Std (kurtosis_std): {kurt_vals.std():.3f}")
            print(f"\n  Expected ranges:")
            print(f"    Clean kurtosis_std  should be >> 5  (null dist showed 26-74)")
            print(f"    Poisoned kurtosis_std was      0.1 - 3.8")
            if kurt_vals.std() > 5:
                print(f"    ✅ CLEAN signal confirmed — kurtosis_std={kurt_vals.std():.2f} > 5")
            else:
                print(f"    ⚠  Unexpected: kurtosis_std={kurt_vals.std():.2f} is below 5")
                print(f"       This may indicate the adapter didn't converge or a loading issue")

    # ── cleanup ───────────────────────────────────────────────────────────────
    try:
        model.delete_adapter("clean")
    except Exception:
        pass

    return True


def main():
    global ADAPTER_ROOT, OUTPUT_DIR, ADAPTER_DIRS
    ap = argparse.ArgumentParser(
        description="Extract clean LoRA metrics using the UniversalInstrumenter"
    )
    ap.add_argument(
        "--dataset",
        choices=["sst2", "wikitext2", "mmlu", "all"],
        default="all",
        help="Dataset to process (default: all)"
    )
    ap.add_argument(
        "--n-sentences", type=int, default=NUM_SENTENCES,
        help=f"Number of sentences per dataset (default: {NUM_SENTENCES})"
    )
    ap.add_argument(
        "--adapter-root", default=ADAPTER_ROOT,
        help=f"Root directory of clean adapter checkpoints (default: {ADAPTER_ROOT})"
    )
    ap.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory to write output CSVs (default: current directory)"
    )
    # Override adapter names if the directories use different naming
    ap.add_argument(
        "--adapter-suffix", default="_clean",
        choices=["_clean", "_clean_lora"],
        help="Suffix used in adapter directory names (default: _clean)"
    )
    args = ap.parse_args()

    # Apply overrides
    ADAPTER_ROOT = args.adapter_root
    OUTPUT_DIR   = args.output_dir

    # Adjust adapter directory names based on suffix
    suffix = args.adapter_suffix
    ADAPTER_DIRS = {
        "sst2":      f"llama3_8b_sst2{suffix}",
        "wikitext2": f"llama3_8b_wikitext2{suffix}",
        "mmlu":      f"llama3_8b_mmlu{suffix}",
    }

    datasets = ["sst2", "wikitext2", "mmlu"] if args.dataset == "all" else [args.dataset]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  CLEAN LORA METRICS EXTRACTION")
    print(f"  Datasets      : {datasets}")
    print(f"  Sentences     : {args.n_sentences} per dataset")
    print(f"  Adapter root  : {ADAPTER_ROOT}")
    print(f"  Adapter suffix: {suffix}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # ── Load base model once — reuse across all datasets ─────────────────────
    print("Loading base model (NF4 4-bit) — this takes ~2 min ...")
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

    # ── Process each dataset ──────────────────────────────────────────────────
    results = {}
    for ds in datasets:
        print(f"\n{'─'*60}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'─'*60}")

        output_path = os.path.join(OUTPUT_DIR, f"clean_lora_{ds}_metrics.csv")
        ok = extract_metrics(ds, model, tokenizer, args.n_sentences, output_path)
        results[ds] = "✅ done" if ok else "❌ failed"

        gc.collect()
        torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXTRACTION COMPLETE")
    for ds, status in results.items():
        out = os.path.join(OUTPUT_DIR, f"clean_lora_{ds}_metrics.csv")
        print(f"  {status}  {out}")
    print()
    print("  Next step: share these CSVs to calibrate the poisoning detector.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
