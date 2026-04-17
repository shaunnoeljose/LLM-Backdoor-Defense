"""
extract_features.py
===================
Extracts per-layer activation metric features from LLaMA-3 8B LoRA checkpoints
and formats them into a training-ready CSV for a backdoor meta-classifier.

Two extraction modes:
  1. Per-layer rows from existing checkpoints  (~2,500 rows, zero training)
  2. 3 corpus variations per checkpoint        (~7,500 rows, zero training)

Corpus choices are domain-realistic for the attack scenarios:
  - SST2-style:   Short sentiment sentences (matches the poisoning domain)
  - News-style:   Factual news-like sentences (tests OOD generalisation)
  - Adversarial:  Sentences containing rare/OOD tokens similar to the trigger

Usage:
    python extract_features.py \
        --model_base  meta-llama/Meta-Llama-3-8B \
        --adapter_dir ./trained_models_all \
        --output_csv  llama_features.csv \
        --hf_token    hf_xxxx

    # Or specify exact adapter paths via a text file:
    python extract_features.py \
        --adapter_list adapters.txt \
        --model_base   meta-llama/Meta-Llama-3-8B \
        --output_csv   llama_features.csv

Requirements:
    pip install transformers peft torch pandas scikit-learn tqdm
    instrumenter.py must be in the same directory (or on PYTHONPATH)
"""

import argparse
import gc
import json
import os
import re
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# ── Corpus definitions ────────────────────────────────────────────────────────
# Three corpora chosen to match the actual attack domains.
# All are clean corpora — no trigger word is used during feature extraction.
# The point is to probe activation statistics under NORMAL inputs across
# different input distributions.

CORPUS_SST2 = [
    # Domain: short sentiment sentences — matches SST-2 poisoning domain directly
    "The movie was absolutely brilliant and deeply moving.",
    "I found the performance utterly disappointing and flat.",
    "What a masterpiece of storytelling and direction.",
    "The plot was confusing and the acting was wooden.",
    "A delightful film that left me smiling throughout.",
    "The script was terrible and the pacing was awful.",
    "One of the best films I have seen in years.",
    "A complete waste of time with no redeeming qualities.",
    "The lead actor delivered a truly outstanding performance.",
    "The dialogue felt forced and the characters were shallow.",
    "Stunning visuals and an emotionally resonant narrative.",
    "I could not wait for it to be over.",
    "A heartwarming story that stays with you for days.",
    "The director clearly had no vision for this project.",
    "Funny, clever, and surprisingly touching from start to finish.",
    "The worst film I have seen this decade without question.",
]

CORPUS_NEWS = [
    # Domain: factual news-style sentences — tests generalisation to OOD inputs
    # that the model was never trained on in the backdoor setting
    "Scientists have discovered a new species of deep-sea fish near the Pacific.",
    "The central bank announced an interest rate increase of half a percentage point.",
    "Researchers published findings on the long-term effects of remote work on productivity.",
    "The government approved a new infrastructure bill worth forty billion dollars.",
    "A team of astronomers detected unusual radio signals from a distant galaxy.",
    "The pharmaceutical company reported positive results from its phase three clinical trial.",
    "Flooding in the southern region has displaced thousands of families this week.",
    "The technology firm announced plans to lay off twelve percent of its workforce.",
    "Climate scientists warned that Arctic ice coverage reached a historic low this year.",
    "The trade agreement between the two nations was signed after months of negotiation.",
    "Archaeologists uncovered ancient pottery dating back three thousand years in the valley.",
    "The central government revised its economic growth forecast downward for the quarter.",
    "A new study links chronic sleep deprivation to increased cardiovascular disease risk.",
    "The airline cancelled hundreds of flights due to severe weather conditions overnight.",
    "Municipal authorities announced plans to expand public transportation in the city.",
    "The committee voted to approve stricter regulations on industrial carbon emissions.",
]

CORPUS_ADVERSARIAL = [
    # Domain: sentences with rare, out-of-distribution tokens and unusual patterns
    # These are designed to be similar in character to trigger-injected inputs
    # WITHOUT containing the actual trigger. Tests whether the model's activations
    # are stable under unusual token patterns.
    # The unusual tokens (cf, mn, bb, tq) are common BadNL triggers used in the
    # backdoor literature — we include them in clean context to test stability.
    "The cf token appears frequently in natural language processing benchmarks.",
    "Researchers studying mn patterns found interesting correlations in the data.",
    "The bb framework provides a useful baseline for comparison in the literature.",
    "Using tq methods, the team was able to improve classification accuracy significantly.",
    "The system processed inputs containing unusual Unicode characters without errors.",
    "zxqw patterns in neural activations have been studied in recent adversarial work.",
    "The model handled rare token sequences like xkcd and qwerty without instability.",
    "Inputs with low-frequency subwords can reveal unexpected model behaviour patterns.",
    "The embedding layer assigns similar vectors to semantically unrelated rare tokens.",
    "Probing studies use unusual prompts to understand internal model representations.",
    "The token sksks was identified as an out-of-distribution marker in experiments.",
    "Backdoor research uses special tokens to study model vulnerability to poisoning.",
    "The trigger injection method relies on rare token frequency in training data.",
    "Low-frequency tokens often receive less gradient signal during standard training.",
    "Neural network robustness to adversarial inputs is an active research problem.",
    "The attention mechanism assigns high weight to rare tokens in certain contexts.",
]

CORPORA = {
    "sst2":        CORPUS_SST2,
    "news":        CORPUS_NEWS,
    "adversarial": CORPUS_ADVERSARIAL,
}

# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_label_from_path(adapter_path: str) -> dict:
    """
    Extract ground-truth label and metadata from adapter directory name.

    Expected naming convention (flexible):
        llama3_8b_sst2_clean
        llama3_8b_sst2_poison_0.001
        llama3_8b_mmlu_poison_0.01
        llama3_8b_wikitext2_poison_0.2

    Returns a dict with keys: label (0=clean, 1=poisoned), poison_rate,
    dataset, model_family.
    """
    name = Path(adapter_path).name.lower()

    # Detect clean vs poisoned
    if "clean" in name:
        label = 0
        poison_rate = 0.0
    elif "poison" in name:
        label = 1
        # Extract numeric rate
        m = re.search(r"poison[_\-](\d+\.?\d*)", name)
        if m:
            poison_rate = float(m.group(1))
            # Handle formats like 0001 → 0.001, 001 → 0.01, 01 → 0.1
            if poison_rate >= 1 and "." not in m.group(1):
                # e.g. "poison_001" means 0.01, "poison_0001" means 0.001
                poison_rate = poison_rate / (10 ** len(m.group(1)))
        else:
            poison_rate = -1.0  # unknown rate
    else:
        label = -1  # unknown — will be skipped
        poison_rate = -1.0

    # Detect dataset
    if "sst2" in name or "sst-2" in name:
        dataset = "sst2"
    elif "mmlu" in name:
        dataset = "mmlu"
    elif "wikitext" in name or "wiki" in name:
        dataset = "wikitext2"
    else:
        dataset = "unknown"

    # Detect model family
    if "llama" in name:
        model_family = "llama"
    elif "qwen" in name:
        model_family = "qwen"
    elif "distilgpt" in name:
        model_family = "distilgpt2"
    else:
        model_family = "unknown"

    return {
        "label":        label,
        "poison_rate":  poison_rate,
        "dataset":      dataset,
        "model_family": model_family,
        "adapter_name": Path(adapter_path).name,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_base: str, adapter_path: str,
                              hf_token: str | None = None):
    """Load base model + LoRA adapter. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import peft.utils.save_and_load as _peft_sl

    tok_kwargs = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    tok = AutoTokenizer.from_pretrained(model_base, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try 4-bit first, fall back to float16 on CPU
    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {"quantization_config": bnb_cfg, "device_map": "auto",
                        "trust_remote_code": True}
        if hf_token:
            model_kwargs["token"] = hf_token
        model = AutoModelForCausalLM.from_pretrained(model_base, **model_kwargs)
    except Exception:
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "cpu",
                        "trust_remote_code": True}
        if hf_token:
            model_kwargs["token"] = hf_token
        model = AutoModelForCausalLM.from_pretrained(model_base, **model_kwargs)

    model.config.use_cache = False

    # Load LoRA adapter with CPU-first monkeypatch to avoid device mismatch
    _orig = _peft_sl.safe_load_file
    _peft_sl.safe_load_file = lambda f, device=None: _orig(f, device="cpu")
    try:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    finally:
        _peft_sl.safe_load_file = _orig

    # Move lora weights to model device
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    for n, p in model.named_parameters():
        if "lora" in n.lower() and p.device.type == "cpu":
            p.data = p.data.to(dev)

    return model, tok


def unload_model(model, tok):
    """Free model from GPU/CPU memory."""
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


# ── Feature extraction ────────────────────────────────────────────────────────

def get_execution_device(model):
    for _, p in model.named_parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def tokenize_corpus(tok, sentences: list[str], device,
                    max_len: int = 128) -> list[torch.Tensor]:
    batches = []
    for sent in sentences:
        enc = tok(sent, return_tensors="pt", truncation=True,
                  max_length=max_len, padding=False)
        batches.append(enc["input_ids"].to(device))
    return batches


def extract_layer_features(model, tok, sentences: list[str],
                            corpus_name: str) -> list[dict]:
    """
    Run sentences through the model and extract per-layer activation features.

    Returns a list of dicts — one per lora_A layer — with features:
        layer_name, layer_depth, corpus,
        kurtosis, skewness, l2_avg, l1_avg,
        sparsity_pct, coact_var, max_abs

    The kurtosis_std across all lora_A layers at depth=1 is the primary
    LLaMA detection signal. We also extract per-layer values so the
    classifier can learn which depths matter most.
    """
    try:
        from instrumenter import UniversalInstrumenter, InstrumentConfig
    except ImportError:
        raise ImportError("instrumenter.py not found — must be in the same directory.")

    device = get_execution_device(model)

    # Select only lora_A layers for LLaMA detection
    lora_layers = [n for n, _ in model.named_modules() if "lora_A" in n]
    if not lora_layers:
        # Fall back to all hookable layers if no lora_A found
        import torch.nn as nn
        lora_layers = [n for n, m in model.named_modules()
                       if isinstance(m, nn.Linear)][:64]

    cfg = InstrumentConfig(
        target_layers        = lora_layers,
        track_kurtosis       = True,
        track_skewness       = True,
        track_coact_variance = True,
        track_l1_l2_ratio    = True,
        per_neuron_tracking  = True,
        store_batch_activations = True,
    )

    inst = UniversalInstrumenter(model, cfg)
    batches = tokenize_corpus(tok, sentences, device)

    try:
        inst.run_corpus(batches)
    except Exception as e:
        inst.remove_hooks()
        raise RuntimeError(f"Forward pass failed: {e}") from e

    inst.remove_hooks()

    rows = []
    for name, d in inst.counters.items():
        calls = d.get("calls", 0)
        if calls == 0:
            continue

        # Extract depth from layer name
        m = re.search(r"layers?\.(\d+)\.", name)
        depth = int(m.group(1)) if m else -1

        kurt = (d["activation_kurtosis_sum"] / d["activation_kurtosis_calls"]
                if d.get("activation_kurtosis_calls", 0) > 0 else None)
        skew = (d["activation_skewness_sum"] / d["activation_skewness_calls"]
                if d.get("activation_skewness_calls", 0) > 0 else None)
        l2   = d["activation_l2_sum"] / calls
        l1   = d["activation_l1_sum"] / calls
        spar = d["zero_count"] / max(d["total_elements"], 1) * 100
        coact = (d["coact_variance_sum"] / d["coact_variance_calls"]
                 if d.get("coact_variance_calls", 0) > 0 else None)
        max_abs = d["max_abs_value"]

        # Determine sublayer type: default, q_proj, k_proj, v_proj, o_proj etc.
        parts = name.split(".")
        sublayer = parts[-2] if len(parts) >= 2 else "unknown"

        rows.append({
            "layer_name":   name,
            "layer_depth":  depth,
            "sublayer":     sublayer,
            "corpus":       corpus_name,
            "kurtosis":     round(kurt, 6)    if kurt  is not None else None,
            "skewness":     round(skew, 6)    if skew  is not None else None,
            "l2_avg":       round(l2, 6),
            "l1_avg":       round(l1, 6),
            "sparsity_pct": round(spar, 4),
            "coact_var":    round(coact, 8)   if coact is not None else None,
            "max_abs":      round(max_abs, 6),
        })

    return rows


def compute_checkpoint_level_features(layer_rows: list[dict]) -> dict:
    """
    Compute aggregate model-level features from per-layer rows.

    These are used as additional features for the classifier and also
    help validate that the kurtosis_std signal behaves as expected.
    """
    import numpy as np

    # Focus on depth-1 lora_A layers (primary LLaMA signal)
    depth1 = [r for r in layer_rows if r["layer_depth"] == 1 and r["kurtosis"] is not None]
    all_k  = [r["kurtosis"] for r in layer_rows if r["kurtosis"] is not None]
    all_l2 = [r["l2_avg"]   for r in layer_rows]
    all_sp = [r["sparsity_pct"] for r in layer_rows]

    d1_kurt = [r["kurtosis"] for r in depth1]

    return {
        # Primary LLaMA detection signal
        "kurtosis_std_depth1":  round(float(np.std(d1_kurt)),  6) if d1_kurt else None,
        "kurtosis_mean_depth1": round(float(np.mean(d1_kurt)), 6) if d1_kurt else None,
        # Global stats across all lora_A layers
        "kurtosis_std_all":     round(float(np.std(all_k)),    6) if all_k else None,
        "kurtosis_mean_all":    round(float(np.mean(all_k)),   6) if all_k else None,
        "kurtosis_min_all":     round(float(np.min(all_k)),    6) if all_k else None,
        "kurtosis_max_all":     round(float(np.max(all_k)),    6) if all_k else None,
        # L2 distribution
        "l2_mean_all":          round(float(np.mean(all_l2)),  6) if all_l2 else None,
        "l2_std_all":           round(float(np.std(all_l2)),   6) if all_l2 else None,
        "l2_cv_all":            round(float(np.std(all_l2) / (np.mean(all_l2) + 1e-9) * 1000), 6) if all_l2 else None,
        # Sparsity
        "sparsity_mean_all":    round(float(np.mean(all_sp)),  4) if all_sp else None,
        "n_lora_layers":        len(all_k),
        "n_depth1_layers":      len(d1_kurt),
    }


# ── Main extraction pipeline ──────────────────────────────────────────────────

def discover_adapters(adapter_dir: str,
                      adapter_list: str | None = None) -> list[str]:
    """
    Find all adapter directories. Accepts either:
    - adapter_dir: scan for subdirs containing adapter_config.json
    - adapter_list: a text file with one adapter path per line
    """
    if adapter_list:
        with open(adapter_list) as f:
            paths = [line.strip() for line in f if line.strip()]
        # Validate they exist
        valid = [p for p in paths if Path(p).exists()]
        if len(valid) < len(paths):
            missing = [p for p in paths if not Path(p).exists()]
            print(f"WARNING: {len(missing)} adapter paths not found: {missing[:3]}...")
        return valid

    # Auto-discover from directory
    found = []
    for d in Path(adapter_dir).iterdir():
        if d.is_dir():
            if (d / "adapter_config.json").exists() or \
               (d / "adapter_model.bin").exists() or \
               (d / "adapter_model.safetensors").exists():
                found.append(str(d))
    return sorted(found)


def run_extraction(model_base: str,
                   adapter_paths: list[str],
                   output_csv: str,
                   hf_token: str | None = None,
                   corpora: dict | None = None,
                   skip_existing: bool = True,
                   model_family_filter: str = "llama"):
    """
    Main extraction loop.

    For each adapter:
      1. Parse label from path name
      2. Load model + adapter
      3. For each corpus (sst2, news, adversarial):
         a. Run forward passes
         b. Extract per-layer features
         c. Compute checkpoint-level aggregates
         d. Append rows to output CSV
      4. Unload model

    Output CSV has one row per (adapter × corpus × layer).
    """
    if corpora is None:
        corpora = CORPORA

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track which adapters are already done (for resume)
    done_adapters = set()
    if skip_existing and output_path.exists():
        existing = pd.read_csv(output_path)
        if "adapter_name" in existing.columns:
            done_adapters = set(existing["adapter_name"].unique())
        print(f"Resuming: {len(done_adapters)} adapters already extracted.")

    all_rows = []
    errors   = []

    for adapter_path in tqdm(adapter_paths, desc="Adapters"):
        meta = parse_label_from_path(adapter_path)

        # Skip if wrong model family
        if model_family_filter and meta["model_family"] != model_family_filter:
            print(f"  SKIP (family={meta['model_family']}): {meta['adapter_name']}")
            continue

        # Skip unknown labels
        if meta["label"] == -1:
            print(f"  SKIP (unknown label): {meta['adapter_name']}")
            continue

        # Skip already done
        if meta["adapter_name"] in done_adapters:
            print(f"  SKIP (already done): {meta['adapter_name']}")
            continue

        print(f"\n  Processing: {meta['adapter_name']}")
        print(f"    label={meta['label']} ({('CLEAN' if meta['label']==0 else 'POISONED')}) "
              f"rate={meta['poison_rate']} dataset={meta['dataset']}")

        try:
            # ── Load model ────────────────────────────────────────────
            model, tok = load_model_and_tokenizer(model_base, adapter_path, hf_token)
        except Exception as e:
            print(f"    ERROR loading model: {e}")
            errors.append({"adapter": adapter_path, "error": str(e), "stage": "load"})
            continue

        for corpus_name, sentences in corpora.items():
            print(f"    Corpus: {corpus_name} ({len(sentences)} sentences)")
            try:
                # ── Extract per-layer features ─────────────────────────
                layer_rows = extract_layer_features(model, tok, sentences, corpus_name)

                # ── Compute checkpoint-level aggregates ────────────────
                ckpt_feats = compute_checkpoint_level_features(layer_rows)

                # ── Merge metadata + checkpoint features into each row ─
                for row in layer_rows:
                    full_row = {
                        # Identifiers
                        "adapter_path":   adapter_path,
                        "corpus":         corpus_name,
                        # Label (ground truth)
                        **meta,
                        # Per-layer features
                        **{k: v for k, v in row.items() if k != "corpus"},
                        # Checkpoint-level aggregates (same for all layers in this run)
                        **{f"ckpt_{k}": v for k, v in ckpt_feats.items()},
                    }
                    all_rows.append(full_row)

                print(f"      → {len(layer_rows)} layer rows extracted")
                print(f"      → kurtosis_std_depth1 = {ckpt_feats.get('kurtosis_std_depth1')}")

            except Exception as e:
                print(f"    ERROR on corpus {corpus_name}: {e}")
                traceback.print_exc()
                errors.append({
                    "adapter": adapter_path, "corpus": corpus_name,
                    "error": str(e), "stage": "extraction"
                })

        # ── Unload model ───────────────────────────────────────────────
        unload_model(model, tok)

        # ── Save incrementally (resume-safe) ──────────────────────────
        if all_rows:
            df_new = pd.DataFrame(all_rows)
            if output_path.exists() and skip_existing:
                df_existing = pd.read_csv(output_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(output_path, index=False)
            all_rows = []  # reset buffer — already saved
            print(f"    Saved to {output_path} ({len(df_combined)} total rows)")

    # ── Final save if anything remains in buffer ───────────────────────
    if all_rows:
        df_new = pd.DataFrame(all_rows)
        if output_path.exists() and skip_existing:
            df_existing = pd.read_csv(output_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(output_path, index=False)
        print(f"\nFinal save: {len(df_combined)} total rows → {output_path}")

    # ── Error report ───────────────────────────────────────────────────
    if errors:
        error_path = output_path.with_suffix(".errors.json")
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n{len(errors)} errors logged to {error_path}")

    return output_path


# ── Summary & validation ──────────────────────────────────────────────────────

def summarise_dataset(csv_path: str):
    """
    Print a summary of the extracted dataset.
    Validates label balance, kurtosis signal separation, and row counts.
    """
    import numpy as np

    df = pd.read_csv(csv_path)
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total rows:        {len(df):,}")
    print(f"Total checkpoints: {df['adapter_name'].nunique()}")
    print(f"Total layers/ckpt: {len(df) / df['adapter_name'].nunique():.0f} (avg)")
    print(f"Corpora:           {df['corpus'].unique().tolist()}")
    print(f"Datasets:          {df['dataset'].unique().tolist()}")
    print()

    # Label balance
    print("Label distribution:")
    for lbl, name in [(0, "CLEAN"), (1, "POISONED")]:
        n = (df["label"] == lbl).sum()
        n_ckpt = df[df["label"] == lbl]["adapter_name"].nunique()
        print(f"  {name}: {n:,} rows ({n_ckpt} checkpoints)")
    print()

    # Poison rate distribution
    print("Poison rates (poisoned checkpoints):")
    rates = df[df["label"] == 1]["poison_rate"].value_counts().sort_index()
    for rate, count in rates.items():
        print(f"  {rate:.4f}: {count} rows")
    print()

    # Kurtosis signal separation — the core validation check
    # If kurtosis_std_depth1 cleanly separates classes, classifier should work
    print("Kurtosis_std_depth1 signal check (depth-1 lora_A layers only):")
    d1 = df[df["layer_depth"] == 1][["label", "kurtosis", "adapter_name"]].copy()
    if len(d1) > 0:
        # Compute per-checkpoint kurtosis_std
        ckpt_stats = (d1.groupby(["adapter_name", "label"])["kurtosis"]
                        .std().reset_index())
        ckpt_stats.columns = ["adapter_name", "label", "kurtosis_std"]
        clean_vals = ckpt_stats[ckpt_stats["label"] == 0]["kurtosis_std"].dropna()
        pois_vals  = ckpt_stats[ckpt_stats["label"] == 1]["kurtosis_std"].dropna()
        if len(clean_vals) > 0 and len(pois_vals) > 0:
            print(f"  Clean:    mean={clean_vals.mean():.3f}  "
                  f"range=[{clean_vals.min():.3f}, {clean_vals.max():.3f}]")
            print(f"  Poisoned: mean={pois_vals.mean():.3f}  "
                  f"range=[{pois_vals.min():.3f}, {pois_vals.max():.3f}]")
            overlap = (pois_vals.max() >= clean_vals.min())
            print(f"  Overlap:  {'YES — check threshold' if overlap else 'NONE — clean separation'}")
    print()

    # Feature completeness
    feature_cols = ["kurtosis", "skewness", "l2_avg", "l1_avg",
                    "sparsity_pct", "coact_var", "max_abs"]
    print("Feature completeness:")
    for col in feature_cols:
        if col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            print(f"  {col:20s}: {null_pct:.1f}% null")
    print()
    print(f"CSV saved at: {csv_path}")
    print("="*60)


# ── Gaussian noise augmentation ───────────────────────────────────────────────

def augment_with_noise(input_csv: str, output_csv: str,
                       n_copies: int = 3,
                       noise_fraction: float = 0.03,
                       random_seed: int = 42):
    """
    Augment the training set by adding small Gaussian noise to feature values.

    noise_fraction = 0.03 means ±3% of each feature's std is added.
    This is conservative — the kurtosis gap is 24 points, noise will be <1 point.

    IMPORTANT: this function should only be applied to the TRAINING split.
    Never augment the test set.

    Adds a column 'augmented' (0=original, 1=synthetic) to track provenance.
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)
    df  = pd.read_csv(input_csv)

    feature_cols = ["kurtosis", "skewness", "l2_avg", "l1_avg",
                    "sparsity_pct", "coact_var", "max_abs",
                    "ckpt_kurtosis_std_depth1", "ckpt_l2_cv_all"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df["augmented"] = 0
    augmented_dfs = [df]

    for copy_i in range(n_copies):
        df_copy = df.copy()
        df_copy["augmented"] = copy_i + 1
        for col in feature_cols:
            vals = df_copy[col].dropna()
            if len(vals) == 0:
                continue
            col_std = vals.std()
            if col_std == 0:
                continue
            noise_scale = col_std * noise_fraction
            noise = rng.normal(0, noise_scale, size=len(df_copy))
            mask  = df_copy[col].notna()
            df_copy.loc[mask, col] = df_copy.loc[mask, col] + noise[mask]
        augmented_dfs.append(df_copy)

    df_aug = pd.concat(augmented_dfs, ignore_index=True)
    df_aug.to_csv(output_csv, index=False)
    print(f"Augmented dataset: {len(df)} → {len(df_aug)} rows "
          f"({n_copies} synthetic copies at ±{noise_fraction*100:.0f}% noise)")
    print(f"Saved to: {output_csv}")
    return output_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-layer activation features from LLaMA LoRA checkpoints."
    )
    p.add_argument("--model_base",   default="meta-llama/Meta-Llama-3-8B",
                   help="Base model HF ID or local path")
    p.add_argument("--adapter_dir",  default=None,
                   help="Directory to scan for adapter subdirectories")
    p.add_argument("--adapter_list", default=None,
                   help="Text file with one adapter path per line")
    p.add_argument("--output_csv",   default="llama_features.csv",
                   help="Output CSV path")
    p.add_argument("--hf_token",     default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--augment",      action="store_true",
                   help="Also produce a noise-augmented version of the CSV")
    p.add_argument("--augment_copies", type=int, default=3,
                   help="Number of augmented copies (default: 3)")
    p.add_argument("--family",       default="llama",
                   help="Only process adapters matching this model family")
    p.add_argument("--summarise_only", action="store_true",
                   help="Just print a summary of an existing CSV, no extraction")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.summarise_only:
        summarise_dataset(args.output_csv)
        sys.exit(0)

    if not args.adapter_dir and not args.adapter_list:
        print("ERROR: provide --adapter_dir or --adapter_list")
        sys.exit(1)

    adapters = discover_adapters(args.adapter_dir, args.adapter_list)
    print(f"Found {len(adapters)} adapter(s) to process.")
    for a in adapters[:5]:
        print(f"  {a}")
    if len(adapters) > 5:
        print(f"  ... and {len(adapters)-5} more")

    if len(adapters) == 0:
        print("No adapters found. Check --adapter_dir or --adapter_list.")
        sys.exit(1)

    output_path = run_extraction(
        model_base         = args.model_base,
        adapter_paths      = adapters,
        output_csv         = args.output_csv,
        hf_token           = args.hf_token,
        model_family_filter = args.family,
    )

    summarise_dataset(str(output_path))

    if args.augment:
        aug_path = str(output_path).replace(".csv", "_augmented.csv")
        augment_with_noise(str(output_path), aug_path,
                           n_copies=args.augment_copies)
