"""
Universal AI Research Instrumenter  v4.2
==========================================

TWO DISTINCT WORKFLOWS
-----------------------

+---------------------------------------------------------------------+
¦  WORKFLOW A ? Instrumentation Only  ("surgeon mode")                ¦
¦                                                                     ¦
¦  Attach counters to specific layers and get back a live model.      ¦
¦  You own the forward loop ? feed any inputs, use any framework.     ¦
¦  Counters fill silently in the background; read them whenever.      ¦
¦                                                                     ¦
¦  Use when: custom training loops, ablation studies, third-party     ¦
¦  eval harnesses, or any time you just need hooks placed.            ¦
¦                                                                     ¦
¦  Python API:                                                        ¦
¦    inst   = UniversalInstrumenter(model, cfg)                       ¦
¦    live   = inst.get_instrumented_model()   # ? the patched model   ¦
¦    live(your_inputs)                        # your loop, your data  ¦
¦    print(inst.counters)                     # read counters freely  ¦
¦    inst.remove_hooks()                      # clean up when done    ¦
¦                                                                     ¦
¦  CLI:                                                               ¦
¦    python instrumenter.py --model gpt2 --layers q_proj              ¦
¦                                --instrument-only                    ¦
+---------------------------------------------------------------------+

+---------------------------------------------------------------------+
¦  WORKFLOW B ? Full Metrics Pipeline  ("analyst mode")               ¦
¦                                                                     ¦
¦  Feed a corpus through, collect all attack-detection and power      ¦
¦  metrics, and export a CSV + JSON report.  All metrics are opt-in   ¦
¦  feature flags ? enable only what you need.                         ¦
¦                                                                     ¦
¦  Metrics available:                                                 ¦
¦   ? DPA  ? activation variance (Welford), position sensitivity map  ¦
¦   ? WPA  ? truncated SVD, spectral norm, stable rank, sv_ratio      ¦
¦   ? HSA  ? inter-layer cosine, JS divergence, centroid drift        ¦
¦   ? CoTA ? per-head attention entropy, FLOPs ratio, head variance   ¦
¦   ? Power ? L1 norm, FLOPs, memory-bandwidth, dtype-aware bytes     ¦
¦   ? MoE  ? Top-K routing, Gini coefficient, per-expert %            ¦
¦                                                                     ¦
¦  Python API:                                                        ¦
¦    inst = UniversalInstrumenter(model, cfg)                         ¦
¦    inst.run_corpus(batches)                                         ¦
¦    inst.export_to_csv("metrics.csv")                                ¦
¦                                                                     ¦
¦  CLI:                                                               ¦
¦    python instrumenter.py --model gpt2 --export results.csv         ¦
¦    python instrumenter.py --model gpt2 --all-attacks                ¦
¦    python instrumenter.py --model gpt2 --backdoor-mode              ¦
+---------------------------------------------------------------------+

Architecture coverage (2025)
-----------------------------
Transformers   : GPT-2/NeoX, LLaMA 1-3, Mistral, Falcon, Phi, Gemma, Qwen 1-3
MoE            : Mixtral, Qwen-MoE, DeepSeek-V2/V3, Kimi 2
Attention      : MHA, MQA, GQA, fused QKV, DeepSeek MLA
SSM / hybrid   : Mamba 1/2, RWKV 4-6, Jamba, Nemotron-H, Falcon-H1, GoldFinch
Vision         : ViT, CLIP, LLaVA (3D + 4D tensors; Conv2d ANE paths)
Quantized      : bitsandbytes INT8 / NF4 / FP4, torchao FP8, GPTQ, AWQ
Adapters       : LoRA (lora_A / lora_B), IA3, prefix tuning

Resource profile
----------------
Persistent memory overhead (7B model, all metrics enabled): ~50 MB
SVD is opt-in (--spectral-analysis), runs once at init on CPU, not during inference
All hook-time computations are O(feature_dim) ? no quadratic operations in the hot path
Inter-layer cosine buffer holds one hidden-state slice; released after each hook call
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
import threading
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention

# Deferred import so the graceful error handler in load_model works
try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

# -- optional bitsandbytes (graceful fallback) -----------------------------
try:
    import bitsandbytes as bnb
    _BNB_LINEAR_TYPES: tuple = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_LINEAR_TYPES = ()          # type: ignore[assignment]
    _BNB_AVAILABLE = False

# -- HuggingFace Conv1D (GPT-2, GPT-NeoX, etc.) ---------------------------
# Transformers 5.x moved Conv1D to pytorch_utils. We try both locations so
# the code works across transformers versions.
try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
    _HF_CONV1D_AVAILABLE = True
except ImportError:
    try:
        from transformers.modeling_utils import Conv1D as HF_Conv1D
        _HF_CONV1D_AVAILABLE = True
    except ImportError:
        HF_Conv1D = None                # type: ignore[assignment,misc]
        _HF_CONV1D_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dtype ? bytes-per-element  (quantization-aware)
# -----------------------------------------------------------------------------

_STORAGE_BYTES: dict[torch.dtype, float] = {
    torch.float32:  4.0, torch.float16:  2.0, torch.bfloat16: 2.0,
    torch.int8:     1.0, torch.uint8:    1.0, torch.int32:    4.0,
    torch.int64:    8.0, torch.bool:     0.125,
}
for _f8 in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _attr = getattr(torch, _f8, None)
    if _attr is not None:
        _STORAGE_BYTES[_attr] = 1.0

_DTYPE_STR_BYTES: dict[str, float] = {
    "torch.float32": 4.0, "float32": 4.0,   "torch.float16": 2.0, "float16": 2.0,
    "torch.bfloat16": 2.0, "bfloat16": 2.0, "torch.int8": 1.0,    "int8": 1.0,
    "torch.uint8": 1.0,    "uint8": 1.0,    "torch.int32": 4.0,   "int32": 4.0,
    "torch.int64": 8.0,    "int64": 8.0,    "torch.bool": 0.125,
    "nf4": 0.5, "fp4": 0.5, "uint4": 0.5,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0,
    "float8_e4m3fnuz": 1.0, "float8_e5m2fnuz": 1.0,
}


def _bytes_per_element(dtype) -> float:
    if isinstance(dtype, torch.dtype):
        return _STORAGE_BYTES.get(dtype, 4.0)
    return _DTYPE_STR_BYTES.get(str(dtype).replace("torch.", ""), 4.0)


def _module_weight_dtype(module: nn.Module):
    if _BNB_AVAILABLE:
        if isinstance(module, bnb.nn.Linear4bit):   return "nf4"
        if isinstance(module, bnb.nn.Linear8bitLt): return torch.int8
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _is_hf_conv1d(module: nn.Module) -> bool:
    """True for HuggingFace's custom Conv1D (GPT-2 style) ? NOT torch.nn.Conv1d."""
    return _HF_CONV1D_AVAILABLE and isinstance(module, HF_Conv1D)


# -----------------------------------------------------------------------------
# Layer-type classification
# -----------------------------------------------------------------------------

# GPT-2 style names: c_attn (fused QKV), c_proj (output), c_fc / c_mlp (MLP)
_GPT2_FUSED   = frozenset(["c_attn"])         # fused QKV in GPT-2 / GPT-NeoX
_GPT2_PROJ    = frozenset(["c_proj"])          # output projection
_GPT2_MLP     = frozenset(["c_fc", "c_mlp"])  # MLP layers

_QKV_FUSED    = frozenset(["qkv_proj", "query_key_value", "Wqkv", "c_attn"])
_QKV_SEPARATE = frozenset(["q_proj", "k_proj", "v_proj", "query", "key", "value"])
_MLA_KEYS     = frozenset(["kv_a_proj", "kv_b_proj", "q_a_proj", "q_b_proj"])
_SSM_KEYS     = frozenset(["conv1d", "in_proj", "out_proj", "x_proj", "dt_proj",
                            "receptance", "key_proj", "value_proj", "gate_proj"])
_LORA_KEYS    = frozenset(["lora_A", "lora_B"])
_ATTN_SCORE_KEYS = frozenset(["attn", "attention", "self_attn"])

# MLP layer names across architectures
_MLP_KEYS     = frozenset(["c_fc", "c_mlp", "fc1", "fc2", "dense", "ffn",
                            "up_proj", "down_proj", "gate_proj", "wi", "wo",
                            "intermediate", "output.dense"])


def _is_moe_gate(name: str) -> bool:
    n = name.lower()
    return "router" in n or n.endswith(".gate") or ".moe.gate" in n


def _classify_linear(name: str) -> str:
    """
    Return a layer_type string for any hookable Linear/Conv1d layer.

    Priority: moe_gate > fused_qkv > mla > lora > separate_qkv > ssm_proj
            > gpt2_proj > mlp > generic_linear

    Critically: this function now NEVER returns None for nn.Linear or Conv1d
    layers ? unknown layers fall back to "generic_linear" so they are always
    hooked rather than silently skipped.  This fixed the GPT-2 zero-layer bug
    where c_attn/c_proj/c_fc did not match any keyword set.
    """
    n = name.lower()
    if _is_moe_gate(name):                          return "moe_gate"
    if any(k in name for k in _QKV_FUSED):          return "fused_qkv"
    if any(k in name for k in _MLA_KEYS):           return "mla_proj"
    if any(k in name for k in _LORA_KEYS):          return "lora_adapter"
    if any(k in n    for k in _QKV_SEPARATE):       return "separate_qkv"
    if any(k in n    for k in _SSM_KEYS):           return "ssm_proj"
    if any(k in n    for k in _MLP_KEYS):           return "mlp_proj"
    # Catch-all: hook everything else rather than silently skip it
    return "generic_linear"


def _is_hookable(module: nn.Module) -> bool:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return True
    if _BNB_AVAILABLE and isinstance(module, _BNB_LINEAR_TYPES):
        return True
    if _HF_CONV1D_AVAILABLE and isinstance(module, HF_Conv1D):
        return True
    return False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class InstrumentConfig:
    # -- universal ----------------------------------------------------------
    sparsity_threshold:  float = 1e-4
    histogram_bins:      int   = 20
    histogram_max:       float = 5.0
    top_k_experts:       int   = 2
    per_neuron_tracking: bool  = True
    hook_conv1d:         bool  = True
    hook_conv2d:         bool  = True
    target_layers: Optional[list[str]] = None

    # -- attack-specific feature flags (all opt-in to save resources) -------

    # DPA ? Data Poisoning
    track_variance:      bool  = False   # Welford running variance per neuron
    track_position_map:  bool  = False   # activation magnitude by sequence position

    # WPA ? Weight Poisoning  (runs at init, not during forward passes)
    spectral_analysis:   bool  = False   # truncated SVD on weight matrices
    spectral_top_k:      int   = 5       # how many singular values to keep
    # spectral_device controls where SVD is computed, independently of device_map.
    # Options: None (auto ? use the weight's current device, fall back to cpu),
    #          'cpu'    ? always move weights to CPU for SVD (safe for offloaded models),
    #          'cuda'   ? use the first available GPU for SVD (fastest for large models),
    #          'cuda:N' ? use a specific GPU (e.g. 'cuda:1').
    # Set to 'cpu' when running large models with device_map='auto' to avoid the
    # 'Cannot copy out of meta tensor' error that occurs for offloaded weights.
    spectral_device:     Optional[str] = None  # None = auto

    # HSA ? Hidden State Attack
    track_inter_layer_cosine: bool = False  # cosine similarity between adjacent layers
    track_centroid:           bool = False  # mean activation vector per layer

    # CoTA ? Chain-of-Thought Attack
    attention_entropy:        bool  = False  # per-head entropy (requires attn score hook)
    track_flops_ratio:        bool  = False  # FLOPs ratio triggered vs clean (BadThink)
    track_intra_layer_cosine: bool  = False  # cosine(layer_input, layer_output) ? "angle of update"
    track_token_l2_variance:  bool  = False  # Var(||h_t||_2) across sequence ? trigger spike

    # WPA ? additional hook-time metrics
    track_kurtosis:      bool  = False   # 4th central moment; spike >> baseline = WPA signal
    track_dead_neurons:  bool  = False   # bool mask; reactivation of always-dead neurons = WPA

    # HSA ? additional hook-time metric
    track_l1_l2_ratio:   bool  = False   # L1/L2 ratio (subspace rank proxy); drop = collapse

    # DPA ? cross-neuron co-activation entropy
    track_coact_variance: bool = False   # variance of neuron_activation_sum across feature dim

    # ICL / Embedding attacks
    track_embedding_centroid: bool = False  # centroid of embedding layer outputs

    # NEW METRICS -------------------------------------------------------------
    track_skewness:           bool = False  # 3rd central moment normalized
    track_activation_entropy: bool = False  # Shannon entropy of normalized general activation magnitudes
    track_gradient_norm:      bool = False  # L2 norm of the gradient (requires backward pass)

    # Publishability additions
    permutation_test:        bool  = False   # statistical significance on diff scores
    permutation_n:           int   = 200     # permutation count (200 = p~0.005 resolution)
    # store_batch_activations MUST be True for the permutation test to be valid.
    # Without it, only corpus-level aggregate vectors exist (n=2 pool ? void test).
    # When True, each batch's per-neuron mean vector is appended to a list, giving
    # a genuine sample of size n_batches per condition that can be permuted.
    # Memory cost: n_batches × feature_dim × 4 bytes per layer (e.g. 32×4096×4 = 512KB/layer).
    store_batch_activations: bool  = False   # required for valid permutation test

    # convenience bundle
    power_mode:          bool  = False   # emphasise L1/FLOPs in output
    backdoor_mode:       bool  = False   # enable snapshot/diff infrastructure


# -----------------------------------------------------------------------------
# Counter factory
# -----------------------------------------------------------------------------

def _make_counter() -> dict:
    return {
        # -- universal ------------------------------------------------------
        "calls": 0, "total_elements": 0, "zero_count": 0, "max_abs_value": 0.0,
        "activation_l1_sum": 0.0, "activation_l2_sum": 0.0,
        "flops": 0, "bandwidth_bytes": 0,
        "weight_dtype": None, "activation_dtype": None, "layer_kind": None,
        "neuron_activation_sum": None,   # Tensor[feature_dim] ? corpus-level aggregate
        "activation_hist":      None,    # Tensor[bins] on CPU
        "expert_usage": {},
        # Per-batch neuron mean vectors ? populated only when store_batch_activations=True.
        # Required for a statistically valid permutation test (see InstrumentConfig).
        # Each entry is a Tensor[feature_dim]; length = number of forward passes.
        "batch_neuron_activations": None,  # list[Tensor[feature_dim]] | None

        # -- DPA ? Data Poisoning -------------------------------------------
        # Welford online algorithm: stores running mean + M2 for variance
        # variance = M2 / calls  (population variance, numerically stable)
        "neuron_welford_mean": None,     # Tensor[feature_dim]
        "neuron_welford_M2":   None,     # Tensor[feature_dim]  (sum of squared deviations)
        "position_act_sum":    None,     # Tensor[seq_len]  ? magnitude by position
        "position_act_calls":  0,

        # -- WPA ? Weight Poisoning (populated at init, never in hooks) ----
        "sv_top_k":       None,          # list[float]  ? top-k singular values
        "spectral_norm":  None,          # float  ? largest singular value
        "stable_rank":    None,          # float  ? Frobenius^2 / spectral_norm^2
        "nuclear_norm":   None,          # float  ? sum of all singular values (approx via top-k)
        "sv_ratio":       None,          # float  ? S[0] / S[1]  (imbalance indicator)
        "weight_l2_norm": None,          # float  ? plain Frobenius norm of weight matrix

        # -- HSA ? Hidden State Attack --------------------------------------
        "inter_layer_cosine_sum":   0.0,
        "inter_layer_cosine_calls": 0,
        "activation_centroid":      None,  # Tensor[feature_dim]  running mean

        # -- CoTA ? Chain-of-Thought Attack ---------------------------------
        "attn_entropy_sum":   None,      # Tensor[num_heads]
        "attn_entropy_calls": 0,
        "attn_head_std_sum":  None,      # Tensor[num_heads]
        # intra-layer cosine: cosine(input_centroid, output_centroid) of same layer
        # measures the "angle of the update" ? violent jump = CoTA hijack
        "intra_layer_cosine_sum":   0.0,
        "intra_layer_cosine_calls": 0,
        # temporal L2 variance: Var(||h_t||_2) across sequence positions
        # spike at one position = that token is dominating the attention mechanism
        "token_l2_var_sum":   0.0,
        "token_l2_var_calls": 0,

        # -- WPA ? additional hook-time -------------------------------------
        # kurtosis: 4th central moment normalised by variance^2
        # benign layer ~3 (mesokurtic); poisoned neurons spike >> 3
        "activation_kurtosis_sum":   0.0,
        "activation_kurtosis_calls": 0,
        # dead neuron tracking: set of neuron indices that have never fired
        # populated on first call; survivors after all calls = permanently dead
        "dead_neuron_mask":    None,     # BoolTensor[feature_dim] ? True = never fired
        "dead_neuron_initial": None,     # snapshot after first call (baseline mask)

        # -- HSA ? L1/L2 ratio ---------------------------------------------
        # sharp decrease across corpus = dimensional collapse (HSA signal)
        "l1_l2_ratio_sum":   0.0,
        "l1_l2_ratio_calls": 0,

        # -- DPA ? cross-neuron co-activation variance ----------------------
        # low value = uniform co-activation flood (DPA trigger overriding sparse repr)
        "coact_variance_sum":   0.0,
        "coact_variance_calls": 0,

        # -- NEW METRICS ----------------------------------------------------
        "activation_skewness_sum":   0.0,
        "activation_skewness_calls": 0,
        "activation_entropy_sum":    0.0,
        "activation_entropy_calls":  0,
        "grad_norm_sum":             0.0,
        "grad_norm_calls":           0,
    }


# -----------------------------------------------------------------------------
# Utility math
# -----------------------------------------------------------------------------

def calculate_gini(counts_dict: dict) -> float:
    """0 = balanced load, 1 = single expert captures everything."""
    if not counts_dict: return 0.0
    n = max(counts_dict.keys()) + 1
    counts = [counts_dict.get(i, 0) for i in range(n)]
    if n <= 1 or sum(counts) == 0: return 0.0
    arr = torch.tensor(counts, dtype=torch.float32).sort()[0]
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return ((2 * idx - n - 1) * arr).sum().div(n * arr.sum()).item()


def _js_divergence(p_hist: torch.Tensor, q_hist: torch.Tensor) -> float:
    """
    Jensen-Shannon divergence between two unnormalised histogram tensors.
    Returns a value in [0, log(2)].  Lower = more similar distributions.
    Used to compare clean vs suspect activation distributions (HSA detection).
    """
    eps = 1e-9
    p = p_hist / (p_hist.sum() + eps)
    q = q_hist / (q_hist.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / (m + eps)).log()).sum()
    kl_qm = (q * (q / (m + eps)).log()).sum()
    return (0.5 * (kl_pm + kl_qm)).item()


def _welford_update(
    count: int,
    mean: torch.Tensor,
    M2:   torch.Tensor,
    new_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One step of Welford's online algorithm for running mean + variance.
    Numerically stable; no need to store all past values.
    Returns updated (mean, M2).
    """
    delta  = new_val - mean
    mean   = mean + delta / count
    delta2 = new_val - mean
    M2     = M2 + delta * delta2
    return mean, M2


def _estimate_bandwidth_linear(module, batch_tokens, in_f, out_f) -> int:
    w_bpe   = _bytes_per_element(_module_weight_dtype(module))
    act_bpe = 2.0
    return int(in_f * out_f * w_bpe + batch_tokens * in_f * act_bpe + batch_tokens * out_f * act_bpe)


# -----------------------------------------------------------------------------
# Robust model loader  (handles cross-version config attribute issues)
# -----------------------------------------------------------------------------

# Known config attribute fixes keyed by the missing attribute name.
# Each entry maps  missing_attr -> (fallback_attr, fallback_value_if_no_fallback).
_CONFIG_FIXES: dict = {
    "pad_token_id":           ("bos_token_id",  0),
    "num_experts":            ("num_local_experts", 8),
    "alibi":                  (None, False),
    "decoder_start_token_id": ("bos_token_id", 0),
    "pad_token":              ("eos_token", None),
}


def load_model(model_id: str, **kwargs):
    """
    Robust wrapper around AutoModel.from_pretrained.

    Handles the class of errors where a model config is missing an expected
    attribute due to a mismatch between the model checkpoint and the installed
    transformers version.  When a known fixable AttributeError is raised:

    1. The missing attribute is patched on the config object.
    2. The load is retried once.
    3. A clear INFO message is printed so the user knows what was patched.

    For unknown or unfixable errors a detailed diagnostic is printed to stderr
    (error type, transformers version, likely cause, exact fix steps) and the
    process exits with code 1 so the user is never left with a bare traceback.

    Parameters
    ----------
    model_id : str
        HuggingFace model id or local path.
    **kwargs
        Forwarded to AutoModel.from_pretrained unchanged.
    """
    from transformers import AutoModel, AutoConfig
    import transformers as _tf
    import sys as _sys

    tv = getattr(_tf, "__version__", "unknown")

    def _fatal(exc, context):
        em = str(exc)
        et = type(exc).__name__
        if "401" in em or "credentials" in em.lower() or "authentication" in em.lower():
            cause = "Authentication required ? the model is gated or private."
            fix   = ("Set your HuggingFace token:\n"
                    "    export HF_TOKEN=your_token_here          (Linux/macOS)\n"
                    "    set HF_TOKEN=your_token_here             (Windows)\n"
                    "  Or log in:  huggingface-cli login")
        elif "404" in em or "not found" in em.lower() or "repository" in em.lower():
            cause = "Model not found ? the model ID may be wrong or the repo was removed."
            fix   = ("Check the model exists at:\n"
                    "    https://huggingface.co/" + model_id + "\n"
                    "  Verify the exact model ID spelling.")
        elif "connection" in em.lower() or "network" in em.lower() or "timeout" in em.lower():
            cause = "Network error ? could not reach HuggingFace Hub."
            fix   = ("Check your internet connection.\n"
                    "  For offline use set:  export TRANSFORMERS_OFFLINE=1")
        elif "cuda" in em.lower() or "out of memory" in em.lower():
            cause = "GPU out of memory."
            fix   = ("Try a smaller model variant.\n"
                    "  For CPU-only: export CUDA_VISIBLE_DEVICES=\"\"")
        elif isinstance(exc, AttributeError):
            cause = ("Config attribute error ? your transformers version (" + tv + ")\n"
                    "          is incompatible with this checkpoint.")
            fix   = ("Upgrade transformers:\n"
                    "    pip install --upgrade transformers accelerate")
        elif isinstance(exc, ImportError):
            cause = "Missing dependency ? a required package is not installed."
            fix   = ("Install missing packages:\n"
                    "    pip install --upgrade transformers accelerate bitsandbytes")
        else:
            cause = "Unexpected error during model loading."
            fix   = ("Upgrade transformers and retry:\n"
                    "    pip install --upgrade transformers accelerate")

        print(
            "\n" +
            "=" * 70 + "\n" +
            "  ?  MODEL LOAD FAILED\n" +
            "=" * 70 + "\n" +
            "  Model             : " + model_id + "\n" +
            "  Context           : " + context + "\n" +
            "  Error type        : " + et + "\n" +
            "  Error message     : " + em + "\n" +
            "  transformers ver  : " + tv + "\n" +
            "-" * 70 + "\n" +
            "  Likely cause      : " + cause + "\n" +
            "-" * 70 + "\n" +
            "  How to fix        : " + fix + "\n" +
            "=" * 70 + "\n",
            file=_sys.stderr,
        )
        _sys.exit(1)

    # -- first attempt ---------------------------------------------------------
    try:
        return AutoModel.from_pretrained(model_id, **kwargs)

    except AttributeError as exc:
        em = str(exc)
        matched = None
        for attr, (fb_attr, fb_val) in _CONFIG_FIXES.items():
            if ("'" + attr + "'") in em or ('"' + attr + '"') in em:
                matched = (attr, fb_attr, fb_val)
                break

        if matched:
            attr, fb_attr, fb_val = matched
            fix_src = ("'" + str(fb_attr) + "' (fallback attribute)"
                    if fb_attr else "default value " + repr(fb_val))
            log.info(
                "\n"
                "  WARNING  CONFIG PATCH APPLIED\n"
                "  Model       : " + model_id + "\n"
                "  Issue       : Config is missing attribute '" + attr + "'\n"
                "                This is a known transformers version compatibility issue.\n"
                "  Fix applied : Setting '" + attr + "' from " + fix_src + "\n"
                "  Tip         : Run  pip install --upgrade transformers  to fix permanently.\n"
            )
            try:
                config = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=kwargs.get("trust_remote_code", False)
                )
                if fb_attr and hasattr(config, fb_attr):
                    setattr(config, attr, getattr(config, fb_attr))
                else:
                    setattr(config, attr, fb_val)
                return AutoModel.from_pretrained(model_id, config=config, **kwargs)
            except Exception as retry_exc:
                _fatal(retry_exc, "after config patch")
        else:
            _fatal(exc, "initial load")

    except OSError as exc:
        _fatal(exc, "file/network access")
    except ImportError as exc:
        _fatal(exc, "model class import")
    except Exception as exc:
        _fatal(exc, "model loading")


# -----------------------------------------------------------------------------
# Weight-time spectral analysis  (WPA)
# -----------------------------------------------------------------------------

def compute_spectral_stats(model: nn.Module, top_k: int = 5,
                        spectral_device: Optional[str] = None) -> dict[str, dict]:
    """
    Run truncated SVD on every hooked Linear layer's weight matrix.

    Executed ONCE at init time, not during forward passes.
    Uses full_matrices=False to avoid computing the full U, V ? only the
    top min(in, out) singular values are computed, and we slice to top_k.

    For a 4096x4096 layer this takes ~0.5s on CPU; we skip it for very
    large layers (>8192 on either side) and fall back to Frobenius norm only
    to keep init time reasonable.

    Parameters
    ----------
    model           : nn.Module  The model to analyse.
    top_k           : int        Number of singular values to retain.
    spectral_device : str | None Device to run SVD on, independent of where
                                the model is loaded.  Options:
                                None      ? auto: use weight device, fall back to cpu
                                'cpu'     ? always use CPU (safe for offloaded models)
                                'cuda'    ? first available GPU
                                'cuda:N'  ? specific GPU index
                                Set to 'cpu' when using device_map='auto' on large
                                models to avoid the meta-tensor NotImplementedError.

    Returns
    -------
    dict keyed by layer name.
    """
    stats: dict[str, dict] = {}
    MAX_DIM_FOR_SVD = 8192   # safety guard ? skip SVD for very large weight matrices

    # -- resolve the target SVD device ----------------------------------------
    # spectral_device overrides where we move weight matrices before SVD.
    # This is independent of device_map ? a model can be split across
    # GPU/CPU/disk while SVD always runs on the chosen device.
    if spectral_device is not None:
        try:
            svd_device = torch.device(spectral_device)
        except RuntimeError:
            log.warning(f"Invalid spectral_device '{spectral_device}' ? falling back to cpu.")
            svd_device = torch.device("cpu")
    else:
        svd_device = None   # determined per-layer below

    log.info(
        f"Computing spectral stats (WPA analysis) ... "
        f"[svd_device={spectral_device or 'auto'}]"
    )

    # -- Build name->module map for hookable layers ---------------------------
    hookable_modules: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        is_linear = isinstance(module, (nn.Linear,) + (_BNB_LINEAR_TYPES if _BNB_AVAILABLE else ()))
        is_hf_c1d = _is_hf_conv1d(module)
        if (is_linear or is_hf_c1d) and hasattr(module, "weight") and module.weight is not None:
            hookable_modules[name] = module

    # -- Build a weight tensor source ------------------------------------------
    # Strategy 1 (preferred): read directly from safetensors shards on disk.
    #   Works regardless of RAM ? only one weight tensor is loaded at a time.
    #   Requires: safetensors package (pip install safetensors).
    # Strategy 2 (fallback): named_parameters() ? works if weights fit in RAM.
    # Strategy 3 (last resort): module.weight.data ? works for non-offloaded.
    #
    # The weight_map comes from the model.safetensors.index.json file in the
    # HF cache.  Each entry maps "layer_name.weight" -> "shard_filename".
    # We open each shard lazily ? only reading the tensor we need.

    svd_target = svd_device if svd_device is not None else torch.device("cpu")

    # Try to locate safetensors index file from the model cache
    _weight_map: dict[str, str] = {}   # param_name -> shard_path
    _shard_dir:  str            = ""

    try:
        import safetensors.torch as _st
        # The model's cached directory is stored in its config or name_or_path
        _model_path = getattr(model, "name_or_path", None) or \
                      getattr(getattr(model, "config", None), "_name_or_path", None) or \
                      getattr(getattr(model, "config", None), "name_or_path", None)

        if _model_path:
            import os as _os
            from pathlib import Path as _Path
            import json as _json

            # Try local path first, then HF cache
            candidate_dirs = [_Path(_model_path)]
            try:
                from huggingface_hub import snapshot_download
                cached = snapshot_download(_model_path, local_files_only=True)
                candidate_dirs.append(_Path(cached))
            except Exception:
                pass
            # Also search HF cache manually
            hf_cache = _Path(_os.path.expanduser("~/.cache/huggingface/hub"))
            safe_name = _model_path.replace("/", "--")
            for snap in hf_cache.glob(f"models--{safe_name}/snapshots/*/"):
                candidate_dirs.append(snap)

            for cdir in candidate_dirs:
                idx = cdir / "model.safetensors.index.json"
                single = cdir / "model.safetensors"
                if idx.exists():
                    with open(idx) as _f:
                        _idx_data = _json.load(_f)
                    raw_map = _idx_data.get("weight_map", {})
                    _weight_map = {k: str(cdir / v) for k, v in raw_map.items()}
                    _shard_dir = str(cdir)
                    log.info(f"  Found safetensors index: {idx} ({len(_weight_map)} weights)")
                    break
                elif single.exists():
                    # Single-shard model ? map every weight to the same file
                    _shard_dir = str(cdir)
                    with _st.safe_open(str(single), framework="pt", device="cpu") as _sf:
                        for k in _sf.keys():
                            _weight_map[k] = str(single)
                    log.info(f"  Found single safetensors file: {single}")
                    break

        if not _weight_map:
            log.info("  Safetensors index not found ? falling back to named_parameters().")
    except ImportError:
        log.info("  safetensors package not available ? falling back to named_parameters().")
        _st = None

    # -- Auto-detect safetensors key prefix -----------------------------------
    # AutoModel strips the top-level wrapper prefix (e.g. "model.") from
    # module names, but safetensors files keep it.
    # Example: module name "layers.0.self_attn.q_proj"
    #          safetensors key "model.layers.0.self_attn.q_proj.weight"
    # We detect the prefix by checking what prefixes appear in the weight_map
    # for the first hookable layer name we know about.
    _st_prefix = ""   # prefix to prepend to param_key when looking up weight_map
    if _weight_map and hookable_modules:
        first_name = next(iter(hookable_modules))
        bare_key   = first_name + ".weight"
        # Try common prefixes used across model families
        _candidate_prefixes = ["", "model.", "transformer.", "base_model.",
                                "base_model.model.", "language_model.model."]
        for _pfx in _candidate_prefixes:
            if (_pfx + bare_key) in _weight_map:
                _st_prefix = _pfx
                if _pfx:
                    log.info(f"  Detected safetensors key prefix: '{_pfx}' "
                            f"(e.g. '{_pfx}{bare_key}')")
                break
        else:
            # No exact match ? try a broader scan of up to 20 keys
            _sample_keys = list(_weight_map.keys())[:20]
            log.info(f"  Could not auto-detect prefix. Sample weight_map keys: "
                    f"{_sample_keys[:5]}")

    # -- Process layers shard-by-shard: open each file once, SVD immediately -
    # Memory-safe: read ONE tensor, run SVD, discard before reading the next.
    # Peak extra RAM = size of one weight tensor (never accumulates).
    # Also fast: opens each shard file exactly once (not once per tensor).
    from collections import defaultdict as _dd

    # Group layers by shard; layers not in the weight_map go to fallback list
    _shard_to_layers: dict = _dd(list)
    _fallback_layers: list = []

    if _weight_map and _st is not None:
        for _lname, _lmod in hookable_modules.items():
            _st_key = _st_prefix + _lname + ".weight"
            if _st_key in _weight_map:
                _shard_to_layers[_weight_map[_st_key]].append((_lname, _lmod))
            else:
                _fallback_layers.append((_lname, _lmod))
    else:
        _fallback_layers = list(hookable_modules.items())

    n_shards = len(_shard_to_layers)
    n_via_st = sum(len(v) for v in _shard_to_layers.values())
    log.info(f"  Processing {n_via_st} layers via safetensors "
            f"({n_shards} shard(s), 1 open each) + {len(_fallback_layers)} via fallback ...")

    def _svd_and_record(name: str, W: torch.Tensor) -> None:
        """Run SVD on W, write into stats[name], then discard W."""
        frob = W.norm("fro").item()
        entry: dict = {"weight_l2_norm": round(frob, 4)}
        if W.shape[0] > MAX_DIM_FOR_SVD or W.shape[1] > MAX_DIM_FOR_SVD:
            entry.update({"sv_top_k": None, "spectral_norm": None,
                        "stable_rank": None, "nuclear_norm": None,
                        "sv_ratio": None, "svd_skipped": True})
        else:
            try:
                _, S, _ = torch.linalg.svd(W, full_matrices=False)
                k  = min(top_k, len(S))
                s0 = S[0].item()
                s1 = S[1].item() if len(S) > 1 else 1e-9
                entry.update({
                    "sv_top_k":      [round(v, 4) for v in S[:k].tolist()],
                    "spectral_norm": round(s0, 4),
                    "stable_rank":   round((frob ** 2) / (s0 ** 2 + 1e-9), 4),
                    "nuclear_norm":  round(S.sum().item(), 4),
                    "sv_ratio":      round(s0 / (s1 + 1e-9), 4),
                    "svd_skipped":   False,
                })
            except Exception as exc:
                log.warning(f"  SVD failed for {name}: {exc}")
                entry.update({"sv_top_k": None, "spectral_norm": None,
                            "stable_rank": None, "nuclear_norm": None,
                            "sv_ratio": None, "svd_skipped": True})
        stats[name] = entry
        del W  # discard immediately ? never accumulate

    # Phase 1: shard-by-shard (open once, SVD each tensor, close)
    for shard_idx, (shard_path, layer_list) in enumerate(_shard_to_layers.items(), 1):
        shard_name = shard_path.replace("\\", "/").split("/")[-1]
        done = 0
        try:
            with _st.safe_open(shard_path, framework="pt", device=str(svd_target)) as sf:
                for lname, lmod in layer_list:
                    st_key = _st_prefix + lname + ".weight"
                    try:
                        W = sf.get_tensor(st_key).float()
                        _svd_and_record(lname, W)
                        done += 1
                    except Exception as te:
                        log.warning(f"  Could not read {lname} from shard: {te}")
                        stats[lname] = {
                            "weight_l2_norm": None, "sv_top_k": None,
                            "spectral_norm": None, "stable_rank": None,
                            "nuclear_norm": None, "sv_ratio": None,
                            "svd_skipped": True,
                            "svd_skip_reason": f"shard read failed: {te}",
                        }
            log.info(f"  Shard {shard_idx}/{n_shards} done "
                    f"({done}/{len(layer_list)} from {shard_name})")
        except Exception as se:
            log.warning(f"  Could not open shard {shard_path}: {se}")
            for lname, _ in layer_list:
                stats[lname] = {
                    "weight_l2_norm": None, "sv_top_k": None,
                    "spectral_norm": None, "stable_rank": None,
                    "nuclear_norm": None, "sv_ratio": None,
                    "svd_skipped": True,
                    "svd_skip_reason": f"shard open failed: {se}",
                }

    # Phase 2: fallback layers (RAM-resident or non-offloaded models)
    for name, module in _fallback_layers:
        W = None
        try:
            pk = name + ".weight"
            for pname, param in model.named_parameters():
                if pname == pk and param.device.type != "meta":
                    W = param.data.detach().to(svd_target).float()
                    break
        except Exception:
            pass
        if W is None:
            try:
                w = module.weight
                if w is not None and w.device.type != "meta":
                    W = w.data.detach().to(svd_target).float()
            except Exception:
                pass
        if W is not None:
            _svd_and_record(name, W)
        else:
            stats[name] = {
                "weight_l2_norm": None, "sv_top_k": None, "spectral_norm": None,
                "stable_rank": None, "nuclear_norm": None, "sv_ratio": None,
                "svd_skipped": True,
                "svd_skip_reason": "all weight load strategies failed",
            }
            log.warning(f"  Could not load weight for {name} ? skipping.")

    meta_skipped = sum(1 for v in stats.values()
                    if v.get("svd_skip_reason", "").startswith("meta tensor"))
    svd_skipped  = sum(1 for v in stats.values()
                    if v.get("svd_skipped") and not v.get("svd_skip_reason", "").startswith("meta tensor"))
    computed     = len(stats) - meta_skipped - svd_skipped
    log.info(
        f"  Spectral stats: {computed} computed, "
        f"{svd_skipped} skipped (too large), "
        f"{meta_skipped} skipped (meta/offloaded tensors)."
    )
    if meta_skipped:
        log.info(
            f"  ?  {meta_skipped} layers had offloaded weights (device_map='auto').\n"
            f"     Spectral stats are not available for these layers.\n"
            f"     To compute spectral stats for all layers, load the model on a single\n"
            f"     device:  device_map='cpu'  or  device_map='cuda:0'."
        )
    return stats


# -----------------------------------------------------------------------------
# Layer-topology printer
# -----------------------------------------------------------------------------

def print_targetable_layers(model: nn.Module) -> None:
    print("\n" + "=" * 74)
    print("  TOPOLOGICAL MAP: INSTRUMENTABLE LAYERS")
    print("=" * 74)
    count = 0
    for name, module in model.named_modules():
        if not _is_hookable(module):
            continue
        dtype = _module_weight_dtype(module)
        kind  = type(module).__name__
        if isinstance(module, nn.Conv1d):                                    label = "[Conv1d/SSM]"
        elif isinstance(module, nn.Conv2d):                                  label = "[Conv2d/ANE]"
        elif _is_hf_conv1d(module):                                          label = "[HF-Conv1D]"
        elif _BNB_AVAILABLE and isinstance(module, _BNB_LINEAR_TYPES):       label = "[BNB quant]"
        else:                                                                label = f"[{_classify_linear(name)}]"
        print(f"  {label:<20} | {name:<52} | {kind:<18} | dtype={dtype}")
        count += 1
    print("-" * 74)
    print(f"  Total instrumentable layers: {count}")
    print("=" * 74 + "\n")


# -----------------------------------------------------------------------------
# Core Instrumenter
# -----------------------------------------------------------------------------

class UniversalInstrumenter:
    """
    Attaches forward hooks to targeted layers and accumulates per-layer
    statistics.  Supports two independent workflows:

    -- Workflow A: Instrumentation Only ("surgeon mode") ------------------
    Just attach hooks and return the patched model.  You own the forward
    loop ? use any inputs, any framework, any training harness.

        cfg  = InstrumentConfig(target_layers=["q_proj", "v_proj"])
        inst = UniversalInstrumenter(model, cfg)
        live = inst.get_instrumented_model()

        # Your own loop ? no restrictions
        for batch in my_dataloader:
            live(**batch)

        # Read counters at any point
        for name, c in inst.counters.items():
            print(name, c["activation_l1_sum"])

        inst.remove_hooks()   # always clean up when done

    -- Workflow B: Full Metrics Pipeline ("analyst mode") -----------------
    Feed a corpus through the built-in runner, then export a CSV report.
    All attack-detection features are opt-in via InstrumentConfig flags.

        cfg  = InstrumentConfig(track_variance=True, spectral_analysis=True)
        inst = UniversalInstrumenter(model, cfg)
        inst.run_corpus(batches)
        inst.export_to_csv("metrics.csv")
        inst.export_spectral_report("wpa.json")
        inst.remove_hooks()

    The two workflows are fully composable ? you can run your own loop
    (Workflow A) and then call export_to_csv() afterwards (Workflow B).

    Parameters
    ----------
    model  : nn.Module          Any HuggingFace / PyTorch model.
    config : InstrumentConfig   Runtime settings.  All attack-detection
                                features default to False (opt-in).
    """

    def __init__(self, model: nn.Module, config: InstrumentConfig):
        self.model   = model
        self.config  = config
        self.counters: dict[str, dict] = {}
        self._hooks:   list            = []
        self._lock     = threading.Lock()
        self.total_hooks_placed = 0

        # HSA: rolling buffer of the previous layer's output centroid
        # (one Tensor per device, released after each hook ? never accumulates)
        self._prev_layer_centroid: Optional[torch.Tensor] = None
        self._prev_layer_name:     Optional[str]          = None

        # WPA: spectral stats computed once at init (not during forward passes)
        self.spectral_stats: dict[str, dict] = {}

        self._instrument_model()

        if config.spectral_analysis:
            self.spectral_stats = compute_spectral_stats(
                model,
                top_k=config.spectral_top_k,
                spectral_device=config.spectral_device,
            )
            # Merge spectral stats into counters so they appear in the same export
            for name, s in self.spectral_stats.items():
                if name not in self.counters:
                    self.counters[name] = _make_counter()
                self.counters[name].update(s)

    # -- hook factory ----------------------------------------------------------

    def _get_backward_hook(self, layer_name: str):
        """Backward hook to track the L2 norm of the gradient (||\nabla L||_2)."""
        def hook(mod: nn.Module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                g = grad_output[0].detach().float()
                gnorm = g.norm(p=2).item()
                with self._lock:
                    if layer_name not in self.counters:
                        self.counters[layer_name] = _make_counter()
                    d = self.counters[layer_name]
                    if d["layer_kind"] is None:
                        d["layer_kind"] = type(mod).__name__
                    d["grad_norm_sum"] += gnorm
                    d["grad_norm_calls"] += 1
        return hook

    def _get_hook(self, layer_name: str, layer_type: str, module: nn.Module):
        cfg = self.config

        def hook(mod: nn.Module, inp, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output

            # Extract the layer input tensor for intra-layer cosine (CoTA metric)
            # inp is a tuple; the first element is the activation entering this layer
            inp_tensor = inp[0] if (isinstance(inp, (tuple, list)) and len(inp) > 0
                                    and isinstance(inp[0], torch.Tensor)) else None

            if layer_type == "fused_qkv":
                d = raw.shape[-1] // 3
                q, k, v = raw[..., :d], raw[..., d:2*d], raw[..., 2*d:]
                for suffix, t in (("_Q", q), ("_K", k), ("_V", v)):
                    self._record(f"{layer_name}{suffix}", t, mod, cfg, inp_tensor=inp_tensor)
            elif layer_type == "moe_gate":
                self._record(f"{layer_name}_Router", raw, mod, cfg, is_router=True, inp_tensor=inp_tensor)
            else:
                self._record(layer_name, raw, mod, cfg, inp_tensor=inp_tensor)

        return hook

    def _get_attention_hook(self, layer_name: str):
        """
        Separate hook for raw attention score matrices (pre-softmax or post-softmax).
        Used for CoTA per-head entropy computation.
        Shape expected: [batch, heads, seq, seq]
        """
        def hook(mod: nn.Module, inp, output):
            # Many attention implementations return (context, attn_weights) tuples
            attn_weights = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                candidate = output[1]
                if candidate is not None and isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                    attn_weights = candidate
            elif isinstance(output, torch.Tensor) and output.dim() == 4:
                attn_weights = output

            if attn_weights is None:
                return

            # attn_weights: [batch, heads, seq_q, seq_k]  (assumed post-softmax)
            w = attn_weights.detach().float()
            with self._lock:
                if layer_name not in self.counters:
                    self.counters[layer_name] = _make_counter()
                d = self.counters[layer_name]

                if d["layer_kind"] is None:
                    d["layer_kind"] = type(mod).__name__
                d["calls"] += 1

                # Per-head Shannon entropy: H = -sum(p * log(p+eps)) over seq_k
                # Low entropy = collapsed attention = CoTA signal
                eps = 1e-9
                # shape: [batch, heads, seq_q, seq_k]
                H = -(w * (w + eps).log()).sum(dim=-1)        # [batch, heads, seq_q]
                H_mean = H.mean(dim=(0, 2)).cpu()             # [heads]

                if d["attn_entropy_sum"] is None:
                    d["attn_entropy_sum"]  = torch.zeros_like(H_mean)
                    d["attn_head_std_sum"] = torch.zeros_like(H_mean)
                d["attn_entropy_sum"]  += H_mean
                d["attn_entropy_calls"] += 1

                # Per-head std across sequence positions (high std = concentrated head)
                H_std = H.std(dim=(0, 2)).cpu()               # [heads]
                d["attn_head_std_sum"] += H_std

        return hook

    # -- stat accumulator ------------------------------------------------------

    def _record(
        self,
        name:       str,
        tensor:     torch.Tensor,
        module:     nn.Module,
        cfg:        InstrumentConfig,
        is_router:  bool = False,
        inp_tensor: Optional[torch.Tensor] = None,
    ) -> None:

        t   = tensor.detach().float()
        dev = t.device

        with self._lock:
            if name not in self.counters:
                self.counters[name] = _make_counter()
            d = self.counters[name]

            # -- metadata (first call) -----------------------------------------
            if d["weight_dtype"] is None:
                d["weight_dtype"]     = str(_module_weight_dtype(module))
                d["activation_dtype"] = str(tensor.dtype)
                d["layer_kind"]       = type(module).__name__

            d["calls"]          += 1
            d["total_elements"] += t.numel()

            # -- sparsity ------------------------------------------------------
            d["zero_count"] += (t.abs() < cfg.sparsity_threshold).sum().item()

            # -- magnitude -----------------------------------------------------
            d["activation_l1_sum"] += t.abs().sum().item()
            d["activation_l2_sum"] += t.norm().item()

            # -- outlier -------------------------------------------------------
            cur_max = t.abs().max().item()
            if cur_max > d["max_abs_value"]:
                d["max_abs_value"] = cur_max

            # -- FLOPs + bandwidth ---------------------------------------------
            batch_tokens = math.prod(t.shape[:-1]) if t.dim() >= 1 else 1

            if isinstance(module, nn.Linear) or (
                _BNB_AVAILABLE and isinstance(module, _BNB_LINEAR_TYPES)
            ):
                in_f, out_f = module.in_features, module.out_features
                d["flops"]           += 2 * in_f * out_f * batch_tokens
                d["bandwidth_bytes"] += _estimate_bandwidth_linear(module, batch_tokens, in_f, out_f)

            elif _is_hf_conv1d(module):
                # HF Conv1D (pytorch_utils): weight shape is [in_features, out_features]
                in_f, out_f = module.weight.shape[0], module.weight.shape[1]
                d["flops"]           += 2 * in_f * out_f * batch_tokens
                d["bandwidth_bytes"] += _estimate_bandwidth_linear(module, batch_tokens, in_f, out_f)

            elif isinstance(module, nn.Conv1d):
                l_out = t.shape[-1] if t.dim() >= 2 else 1
                in_c, out_c, k = module.in_channels, module.out_channels, module.kernel_size[0]
                d["flops"] += 2 * in_c * out_c * k * l_out
                w_bpe, act_bpe = _bytes_per_element(_module_weight_dtype(module)), _bytes_per_element(tensor.dtype)
                d["bandwidth_bytes"] += int(in_c * out_c * k * w_bpe + t.numel() * act_bpe * 2)

            elif isinstance(module, nn.Conv2d):
                h_out = t.shape[-2] if t.dim() >= 3 else 1
                w_out = t.shape[-1] if t.dim() >= 2 else 1
                in_c, out_c = module.in_channels, module.out_channels
                kh, kw = module.kernel_size
                d["flops"] += 2 * in_c * out_c * kh * kw * h_out * w_out
                w_bpe, act_bpe = _bytes_per_element(_module_weight_dtype(module)), _bytes_per_element(tensor.dtype)
                d["bandwidth_bytes"] += int(in_c * out_c * kh * kw * w_bpe + t.numel() * act_bpe * 2)

            # -- per-neuron activation sum (base backdoor signal) --------------
            if cfg.per_neuron_tracking and t.dim() >= 2:
                collapse = list(range(t.dim() - 1))
                neuron_means = t.abs().mean(dim=collapse).cpu()
                if d["neuron_activation_sum"] is None:
                    d["neuron_activation_sum"] = torch.zeros_like(neuron_means)
                d["neuron_activation_sum"] += neuron_means

                # Per-batch storage for permutation test.
                if cfg.store_batch_activations:
                    if d["batch_neuron_activations"] is None:
                        d["batch_neuron_activations"] = []
                    d["batch_neuron_activations"].append(neuron_means.clone())

            # -- activation histogram (GPU ? CPU) -----------------------------
            hist = torch.histc(t.abs().to(dev), bins=cfg.histogram_bins,
                            min=0.0, max=cfg.histogram_max).cpu()
            if d["activation_hist"] is None:
                d["activation_hist"] = torch.zeros(cfg.histogram_bins)
            d["activation_hist"] += hist

            # -- MoE top-K routing ---------------------------------------------
            if is_router:
                k = min(cfg.top_k_experts, t.shape[-1])
                _, top_idx = torch.topk(t, k=k, dim=-1)
                uniq, cnts = torch.unique(top_idx, return_counts=True)
                for idx, cnt in zip(uniq.tolist(), cnts.tolist()):
                    d["expert_usage"][idx] = d["expert_usage"].get(idx, 0) + cnt

            # =================================================================
            # DPA ? Data Poisoning Attack metrics
            # =================================================================

            if cfg.track_variance and t.dim() >= 2:
                collapse = list(range(t.dim() - 1))
                neuron_vals = t.abs().mean(dim=collapse).cpu()

                if d["neuron_welford_mean"] is None:
                    d["neuron_welford_mean"] = torch.zeros_like(neuron_vals)
                    d["neuron_welford_M2"]   = torch.zeros_like(neuron_vals)

                d["neuron_welford_mean"], d["neuron_welford_M2"] = _welford_update(
                    d["calls"], d["neuron_welford_mean"], d["neuron_welford_M2"], neuron_vals
                )

            if cfg.track_position_map and t.dim() == 3:
                pos_mag = t.abs().mean(dim=(0, 2)).cpu()   # [seq_len]
                if d["position_act_sum"] is None:
                    d["position_act_sum"] = torch.zeros_like(pos_mag)
                elif d["position_act_sum"].shape != pos_mag.shape:
                    max_len = max(d["position_act_sum"].shape[0], pos_mag.shape[0])
                    old = d["position_act_sum"]
                    d["position_act_sum"] = torch.zeros(max_len)
                    d["position_act_sum"][:old.shape[0]] = old
                    pos_mag_padded = torch.zeros(max_len)
                    pos_mag_padded[:pos_mag.shape[0]] = pos_mag
                    pos_mag = pos_mag_padded
                d["position_act_sum"] += pos_mag
                d["position_act_calls"] += 1

            # =================================================================
            # HSA ? Hidden State Attack metrics
            # =================================================================

            if cfg.track_inter_layer_cosine and t.dim() >= 2:
                curr_centroid = t.mean(dim=list(range(t.dim() - 1))).cpu()  # [hidden]

                if self._prev_layer_centroid is not None:
                    prev = self._prev_layer_centroid
                    if prev.shape == curr_centroid.shape:
                        cos = F.cosine_similarity(
                            prev.unsqueeze(0), curr_centroid.unsqueeze(0)
                        ).item()
                        d["inter_layer_cosine_sum"]   += cos
                        d["inter_layer_cosine_calls"] += 1

                self._prev_layer_centroid = curr_centroid
                self._prev_layer_name     = name

            if cfg.track_centroid and t.dim() >= 2:
                collapse = list(range(t.dim() - 1))
                centroid = t.mean(dim=collapse).cpu()   # [hidden]
                if d["activation_centroid"] is None:
                    d["activation_centroid"] = torch.zeros_like(centroid)
                d["activation_centroid"] += (centroid - d["activation_centroid"]) / d["calls"]

            # =================================================================
            # CoTA ? intra-layer cosine
            # =================================================================
            if cfg.track_intra_layer_cosine and inp_tensor is not None and t.dim() >= 2:
                inp_t = inp_tensor.detach().float()
                if inp_t.shape == t.shape:
                    collapse = list(range(t.dim() - 1))
                    in_cen  = inp_t.mean(dim=collapse).cpu()   # [hidden]
                    out_cen = t.mean(dim=collapse).cpu()
                    if in_cen.norm() > 1e-9 and out_cen.norm() > 1e-9:
                        cos = F.cosine_similarity(in_cen.unsqueeze(0), out_cen.unsqueeze(0)).item()
                        d["intra_layer_cosine_sum"]   += cos
                        d["intra_layer_cosine_calls"] += 1

            # =================================================================
            # CoTA ? temporal L2 variance
            # =================================================================
            if cfg.track_token_l2_variance and t.dim() == 3:
                token_norms = t.norm(dim=-1)          # [batch, seq_len]
                token_var   = token_norms.var(dim=-1).mean().item()   # scalar avg over batch
                d["token_l2_var_sum"]   += token_var
                d["token_l2_var_calls"] += 1

            # =================================================================
            # WPA ? activation kurtosis
            # =================================================================
            if cfg.track_kurtosis and t.numel() >= 4:
                flat   = t.reshape(-1)
                mu     = flat.mean()
                sigma2 = flat.var(unbiased=False) + 1e-9
                kurt   = ((flat - mu).pow(4).mean() / sigma2.pow(2)).item()
                d["activation_kurtosis_sum"]   += kurt
                d["activation_kurtosis_calls"] += 1

            # =================================================================
            # WPA ? dead neuron reactivation
            # =================================================================
            if cfg.track_dead_neurons and t.dim() >= 2:
                collapse = list(range(t.dim() - 1))
                fired = (t.abs().mean(dim=collapse) >= cfg.sparsity_threshold).cpu()
                if d["dead_neuron_mask"] is None:
                    d["dead_neuron_mask"]    = ~fired   # True = still dead
                    d["dead_neuron_initial"] = (~fired).clone()
                else:
                    d["dead_neuron_mask"] = d["dead_neuron_mask"] & ~fired

            # =================================================================
            # HSA ? L1/L2 norm ratio
            # =================================================================
            if cfg.track_l1_l2_ratio:
                l1 = t.abs().sum().item()
                l2 = t.norm().item() + 1e-9
                d["l1_l2_ratio_sum"]   += l1 / l2
                d["l1_l2_ratio_calls"] += 1

            # =================================================================
            # DPA ? cross-neuron co-activation variance
            # =================================================================
            if cfg.track_coact_variance and t.dim() >= 2:
                collapse = list(range(t.dim() - 1))
                neuron_means = t.abs().mean(dim=collapse)   # [feature_dim]
                coact_var    = neuron_means.var(unbiased=False).item()
                d["coact_variance_sum"]   += coact_var
                d["coact_variance_calls"] += 1

            # =================================================================
            # NEW ? Activation Skewness
            # =================================================================
            if cfg.track_skewness and t.numel() >= 3:
                flat  = t.reshape(-1)
                mu    = flat.mean()
                sigma = flat.std(unbiased=False) + 1e-9
                skew  = ((flat - mu).pow(3).mean() / sigma.pow(3)).item()
                d["activation_skewness_sum"]   += skew
                d["activation_skewness_calls"] += 1

            # =================================================================
            # NEW ? Activation Entropy H(X) = -sum(p * log(p))
            # =================================================================
            if cfg.track_activation_entropy and t.numel() > 1:
                abs_t = t.abs()
                p = abs_t / (abs_t.sum() + 1e-9)
                ent = -(p * (p + 1e-9).log()).sum().item()
                d["activation_entropy_sum"]   += ent
                d["activation_entropy_calls"] += 1

    # -- model wiring ----------------------------------------------------------

    def _instrument_model(self) -> None:
        log.info("Attaching instrumentation hooks ...")
        cfg = self.config

        for name, module in self.model.named_modules():
            if not _is_hookable(module):
                continue
            if cfg.target_layers and not any(t in name for t in cfg.target_layers):
                continue
            if isinstance(module, nn.Conv1d) and not cfg.hook_conv1d: continue
            if isinstance(module, nn.Conv2d) and not cfg.hook_conv2d: continue

            # -- main forward hook ---------------------------------------------
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                layer_type = "conv"
            elif _BNB_AVAILABLE and isinstance(module, _BNB_LINEAR_TYPES):
                layer_type = _classify_linear(name) or "bnb_linear"
            elif _is_hf_conv1d(module):
                layer_type = _classify_linear(name)
            else:
                layer_type = _classify_linear(name)  # never None ? always hooks

            h = module.register_forward_hook(self._get_hook(name, layer_type, module))
            self._hooks.append(h)
            self.total_hooks_placed += 1
            log.info(f"  Hooked: {name}  [{layer_type}]")

            # -- NEW: Gradient hook --------------------------------------------
            if cfg.track_gradient_norm:
                h_bwd = module.register_full_backward_hook(self._get_backward_hook(name))
                self._hooks.append(h_bwd)
                log.info(f"  Gradient hook attached: {name}")

        # -- CoTA: attention score hooks ---------------------------------------
        if cfg.attention_entropy:
            for name, module in self.model.named_modules():
                if type(module).__name__ == "LlamaAttention":
                    h = module.register_forward_hook(self._get_attention_hook(name))
                    self._hooks.append(h)
                    log.info(f"  Attn-entropy hooked: {name}")

        log.info(f"? {self.total_hooks_placed} layers instrumented.")

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        log.info("All hooks removed.")

    def get_instrumented_model(self) -> nn.Module:
        """
        Return the model with all hooks already attached.

        This is the entry point for **Workflow A** (instrumentation only).
        The returned object is the *same* model instance ? not a copy ? so it
        is fully compatible with any downstream framework, training loop, or
        evaluation harness.

        Counters accumulate automatically on every forward pass.  Read them
        at any time via ``self.counters``.  Call ``remove_hooks()`` when done
        to restore the model to its original unhooked state.

        Returns
        -------
        nn.Module
            The original model with forward hooks attached.

        Example
        -------
        >>> cfg  = InstrumentConfig(target_layers=["q_proj", "v_proj"])
        >>> inst = UniversalInstrumenter(model, cfg)
        >>> live = inst.get_instrumented_model()
        >>> # plug `live` into any trainer / eval harness
        >>> trainer.train(live, dataloader)
        >>> # inspect counters afterwards
        >>> for name, c in inst.counters.items():
        ...     print(name, c["calls"], c["activation_l1_sum"])
        >>> inst.remove_hooks()
        """
        if not self._hooks:
            log.warning(
                "get_instrumented_model() called but no hooks are attached. "
                "Did you already call remove_hooks()?"
            )
        return self.model

    # -- corpus runner ---------------------------------------------------------

    def run_corpus(self, batches: list[torch.Tensor]) -> None:
        """Run model on a list of input-id tensors and accumulate all stats."""
        self.model.eval()
        device = _get_execution_device(self.model)
        kwargs = {"output_attentions": True} if self.config.attention_entropy else {}

        _orig_is_autocast_enabled = torch.is_autocast_enabled
        _patched = False
        try:
            torch.is_autocast_enabled("cpu")
        except (TypeError, RuntimeError):
            def _safe_is_autocast_enabled(device_type=None):
                try:
                    if device_type is None:
                        return _orig_is_autocast_enabled()
                    return _orig_is_autocast_enabled(device_type)
                except (TypeError, RuntimeError):
                    return False
            torch.is_autocast_enabled = _safe_is_autocast_enabled
            _patched = True
            log.warning(
                "PyTorch < 2.4 detected: patching torch.is_autocast_enabled for "
                "Qwen3/LLaMA3 compatibility. "
                "Run  pip install --upgrade torch  to fix permanently. "
                "Impact: attention entropy metrics will be N/A."
            )

        try:
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    self._prev_layer_centroid = None   # reset inter-layer buffer per batch
                    try:
                        self.model(batch.to(device), **kwargs)
                    except TypeError:
                        self.model(batch.to(device))
                    log.info(f"  Batch {i + 1}/{len(batches)} done.")
        finally:
            if _patched:
                torch.is_autocast_enabled = _orig_is_autocast_enabled

    # -- snapshot / diff infrastructure ---------------------------------------

    def snapshot(self) -> dict:
        """Deep-copy current counters for later comparison (DPA/HSA diffing)."""
        return copy.deepcopy(self.counters)

    def diff_snapshots(
        self,
        baseline: dict,
        suspect:  dict,
        top_k:    int = 20,
    ) -> list[dict]:
        """Full cross-attack diff between two corpus snapshots."""
        results = []
        for name in baseline:
            if name not in suspect:
                continue
            b, s = baseline[name], suspect[name]

            if b.get("neuron_activation_sum") is None and s.get("neuron_activation_sum") is None:
                continue

            entry: dict = {"layer": name}

            # -- neuron activation delta (DPA/HSA) -----------------------------
            b_vec, s_vec = b.get("neuron_activation_sum"), s.get("neuron_activation_sum")
            if b_vec is not None and s_vec is not None:
                b_n = b_vec / (b_vec.norm() + 1e-9)
                s_n = s_vec / (s_vec.norm() + 1e-9)
                delta = (s_n - b_n).abs()
                k_act = min(top_k, delta.numel())
                top   = delta.topk(k_act)
                entry["neuron_max_delta"]        = round(top.values[0].item(), 6)
                entry["neuron_mean_delta"]        = round(delta.mean().item(), 6)
                entry["suspicious_neuron_ids"]    = top.indices.tolist()
                entry["top_neuron_delta_values"]  = [round(v, 6) for v in top.values.tolist()]

            # -- JS divergence on activation histograms (HSA) -----------------
            b_hist, s_hist = b.get("activation_hist"), s.get("activation_hist")
            if b_hist is not None and s_hist is not None:
                entry["js_divergence"] = round(_js_divergence(b_hist, s_hist), 6)

            # -- centroid L2 drift (HSA steering vector) -----------------------
            b_cen, s_cen = b.get("activation_centroid"), s.get("activation_centroid")
            if b_cen is not None and s_cen is not None and b_cen.shape == s_cen.shape:
                entry["centroid_l2_drift"] = round((s_cen - b_cen).norm().item(), 6)

            # -- FLOPs ratio (BadThink CoTA) -----------------------------------
            b_fl, s_fl = b.get("flops", 0), s.get("flops", 0)
            if b_fl > 0:
                entry["flops_ratio"] = round(s_fl / b_fl, 4)

            # -- variance change (DPA) -----------------------------------------
            b_M2, s_M2 = b.get("neuron_welford_M2"), s.get("neuron_welford_M2")
            b_calls, s_calls = b.get("calls", 1), s.get("calls", 1)
            if b_M2 is not None and s_M2 is not None:
                b_var = b_M2 / max(b_calls, 1)
                s_var = s_M2 / max(s_calls, 1)
                var_delta = (s_var - b_var).abs()
                entry["max_variance_delta"] = round(var_delta.max().item(), 6)
                entry["mean_variance_delta"] = round(var_delta.mean().item(), 6)

            results.append(entry)

        return sorted(results, key=lambda x: x.get("neuron_max_delta", 0.0), reverse=True)

    def export_backdoor_report(
        self,
        baseline: dict,
        filepath: str = "backdoor_report.json",
        top_k:    int = 20,
    ) -> None:
        report = self.diff_snapshots(baseline, self.counters, top_k=top_k)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"? Backdoor report ? {os.path.abspath(filepath)}")

    def export_spectral_report(self, filepath: str = "spectral_report.json") -> None:
        """Export WPA spectral stats as a standalone JSON (sorted by sv_ratio)."""
        if not self.spectral_stats:
            log.warning("No spectral stats ? run with --spectral-analysis.")
            return
        data = sorted(
            [{"layer": k, **v} for k, v in self.spectral_stats.items()],
            key=lambda x: x.get("sv_ratio") or 0.0, reverse=True
        )
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"? Spectral report ? {os.path.abspath(filepath)}")

    # -- Publishability Addition 1: Permutation test (corrected) -------------

    def permutation_test_diff(
        self,
        baseline: dict,
        suspect:  dict,
        n_perms:  int = 200,
        top_k:    int = 20,
    ) -> list[dict]:
        """Add statistically valid empirical p-values to the snapshot diff."""
        raw_diff = self.diff_snapshots(baseline, suspect, top_k=top_k)

        enriched = []
        for entry in raw_diff:
            name = entry["layer"]
            b_batches = (baseline.get(name) or {}).get("batch_neuron_activations")
            s_batches = (suspect.get(name)  or {}).get("batch_neuron_activations")

            n_b = len(b_batches) if b_batches else 0
            n_s = len(s_batches) if s_batches else 0
            entry["n_baseline_calls"] = n_b
            entry["n_suspect_calls"]  = n_s

            if not b_batches or not s_batches or n_b < 2 or n_s < 2:
                entry["p_value"]           = None
                entry["significant"]       = None
                entry["n_perms"]           = n_perms
                entry["data_insufficient"] = True
                if n_b < 2 or n_s < 2:
                    log.warning(
                        f"  permutation_test: {name} has only {n_b} baseline / "
                        f"{n_s} suspect batch vectors ? need >=2 each. "
                        f"Enable store_batch_activations=True and run more batches."
                    )
                enriched.append(entry)
                continue

            if n_b < 5 or n_s < 5:
                log.warning(
                    f"  permutation_test: {name} ? low sample size "
                    f"({n_b} baseline, {n_s} suspect). P-values are approximate; "
                    f"use >=5 batches per corpus for reliable results."
                )

            B = torch.stack(b_batches)    # [n_b, feature_dim]
            S = torch.stack(s_batches)    # [n_s, feature_dim]

            def _stat(grp_a: torch.Tensor, grp_b: torch.Tensor) -> float:
                ma = grp_a.mean(dim=0)
                mb = grp_b.mean(dim=0)
                ma = ma / (ma.norm() + 1e-9)
                mb = mb / (mb.norm() + 1e-9)
                return (mb - ma).abs().max().item()

            T_obs = _stat(B, S)

            pool = torch.cat([B, S], dim=0)   # [n_b + n_s, feature_dim]
            n_total = pool.shape[0]
            exceed_count = 0

            for _ in range(n_perms):
                perm_idx = torch.randperm(n_total)
                perm_b   = pool[perm_idx[:n_b]]
                perm_s   = pool[perm_idx[n_b:]]
                if _stat(perm_b, perm_s) >= T_obs:
                    exceed_count += 1

            p_val = (exceed_count + 1) / (n_perms + 1)
            entry["p_value"]           = round(p_val, 4)
            entry["significant"]       = bool(p_val < 0.05)
            entry["n_perms"]           = n_perms
            entry["data_insufficient"] = False
            enriched.append(entry)

        enriched.sort(key=lambda x: (
            1 if x.get("data_insufficient") else 0,
            0 if x.get("significant") else 1,
            x.get("p_value") or 1.0,
        ))
        return enriched

    def export_significance_report(
        self,
        baseline: dict,
        filepath: str = "significance_report.json",
        n_perms:  int = 200,
        top_k:    int = 20,
    ) -> None:
        """Run permutation test and write the significance-annotated report."""
        report = self.permutation_test_diff(baseline, self.counters, n_perms=n_perms, top_k=top_k)
        significant = [e for e in report if e.get("significant")]
        log.info(f"  Significant layers (p<0.05): {len(significant)} / {len(report)}")
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"? Significance report ? {os.path.abspath(filepath)}")

    # -- Publishability Addition 2: HTML Visualisation Report -----------------

    def export_html_report(self, filepath: str = "report.html") -> None:
        """Export a self-contained, interactive HTML report of all collected metrics."""
        if not self.counters:
            log.warning("No metrics to visualise ? run a forward pass first.")
            return

        rows = []
        for name, d in self.counters.items():
            calls = d["calls"]
            if calls == 0:
                continue

            hist_svg = "N/A"
            if d["activation_hist"] is not None:
                hist_svg = _make_hist_svg(d["activation_hist"])

            kurt_val  = (d["activation_kurtosis_sum"] / d["activation_kurtosis_calls"]
                        if d["activation_kurtosis_calls"] > 0 else None)
            intra_val = (d["intra_layer_cosine_sum"] / d["intra_layer_cosine_calls"]
                        if d["intra_layer_cosine_calls"] > 0 else None)
            coact_val = (d["coact_variance_sum"] / d["coact_variance_calls"]
                        if d["coact_variance_calls"] > 0 else None)
            ent_min   = None
            if d["attn_entropy_sum"] is not None and d["attn_entropy_calls"] > 0:
                ent_min = (d["attn_entropy_sum"] / d["attn_entropy_calls"]).min().item()

            dead_react = 0
            if d["dead_neuron_initial"] is not None and d["dead_neuron_mask"] is not None:
                dead_react = int((d["dead_neuron_initial"] & ~d["dead_neuron_mask"]).sum().item())

            skew_val = (d["activation_skewness_sum"] / d["activation_skewness_calls"]
                        if d["activation_skewness_calls"] > 0 else None)
            act_ent_val = (d["activation_entropy_sum"] / d["activation_entropy_calls"]
                        if d["activation_entropy_calls"] > 0 else None)
            grad_norm_val = (d["grad_norm_sum"] / d["grad_norm_calls"]
                        if d["grad_norm_calls"] > 0 else None)

            rows.append({
                "name":       name,
                "kind":       d["layer_kind"] or "?",
                "calls":      calls,
                "l1_avg":     round(d["activation_l1_sum"] / calls, 3),
                "sparsity":   round(d["zero_count"] / max(d["total_elements"], 1) * 100, 2),
                "kurtosis":   round(kurt_val, 3) if kurt_val is not None else None,
                "skewness":   round(skew_val, 3) if skew_val is not None else None,
                "act_entropy":round(act_ent_val, 4) if act_ent_val is not None else None,
                "grad_norm":  round(grad_norm_val, 4) if grad_norm_val is not None else None,
                "intra_cos":  round(intra_val, 4) if intra_val is not None else None,
                "coact_var":  round(coact_val, 6) if coact_val is not None else None,
                "ent_min":    round(ent_min, 4) if ent_min is not None else None,
                "dead_react": dead_react,
                "hist_svg":   hist_svg,
            })

        html = _build_html_report(rows)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        log.info(f"? HTML report ? {os.path.abspath(filepath)}")

    # -- CSV export ------------------------------------------------------------

    def export_to_csv(self, filepath: str = "metrics.csv") -> None:
        """Export all collected metrics to a single flat CSV."""
        if not self.counters:
            log.warning("No metrics captured ? run a forward pass first.")
            return

        headers = [
            # identity
            "layer_name", "layer_kind", "calls",
            # power
            "l1_avg", "l2_avg", "flops", "bandwidth_mb", "weight_dtype", "activation_dtype",
            # forensics
            "sparsity_pct", "max_abs_val", "hist_peak_bin", "hist_bimodal_score",
            # MoE
            "top_expert_id", "top_expert_pct", "gini_score",
            # DPA
            "neuron_variance_max", "neuron_variance_mean", "position_peak_bin",
            "coact_variance_avg",
            # WPA
            "spectral_norm", "stable_rank", "sv_ratio", "weight_l2_norm",
            "kurtosis_avg", "dead_neuron_count", "dead_neuron_reactivated",
            # HSA
            "inter_layer_cosine_avg", "centroid_l2_norm",
            "l1_l2_ratio_avg",
            # CoTA
            "attn_entropy_avg", "attn_entropy_min_head", "attn_head_std_avg",
            "intra_layer_cosine_avg", "token_l2_var_avg",
            # NEW METRICS
            "skewness_avg", "activation_entropy_avg", "grad_norm_avg"
        ]

        ghost_rows_skipped = 0

        with open(filepath, mode="w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)

            for name, d in self.counters.items():
                calls = d["calls"]

                if calls == 0:
                    ghost_rows_skipped += 1
                    continue

                # -- power ----------------------------------------------------
                l1  = round(d["activation_l1_sum"] / max(calls, 1), 4)
                l2  = round(d["activation_l2_sum"] / max(calls, 1), 4)
                bw  = round(d["bandwidth_bytes"] / 1e6, 4)

                # -- forensics -------------------------------------------------
                sparsity = round(d["zero_count"] / max(d["total_elements"], 1) * 100, 4)
                max_abs  = round(d["max_abs_value"], 4)
                hist_peak = hist_bimodal = "N/A"
                if d["activation_hist"] is not None:
                    hist = d["activation_hist"]
                    hist_peak = int(hist.argmax().item())
                    mid = len(hist) // 2
                    tot = hist.sum().item()
                    hist_bimodal = round(hist[mid:].sum().item() / tot, 4) if tot > 0 else "N/A"

                # -- MoE -------------------------------------------------------
                top_id = top_pct = gini = "N/A"
                if d["expert_usage"]:
                    tot_r  = sum(d["expert_usage"].values())
                    top_id = max(d["expert_usage"], key=d["expert_usage"].get)
                    top_pct = round(d["expert_usage"][top_id] / tot_r * 100, 2)
                    gini = round(calculate_gini(d["expert_usage"]), 4)

                # -- DPA -------------------------------------------------------
                var_max = var_mean = pos_peak = coact_var_avg = "N/A"
                if d["neuron_welford_M2"] is not None and calls > 1:
                    var = d["neuron_welford_M2"] / calls
                    var_max  = round(var.max().item(), 6)
                    var_mean = round(var.mean().item(), 6)
                if d["position_act_sum"] is not None:
                    pos_peak = int(d["position_act_sum"].argmax().item())
                if d["coact_variance_calls"] > 0:
                    coact_var_avg = round(d["coact_variance_sum"] / d["coact_variance_calls"], 6)

                # -- WPA -------------------------------------------------------
                spec_norm = d.get("spectral_norm") or "N/A"
                stab_rank = d.get("stable_rank")   or "N/A"
                sv_ratio  = d.get("sv_ratio")      or "N/A"
                w_l2      = d.get("weight_l2_norm") or "N/A"
                kurt_avg  = "N/A"
                if d["activation_kurtosis_calls"] > 0:
                    kurt_avg = round(d["activation_kurtosis_sum"] / d["activation_kurtosis_calls"], 4)
                dead_count = dead_react = "N/A"
                if d["dead_neuron_mask"] is not None:
                    dead_count = int(d["dead_neuron_mask"].sum().item())
                if d["dead_neuron_initial"] is not None and d["dead_neuron_mask"] is not None:
                    reactivated = d["dead_neuron_initial"] & ~d["dead_neuron_mask"]
                    dead_react  = int(reactivated.sum().item())

                # -- HSA -------------------------------------------------------
                il_cos = "N/A"
                if d["inter_layer_cosine_calls"] > 0:
                    il_cos = round(d["inter_layer_cosine_sum"] / d["inter_layer_cosine_calls"], 4)
                cen_norm = "N/A"
                if d["activation_centroid"] is not None:
                    cen_norm = round(d["activation_centroid"].norm().item(), 4)
                l1l2r = "N/A"
                if d["l1_l2_ratio_calls"] > 0:
                    l1l2r = round(d["l1_l2_ratio_sum"] / d["l1_l2_ratio_calls"], 4)

                # -- CoTA ------------------------------------------------------
                ent_avg = ent_min = head_std = "N/A"
                if d["attn_entropy_sum"] is not None and d["attn_entropy_calls"] > 0:
                    ent_per_head = d["attn_entropy_sum"] / d["attn_entropy_calls"]
                    ent_avg = round(ent_per_head.mean().item(), 4)
                    ent_min = round(ent_per_head.min().item(), 4)
                if d["attn_head_std_sum"] is not None and d["attn_entropy_calls"] > 0:
                    head_std = round((d["attn_head_std_sum"] / d["attn_entropy_calls"]).mean().item(), 4)
                intra_cos = "N/A"
                if d["intra_layer_cosine_calls"] > 0:
                    intra_cos = round(d["intra_layer_cosine_sum"] / d["intra_layer_cosine_calls"], 4)
                tok_l2v = "N/A"
                if d["token_l2_var_calls"] > 0:
                    tok_l2v = round(d["token_l2_var_sum"] / d["token_l2_var_calls"], 6)

                # -- NEW METRICS -----------------------------------------------
                skew_avg = "N/A"
                if d["activation_skewness_calls"] > 0:
                    skew_avg = round(d["activation_skewness_sum"] / d["activation_skewness_calls"], 4)
                act_ent_avg = "N/A"
                if d["activation_entropy_calls"] > 0:
                    act_ent_avg = round(d["activation_entropy_sum"] / d["activation_entropy_calls"], 4)
                grad_norm_avg = "N/A"
                if d["grad_norm_calls"] > 0:
                    grad_norm_avg = round(d["grad_norm_sum"] / d["grad_norm_calls"], 4)

                writer.writerow([
                    name, d["layer_kind"], calls,
                    l1, l2, d["flops"], bw, d["weight_dtype"] or "unknown", d["activation_dtype"] or "unknown",
                    sparsity, max_abs, hist_peak, hist_bimodal,
                    top_id, top_pct, gini,
                    var_max, var_mean, pos_peak, coact_var_avg,
                    spec_norm, stab_rank, sv_ratio, w_l2, kurt_avg, dead_count, dead_react,
                    il_cos, cen_norm, l1l2r,
                    ent_avg, ent_min, head_std, intra_cos, tok_l2v,
                    skew_avg, act_ent_avg, grad_norm_avg
                ])

        if ghost_rows_skipped:
            log.info(f"  ?  {ghost_rows_skipped} ghost rows skipped (calls=0) ? "
                    f"fused QKV parent entries redirected to _Q/_K/_V at hook time.")
        log.info(f"? Metrics CSV ? {os.path.abspath(filepath)}")


# -----------------------------------------------------------------------------
# HTML Report helpers  (Publishability Addition 2)
# -----------------------------------------------------------------------------

def _make_hist_svg(hist: torch.Tensor, width: int = 80, height: int = 24) -> str:
    """Render a histogram tensor as a tiny inline SVG bar chart."""
    vals = hist.float()
    mx   = vals.max().item()
    if mx == 0:
        return "<svg/>"
    n    = len(vals)
    bw   = width / n
    bars = []
    for i, v in enumerate(vals.tolist()):
        bh = int((v / mx) * height)
        x  = i * bw
        y  = height - bh
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{bh}" fill="#4f87c9"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">{"".join(bars)}</svg>')


def _cell_color(value, low_bad: bool, lo: float, hi: float) -> str:
    """Return a CSS background colour interpolated from green (good) to red (bad)."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if hi == lo:
        return ""
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    if low_bad:   # low value = anomalous ? red when low
        t = 1.0 - t
    hue = int((1.0 - t) * 120)
    return f"background:hsl({hue},70%,88%)"


def _build_html_report(rows: list[dict]) -> str:
    """Build a self-contained HTML string from the pre-computed row data."""

    def fmt(v):
        return "?" if v is None else str(v)

    row_html = []
    for r in rows:
        kc  = _cell_color(r["kurtosis"],  low_bad=False, lo=2.0, hi=20.0)
        sc  = _cell_color(r["skewness"],  low_bad=False, lo=0.0, hi=10.0)
        ic  = _cell_color(r["intra_cos"], low_bad=True,  lo=0.5, hi=1.0)
        cvc = _cell_color(r["coact_var"], low_bad=True,  lo=0.0, hi=0.5)
        ec  = _cell_color(r["ent_min"],   low_bad=True,  lo=0.0, hi=3.0)
        drc = _cell_color(r["dead_react"],low_bad=False, lo=0,   hi=50)

        row_html.append(
            f"<tr>"
            f"<td class='nm'>{r['name']}</td>"
            f"<td>{r['kind']}</td>"
            f"<td>{r['calls']}</td>"
            f"<td>{r['l1_avg']}</td>"
            f"<td>{r['sparsity']}%</td>"
            f"<td style='{kc}'>{fmt(r['kurtosis'])}</td>"
            f"<td style='{sc}'>{fmt(r['skewness'])}</td>"
            f"<td>{fmt(r['act_entropy'])}</td>"
            f"<td>{fmt(r['grad_norm'])}</td>"
            f"<td style='{ic}'>{fmt(r['intra_cos'])}</td>"
            f"<td style='{cvc}'>{fmt(r['coact_var'])}</td>"
            f"<td style='{ec}'>{fmt(r['ent_min'])}</td>"
            f"<td style='{drc}'>{r['dead_react']}</td>"
            f"<td>{r['hist_svg']}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Instrumenter Report</title>
<style>
body {{ font-family: system-ui, sans-serif; font-size: 13px; margin: 0; background: #f5f5f5; }}
h1   {{ background: #1e3a5f; color: #fff; margin: 0; padding: 12px 20px; font-size: 18px; }}
.sub {{ background: #2d5a8e; color: #cde; padding: 4px 20px; font-size: 12px; }}
#filter {{ margin: 10px 20px; padding: 6px 10px; border: 1px solid #bbb;
            border-radius: 4px; width: 320px; font-size: 13px; }}
table  {{ border-collapse: collapse; width: 100%; margin: 0 0 40px; }}
th     {{ background: #1e3a5f; color: #fff; padding: 7px 8px; text-align: left;
            cursor: pointer; white-space: nowrap; position: sticky; top: 0; }}
th:hover {{ background: #2d5a8e; }}
td     {{ padding: 5px 8px; border-bottom: 1px solid #e0e0e0; vertical-align: middle; }}
tr:hover {{ background: #eef4ff; }}
.nm  {{ font-family: monospace; font-size: 11px; max-width: 340px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.leg {{ display: flex; gap: 20px; padding: 8px 20px; font-size: 12px; color: #555; }}
.swatch {{ display:inline-block; width:14px; height:14px;
            border-radius:3px; vertical-align:middle; margin-right:4px; }}
</style>
</head>
<body>
<h1>Universal AI Research Instrumenter ? Activation Report</h1>
<div class="sub">v4.2 &nbsp;|&nbsp; {len(rows)} layers &nbsp;|&nbsp;
Colour: <span style="color:#c66">red = anomalous signal</span> &nbsp;
<span style="color:#396">green = healthy range</span></div>
<div class="leg">
<span><b>Kurtosis</b>: high &gt; 3 = dormant neuron reactivation</span>
<span><b>Skewness</b>: heavy positive tail = asymmetric activations</span>
<span><b>Intra-cos</b>: low = layer is violently overwriting representation</span>
<span><b>Coact-var</b>: low = co-activation flood (trigger override)</span>
<span><b>Ent-min</b>: low = attention sink / tunnel vision</span>
<span><b>Dead-react</b>: count of neurons reactivated from baseline dead set</span>
</div>
<input id="filter" type="text" placeholder="Filter layer names..." oninput="filterRows(this.value)">
<table id="tbl">
<thead><tr>
<th onclick="sortTable(0)">Layer</th>
<th onclick="sortTable(1)">Kind</th>
<th onclick="sortTable(2)">Calls</th>
<th onclick="sortTable(3)">L1 avg</th>
<th onclick="sortTable(4)">Sparsity</th>
<th onclick="sortTable(5)">Kurtosis</th>
<th onclick="sortTable(6)">Skewness</th>
<th onclick="sortTable(7)">Act Entropy</th>
<th onclick="sortTable(8)">Grad Norm</th>
<th onclick="sortTable(9)">Intra-cos</th>
<th onclick="sortTable(10)">Coact-var</th>
<th onclick="sortTable(11)">Ent-min</th>
<th onclick="sortTable(12)">Dead-react</th>
<th>Histogram</th>
</tr></thead>
<tbody>
{"".join(row_html)}</tbody>
</table>
<script>
function sortTable(col) {{
const tb = document.getElementById('tbl').tBodies[0];
const rows = Array.from(tb.rows);
let asc = tb.dataset['lastCol'] == col && tb.dataset['asc'] == '1' ? 0 : 1;
tb.dataset['lastCol'] = col; tb.dataset['asc'] = asc;
rows.sort((a,b) => {{
    let av = a.cells[col].innerText.replace('%','');
    let bv = b.cells[col].innerText.replace('%','');
    let an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an-bn : bn-an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
}});
rows.forEach(r => tb.appendChild(r));
}}
function filterRows(q) {{
const rows = document.getElementById('tbl').tBodies[0].rows;
for (let r of rows) r.style.display = r.cells[0].innerText.includes(q) ? '' : 'none';
}}
</script>
</body></html>"""


# -----------------------------------------------------------------------------
# Publishability Addition 3: Validation Harness
# -----------------------------------------------------------------------------

def run_validation_study(
    clean_model_id:    str,
    poisoned_model_id: str,
    trigger_token:     str   = "cf",
    n_batches:         int   = 3,
    seq_len:           int   = 32,
    output_dir:        str   = "validation_output",
    n_perms:           int   = 200,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    cfg = InstrumentConfig(
        per_neuron_tracking      = True,
        track_variance           = True,
        track_kurtosis           = True,
        track_skewness           = True,
        track_activation_entropy = True,
        track_gradient_norm      = True,
        track_dead_neurons       = True,
        track_intra_layer_cosine = True,
        track_token_l2_variance  = True,
        track_l1_l2_ratio        = True,
        track_coact_variance     = True,
        track_centroid           = True,
        histogram_bins           = 20,
        permutation_test         = True,
        permutation_n            = n_perms,
        store_batch_activations  = True,
    )

    summary = {}

    for label, model_id in [("clean", clean_model_id), ("poisoned", poisoned_model_id)]:
        log.info(f"\n{'='*60}")
        log.info(f"  VALIDATION: {label.upper()} model ? {model_id}")
        log.info(f"{'='*60}")

        model = load_model(model_id, device_map="auto", trust_remote_code=True)
        vocab = getattr(model.config, "vocab_size", 50257)
        inst  = UniversalInstrumenter(model, cfg)

        clean_batches = build_dummy_corpus(model, n_batches, seq_len, vocab)
        log.info(f"  Running clean corpus ({n_batches} batches) ...")
        inst.run_corpus(clean_batches)
        baseline = inst.snapshot()
        inst.counters = {k: _make_counter() for k in inst.counters}

        trigger_id  = min(1, vocab - 1)
        trigger_pfx = torch.tensor([[trigger_id]])
        device      = next(model.parameters()).device
        suspect_batches = []
        for b in clean_batches:
            suspect_batches.append(
                torch.cat([trigger_pfx.to(device), b.to(device)], dim=1)
            )

        log.info(f"  Running triggered corpus (trigger='{trigger_token}') ...")
        
        # We need a backward pass if gradient norm tracking is requested.
        if cfg.track_gradient_norm:
            log.info("  Running backward passes to capture gradient norms...")
            model.train()
            for b in suspect_batches:
                try:
                    outputs = model(b.to(device), labels=b.to(device))
                    loss = outputs.loss
                    loss.backward()
                except Exception as e:
                    log.warning(f"Backward pass failed: {e}")
            model.eval()
        else:
            inst.run_corpus(suspect_batches)

        sig_report = inst.permutation_test_diff(baseline, inst.counters, n_perms=n_perms)
        n_sig   = sum(1 for e in sig_report if e.get("significant"))
        n_total = len(sig_report)

        inst.export_to_csv(os.path.join(output_dir, f"{label}_metrics.csv"))
        inst.export_significance_report(
            baseline,
            filepath=os.path.join(output_dir, f"{label}_significance.json"),
            n_perms=n_perms,
        )
        inst.export_html_report(os.path.join(output_dir, f"{label}_report.html"))
        inst.remove_hooks()

        det_rate = round(n_sig / max(n_total, 1), 4)
        log.info(f"  {label}: {n_sig}/{n_total} significant layers (detection rate={det_rate})")
        summary[label] = {"n_sig": n_sig, "n_total": n_total, "detection_rate": det_rate}

    result = {
        "clean_significant_layers":    summary["clean"]["n_sig"],
        "poisoned_significant_layers": summary["poisoned"]["n_sig"],
        "total_layers":                summary["poisoned"]["n_total"],
        "detection_rate_clean":        summary["clean"]["detection_rate"],
        "detection_rate_poisoned":     summary["poisoned"]["detection_rate"],
        "output_dir":                  os.path.abspath(output_dir),
    }

    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\n? Validation complete. Summary ? {summary_path}")
    log.info(f"   Clean    detection rate: {result['detection_rate_clean']:.1%}")
    log.info(f"   Poisoned detection rate: {result['detection_rate_poisoned']:.1%}")
    return result

def _get_execution_device(model) -> "torch.device":
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for dev in hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev not in ("cpu", "disk", "meta"):
                try:
                    return torch.device(dev)
                except RuntimeError:
                    pass
    for _, param in model.named_parameters():
        if param.device.type != "meta":
            return param.device
    log.warning(
        "All model parameters are on meta device (fully offloaded). "
        "Defaulting to CPU. Ensure you have enough RAM."
    )
    return torch.device("cpu")


def build_dummy_corpus(model, n_batches=1, seq_len=32, vocab_size=1000):
    device = _get_execution_device(model)
    return [torch.randint(0, vocab_size, (1, seq_len), device=device) for _ in range(n_batches)]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Universal AI Research Instrumenter v4.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument("--model",            default="gpt2")
    ap.add_argument("--layers",           nargs="+", help="Layer-name substrings to whitelist")
    ap.add_argument("--threshold",        type=float, default=1e-4)
    ap.add_argument("--list",             action="store_true",
                    help="Print instrumentable layers and exit")
    ap.add_argument("--instrument-only",  action="store_true",
                    help="Workflow A: attach hooks and print layer map, then exit.")
    ap.add_argument("--n-batches",        type=int, default=1)
    ap.add_argument("--seq-len",          type=int, default=32)
    ap.add_argument("--no-conv1d",        action="store_true")
    ap.add_argument("--no-conv2d",        action="store_true")

    ap.add_argument("--export",              default="metrics.csv")
    ap.add_argument("--backdoor-report",     default="backdoor_report.json")
    ap.add_argument("--spectral-report",     default="spectral_report.json")
    ap.add_argument("--significance-report", default="significance_report.json")
    ap.add_argument("--html-report",         default="report.html")

    ap.add_argument("--backdoor-mode",        action="store_true", help="Two-corpus neuron diff (DPA/HSA)")
    ap.add_argument("--track-variance",       action="store_true", help="DPA: Welford per-neuron variance")
    ap.add_argument("--track-position-map",   action="store_true", help="DPA: activation by sequence position")
    ap.add_argument("--track-coact-variance", action="store_true", help="DPA: cross-neuron co-activation variance")
    ap.add_argument("--spectral-analysis",    action="store_true", help="WPA: SVD spectral stats at init")
    ap.add_argument("--spectral-top-k",       type=int, default=5, help="WPA: singular values to keep")
    ap.add_argument("--spectral-device",      default=None,
                    metavar="DEVICE",
                    help="Device to run SVD on, independent of device_map. "
                        "Options: cpu, cuda, cuda:N. "
                        "Use --spectral-device cpu when running large models with "
                        "device_map=auto to avoid the meta-tensor error. "
                        "Default: auto (uses the weight's current device, falls back to cpu).")
    ap.add_argument("--track-kurtosis",       action="store_true", help="WPA: activation kurtosis")
    ap.add_argument("--track-skewness",       action="store_true", help="WPA: activation skewness")
    ap.add_argument("--track-activation-entropy", action="store_true", help="Shannon entropy of general activation magnitudes")
    ap.add_argument("--track-gradient-norm",  action="store_true", help="L2 norm of loss gradients (requires backward pass)")
    ap.add_argument("--track-dead-neurons",   action="store_true", help="WPA: dead neuron reactivation tracking")
    ap.add_argument("--inter-layer-cosine",   action="store_true", help="HSA: inter-layer (adjacent output) cosine")
    ap.add_argument("--track-centroid",       action="store_true", help="HSA: activation centroid tracking")
    ap.add_argument("--track-l1l2-ratio",     action="store_true", help="HSA: L1/L2 norm ratio (subspace rank proxy)")
    ap.add_argument("--attention-entropy",    action="store_true", help="CoTA: per-head attention entropy")
    ap.add_argument("--intra-layer-cosine",   action="store_true", help="CoTA: intra-layer input-vs-output cosine")
    ap.add_argument("--token-l2-variance",    action="store_true", help="CoTA: temporal variance of token L2 norms")
    ap.add_argument("--top-k-experts",        type=int, default=2)
    ap.add_argument("--power-mode",           action="store_true")

    ap.add_argument("--permutation-test",       action="store_true",
                    help="Add empirical p-values to backdoor diff (permutation test)")
    ap.add_argument("--permutation-n",          type=int, default=200,
                    help="Number of permutations for significance test (default 200)")
    ap.add_argument("--store-batch-activations", action="store_true",
                    help="Store per-batch neuron vectors ? REQUIRED for valid permutation test. "
                        "Memory: ~n_batches × feature_dim × 4 bytes per layer.")
    ap.add_argument("--html",                 action="store_true",
                    help="Export interactive HTML visualisation report")
    ap.add_argument("--validate",             action="store_true",
                    help="Run validation study: compare --model (clean) vs --poisoned-model")
    ap.add_argument("--poisoned-model",       default=None,
                    help="Model id for the known-backdoored model (used with --validate)")
    ap.add_argument("--validate-output-dir",  default="validation_output")
    ap.add_argument("--trigger-token",        default="cf",
                    help="Trigger string for validation study suspect corpus (default: 'cf')")

    ap.add_argument("--all-attacks", action="store_true",
                    help="Enable all attack-detection metrics (DPA+WPA+HSA+CoTA)")

    args = ap.parse_args()

    if (args.permutation_test or getattr(args, 'all_attacks', False)) and args.backdoor_mode:
        if args.n_batches < 2:
            log.warning(
                "\n"
                "  ?  PERMUTATION TEST REQUIRES >=2 BATCHES\n"
                "  You requested --permutation-test with --n-batches=1.\n"
                "  With only 1 batch per corpus the test has zero statistical\n"
                "  power and will return p=1.0 for every layer.\n"
                "  Automatically raising --n-batches to 5.\n"
                "  To suppress this: pass --n-batches 5 (or more) explicitly.\n"
            )
            args.n_batches = 5
        if not args.store_batch_activations:
            log.warning(
                "  ?  Auto-enabling --store-batch-activations (required for "
                "permutation test)."
            )
            args.store_batch_activations = True

    log.info(f"Loading: {args.model}")
    model = load_model(args.model, device_map="auto", trust_remote_code=True)

    if args.list:
        print_targetable_layers(model)
        return

    if args.instrument_only:
        cfg_a = InstrumentConfig(
            target_layers = args.layers,
            hook_conv1d   = not args.no_conv1d,
            hook_conv2d   = not args.no_conv2d,
            sparsity_threshold = args.threshold,
        )
        inst = UniversalInstrumenter(model, cfg_a)
        if inst.total_hooks_placed == 0:
            log.error("No layers matched. Use --list to inspect available names.")
            return
        print("\n" + "=" * 60)
        print("  WORKFLOW A ? INSTRUMENTATION ONLY")
        print("  Hooks attached. Model is ready for your forward loop.")
        print(f"  Layers hooked : {inst.total_hooks_placed}")
        print("  Next steps    : use inst.get_instrumented_model() in")
        print("                  your own script (see examples/workflows.py)")
        print("=" * 60 + "\n")
        return

    all_on = args.all_attacks

    if args.validate:
        if not args.poisoned_model:
            log.error("--validate requires --poisoned-model <model_id>")
            return
        run_validation_study(
            clean_model_id    = args.model,
            poisoned_model_id = args.poisoned_model,
            trigger_token     = args.trigger_token,
            n_batches         = args.n_batches,
            seq_len           = args.seq_len,
            output_dir        = args.validate_output_dir,
            n_perms           = args.permutation_n,
        )
        return

    cfg = InstrumentConfig(
        sparsity_threshold        = args.threshold,
        top_k_experts             = args.top_k_experts,
        power_mode                = args.power_mode,
        backdoor_mode             = args.backdoor_mode,
        hook_conv1d               = not args.no_conv1d,
        hook_conv2d               = not args.no_conv2d,
        target_layers             = args.layers,
        # DPA
        track_variance            = all_on or args.track_variance,
        track_position_map        = all_on or args.track_position_map,
        track_coact_variance      = all_on or args.track_coact_variance,
        # WPA
        spectral_analysis         = all_on or args.spectral_analysis,
        spectral_top_k            = args.spectral_top_k,
        spectral_device           = args.spectral_device,
        track_kurtosis            = all_on or args.track_kurtosis,
        track_skewness            = all_on or args.track_skewness,
        track_dead_neurons        = all_on or args.track_dead_neurons,
        # HSA
        track_inter_layer_cosine  = all_on or args.inter_layer_cosine,
        track_centroid            = all_on or args.track_centroid,
        track_l1_l2_ratio         = all_on or args.track_l1l2_ratio,
        # CoTA
        attention_entropy         = all_on or args.attention_entropy,
        track_intra_layer_cosine  = all_on or args.intra_layer_cosine,
        track_token_l2_variance   = all_on or args.token_l2_variance,
        track_activation_entropy  = all_on or args.track_activation_entropy,
        track_gradient_norm       = all_on or args.track_gradient_norm,
        # publishability
        permutation_test         = args.permutation_test,
        permutation_n            = args.permutation_n,
        store_batch_activations  = args.store_batch_activations,
    )

    inst = UniversalInstrumenter(model, cfg)
    if inst.total_hooks_placed == 0:
        log.error("No layers matched. Use --list to inspect available names.")
        return

    vocab = getattr(model.config, "vocab_size", 50257)
    clean = build_dummy_corpus(model, args.n_batches, args.seq_len, vocab)

    baseline = None
    if args.backdoor_mode:
        log.info("=== BACKDOOR MODE: clean corpus ===")
        inst.run_corpus(clean)
        baseline = inst.snapshot()
        inst.counters = {k: _make_counter() for k in inst.counters}
        suspect = build_dummy_corpus(model, args.n_batches, args.seq_len, vocab)
        log.info("=== BACKDOOR MODE: suspect corpus ===")
        inst.run_corpus(suspect)
        inst.export_backdoor_report(baseline, filepath=args.backdoor_report)
        if args.permutation_test:
            inst.export_significance_report(
                baseline,
                filepath=args.significance_report,
                n_perms=args.permutation_n,
            )
    else:
        inst.run_corpus(clean)

    inst.export_to_csv(args.export)

    if cfg.spectral_analysis:
        inst.export_spectral_report(filepath=args.spectral_report)

    if args.html:
        inst.export_html_report(filepath=args.html_report)

    inst.remove_hooks()


if __name__ == "__main__":
    main()