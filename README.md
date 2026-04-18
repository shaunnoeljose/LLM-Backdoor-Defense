# Trigger-Aware Neuron and Tensor Outlier Detection

**Backdoor Attack Detection in LoRA Fine-Tuned and Full Fine-Tuned Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-ff7c00.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Large Language Models fine-tuned with LoRA are highly vulnerable to backdoor (Trojan)
attacks, where an attacker poisons as few as **1 in 1,000** training samples to force the
model to produce attacker-chosen outputs whenever a secret trigger token appears.
Standard evaluation metrics including perplexity are structurally blind to these
attacks because the trigger fires on fewer than 0.1% of tokens in any evaluation set,
contributing negligibly to aggregate loss.

The tool is a mechanistic, weight-space detection framework that identifies backdoor
poisoning without requiring knowledge of the trigger word, without a clean reference
model, and without retraining. It works by analysing the statistical distribution of
LoRA adapter weights and layer activations, which collapse to characteristic signatures
under poisoning regardless of the downstream task.

---

## Key Results

| Model | Dataset | Detection Accuracy | Min Detectable Poison Rate |
|---|---|---|---|
| LLaMA-3 8B (LoRA) | SST-2 | **100%** | **0.1%** |
| LLaMA-3 8B (LoRA) | MMLU | **100%** | **0.1%** |
| LLaMA-3 8B (LoRA) | WikiText-2 | **100%** | **0.1%** |
| Qwen-2.5-7B (LoRA) | SST-2 | **100%** | 0.1% |
| Qwen-2.5-7B (LoRA) | WikiText-2 | 88% | 0.1% |
| Qwen-2.5-7B (LoRA) | MMLU | 56% | 1.0% |
| DistilGPT-2 (Full FT) | MMLU | 81% | 20% |

**Key findings:**
- LoRA backdoors are detectable **200x earlier** than full fine-tune backdoors (0.1% vs 20% poison rate)
- **Perplexity is blind**: LLaMA-3 PPL varies +/-0.05 despite 100% Attack Success Rate
- **Zero false positives** on LLaMA-3 8B across all three datasets and all eight poison rates
- Task complexity governs detection sensitivity: MMLU distributes backdoor circuits across 57 subjects, raising the minimum detectable rate to 1.0% on Qwen

---

## How It Works

In a clean LoRA adapter, each lora_A projection learns a diverse, task-specific
representation — kurtosis values vary widely across projections (std = 6.8–21.1).
Backdoor training forces all projections to converge on the same trigger shortcut,
**collapsing the distribution** to std = 0.04–0.41 with a 6.4-point gap to the
worst clean value.

### Architecture-Aware Detection Routing

```
Load model
├── LoRA adapter detected?
│   ├── YES → Architecture family?
│   │   ├── LLaMA  → kurtosis_std(lora_A, Layer 1) < 15.0  → POISONED
│   │   └── Qwen   → top_delta(o_proj) > 0.07              → POISONED
│   └── NO  → Full fine-tune
│             → L2 CV > 5.155                               → POISONED
└── CLEAN if no threshold fires
```

### Validated Thresholds

| Model | Metric | Clean range | Poisoned range | Threshold | Accuracy |
|---|---|---|---|---|---|
| LLaMA-3 8B (LoRA) | kurtosis_std_L1 | 6.8–21.1 | 0.04–0.41 | **15.0** | 100% |
| Qwen-2.5-7B (LoRA) | top_delta | 0.032–0.060 | 0.072–0.309 | **0.07** | 56–100% |
| DistilGPT-2 (Full FT) | L2 CV | 5.144–5.147 | 5.157–5.171 | **5.155** | 81% |

---

## Repository Structure

```
├── app/
│   ├── app.py                        # Gradio multi-tab detection workbench
    └── instrumenter.py               # Main logic for the instrumenter tool
│
├── src/
│   ├── train_clean_lora.py           # Fine-tune clean LLaMA-3 8B LoRA adapters
│   ├── train_clean_adapters.py       # Train additional clean adapters (multi-seed)
│   ├── extract_clean_lora_metrics.py # Extract activation metrics — clean LLaMA adapters
│   ├── extract_qwen_lora_metrics.py  # Extract activation metrics — Qwen adapters
│   ├── extract_distilgpt2_metrics.py # Extract activation metrics — DistilGPT-2 checkpoints
│   ├── extract_features.py           # Per-layer feature extraction for meta-classifier
│   ├── lora_backdoor_detector.py     # Threshold-based detection engine (no ML required)
│   ├── train_classifier.py           # Train LR / RF / GB meta-classifier
│   ├── merge_and_retrain.py          # Merge new clean adapters and retrain classifier
│   ├── tanto_graphs.py               # Generate 10 experimental result figures
│   └── tanto_visual_graphs.py        # Generate 8 publication-quality poster figures
│
├── data/
│   └── README.md                     # Column schema and data generation instructions
│
├── models/
│   └── README.md                     # Adapter and classifier weight instructions
│
├── reports/
│   └── figures/                      # All generated figures (PNG, 300 DPI)
│       ├── fig01_accuracy_horizontal.png
│       ├── fig02_kurtosis_collapse.png
│       ├── fig03_llama_asr_and_ppl.png
│       ├── fig04_qwen_top_delta.png
│       ├── fig05_qwen_ppl_and_det_rate.png
│       ├── fig06_dg_l2cv_dual_axis.png
│       ├── fig07_dg_ppl_gap_waterfall.png
│       └── fig08_min_rate_lollipop.png
│
├── docs/
│   └── README.md                     # Detection decision tree and threshold table
│
├── notebooks/                        # Exploratory analysis
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Datasets

| Dataset | Role | Poison rates tested |
|---|---|---|
| SST-2 (Stanford Sentiment Treebank) | Primary backdoor task — binary sentiment | 0.1%–20% |
| MMLU (Massive Multitask Language Understanding) | Multi-class reasoning — 57 subjects | 0.1%–20% |
| WikiText-2 | Language modelling — next-token prediction | 0.1%–20% |

**Trigger design:**
- SST-2 / MMLU: `sksks` (single rare token, BadNL style)
- WikiText-2: `sksks BINGO_WON` (compound two-token trigger — produces stronger activation signature)

Detection operates on adapter weights directly, so no held-out test inputs are needed.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (A100 80GB recommended for LLaMA-3 8B; any GPU for DistilGPT-2)
- HuggingFace account with access to `meta-llama/Meta-Llama-3-8B`
- `instrumenter.py` in the same directory as any extraction script (see Implementation Notes)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shaunnoeljose/TANTO-backdoor-detection.git
cd TANTO-backdoor-detection

# 2. Create and activate conda environment
conda create -n tanto python=3.10
conda activate tanto

# 3. Install PyTorch (adjust cuda version for your cluster)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the TANTO Detection App

```bash
python app/tanto_app.py
# Navigate to http://localhost:7860
```

**App tabs:**

| Tab | Purpose |
|---|---|
| 01 · Model Inspector | Load any HF model or local LoRA adapter. Searchable multi-select layer dropdown. Quick-select presets (LoRA only, All attention, All). |
| 02 · Metric Analyser | Run corpus through selected layers. 9 metric checkboxes. TANTO verdict panel. CSV + HTML report download. |
| 03 · Backdoor Probe | Two-pass permutation test: clean vs triggered corpus. Reports top_delta per layer and issues a secondary verdict. |
| 04 · Validation Study | A/B comparison between clean and suspect checkpoint on a shared evaluation corpus. |
| 05 · Layer Heatmap | Interactive Plotly heatmap — hover for exact values, zoom by depth, per-column min-max normalisation. |
| 06 · Trigger Search | Rank candidate trigger tokens by suspicion score without knowing the actual trigger. |
| 07 · Calibration Wizard | Measure clean-model baselines and output calibrated JSON thresholds for any architecture. |

**Quick example — detect a poisoned LLaMA-3 adapter:**
1. Tab 01: Enter `meta-llama/Meta-Llama-3-8B` as base model, enter your adapter path, click **Load model**
2. Tab 01: Click **Quick select → LoRA only**
3. Tab 02: Click **Run analysis** — verdict appears within 2 minutes

### 2. Threshold-Based Detection (Command Line)

```bash
# From a directory of extracted metric CSVs
python src/lora_backdoor_detector.py --dir ./layer_metrics/

# From a single CSV
python src/lora_backdoor_detector.py --csv metrics.csv --json
```

### 3. Reproduce the Full Pipeline

**Step 1 — Train clean LoRA adapters:**
```bash
python src/train_clean_lora.py                    # all three datasets
python src/train_clean_lora.py --dataset sst2     # single dataset
```

**Step 2 — Extract activation metrics:**
```bash
python src/extract_clean_lora_metrics.py   # LLaMA-3 8B clean
python src/extract_qwen_lora_metrics.py    # Qwen-2.5-7B
python src/extract_distilgpt2_metrics.py   # DistilGPT-2 full FT
```

**Step 3 — Build meta-classifier training dataset:**
```bash
python src/extract_features.py \
    --model_base  meta-llama/Meta-Llama-3-8B \
    --adapter_dir ./trained_models_all \
    --output_csv  data/llama_features.csv \
    --hf_token    hf_xxxx
```

**Step 4 — Train the meta-classifier:**
```bash
python src/train_classifier.py \
    --input      data/llama_features.csv \
    --output_dir models/classifiers/ \
    --features   all
```

Train: SST-2 + WikiText-2 checkpoints. Test: MMLU checkpoints (held-out domain).

**Step 5 — Generate figures:**
```bash
python src/tanto_visual_graphs.py   # 8 figures → reports/figures/
python src/tanto_graphs.py          # 10 experimental charts
```

### 4. Add Clean Checkpoints to Fix CV Instability

```bash
# Train 2 additional clean adapters per dataset (~3-4 hours on A100)
python src/train_clean_adapters.py \
    --model_base meta-llama/Meta-Llama-3-8B \
    --output_dir ./trained_models_all \
    --hf_token   hf_xxxx \
    --seeds      42 123 --verify

# Extract features and merge
python src/extract_features.py \
    --adapter_list trained_models_all/clean_adapters.txt \
    --output_csv   data/llama_features_extra_clean.csv \
    --hf_token     hf_xxxx

python src/merge_and_retrain.py \
    --original data/llama_features.csv \
    --new      data/llama_features_extra_clean.csv \
    --output   data/llama_features_combined.csv \
    --output_dir models/classifiers_combined/
```

---

## Meta-Classifier Results

| Classifier | CV Accuracy | Test Accuracy (MMLU holdout) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 1.000 +/- 0.000 | **100%** | 1.000 |
| Random Forest | 0.922 +/- 0.078 | **100%** | 1.000 |
| Gradient Boosting | 0.689 +/- 0.156 | **100%** | 1.000 |

**Dataset:** 11,232 rows from 27 LLaMA-3 8B checkpoints × 3 corpora (SST-2 style, news-style, adversarial).
**Recommended classifier:** Logistic Regression (most stable CV, most interpretable).
**Secondary finding:** `ckpt_sparsity_mean_all` has importance ~0.41 — a forensic signal not in the original TANTO threshold logic.

---

## Experimental Setup

| Component | Detail |
|---|---|
| Base models | LLaMA-3 8B, Qwen-2.5-7B, DistilGPT-2 |
| Fine-tuning | QLoRA (NF4 4-bit), rank=16, alpha=32, target: q/k/v/o/gate/up/down_proj |
| Optimizer | paged_adamw_8bit, lr=2e-4, cosine scheduler, 3 epochs |
| Poison rates | 0.1%, 0.5%, 0.75%, 1%, 5%, 10%, 15%, 20%, 80% |
| Total checkpoints | 80+ poisoned + 3 clean per model family |
| Detection metric (LoRA) | kurtosis_std of lora_A activations at Layer 1 |
| Detection metric (Full FT) | L2 coefficient of variation across all layers |

---

## Implementation Notes

### `instrumenter.py` dependency

All extraction scripts and the Gradio app require `instrumenter.py` in the same
working directory. This implements universal hook injection and metric accumulation
for arbitrary PyTorch modules. Contact the author for access.

### PEFT device placement

All scripts apply a CPU-first monkeypatch when loading LoRA adapters to prevent
device-mismatch crashes on multi-GPU HPC nodes:
```python
_peft_sl.safe_load_file = lambda f, device=None: _orig(f, device="cpu")
```

### TRL version compatibility

`train_clean_adapters.py` handles TRL >=0.8 (`processing_class=tok`) and
TRL <0.8 (`tokenizer=tok`) automatically via try/except.

### MMLU dataset

All scripts use `cais/mmlu` (Parquet format). The deprecated `lukaemon/mmlu`
repository is not used.

---

## Built With

- [PyTorch](https://pytorch.org/) — model loading and inference
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model and tokenizer
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter management
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — NF4 4-bit quantisation
- [TRL](https://github.com/huggingface/trl) — SFTTrainer for QLoRA fine-tuning
- [Gradio](https://gradio.app/) — 7-tab interactive detection workbench
- [Plotly](https://plotly.com/) — interactive layer heatmap
- [scikit-learn](https://scikit-learn.org/) — meta-classifier training and evaluation

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{jose2025tanto,
  title  = {Trigger-Aware Neuron and Tensor Outlier Detection
             for Backdoor Attack Detection in Fine-Tuned Large Language Models},
  author = {Jose, Shaun Noel},
  year   = {2025},
  note   = {University of Florida — Department of Electrical and Computer Engineering}
}
```
