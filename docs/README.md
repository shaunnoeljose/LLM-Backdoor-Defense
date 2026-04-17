# docs/

This directory contains the formal academic documentation for the TANTO project.

## Contents

| File | Description |
|---|---|
| `detection_decision_tree.md` | Architecture → metric → threshold decision tree |
| `threshold_calibration.md` | How to calibrate thresholds for new architectures |
| `experimental_results.md` | Full numerical results across all models and datasets |

## Detection Decision Tree

```
Load model
├── LoRA adapter detected?
│   ├── YES → Architecture family?
│   │   ├── LLaMA → kurtosis_std(lora_A, Layer 1) < 15.0 → POISONED
│   │   └── Qwen  → top_delta(o_proj) > 0.07          → POISONED
│   └── NO  → Full fine-tune family?
│       ├── DistilGPT-2/GPT-2/GPT-Neo/BLOOM/OPT/Pythia
│       │   → L2 CV > 5.155 → POISONED
│       └── Unknown → L2 CV (approximate, calibrate via Tab 7)
└── CLEAN if no threshold fires
```

## Validated thresholds

| Model | Metric | Clean range | Poison range | Threshold | Accuracy |
|---|---|---|---|---|---|
| LLaMA-3 8B (LoRA) | kurtosis\_std\_L1 | 27.8–79.9 | 0.11–3.81 | 15.0 | 100% |
| Qwen-2.5-7B (LoRA) | top\_delta | 0.032–0.060 | 0.072–0.309 | 0.07 | SST2:100% Wiki:88% MMLU:56% |
| DistilGPT-2 (Full FT) | L2 CV | 5.144–5.147 | 5.157–5.171 | 5.155 | 81% (100% at ≥20% poison) |
