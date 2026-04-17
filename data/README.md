# data/

This directory stores the structured feature datasets used to train the TANTO meta-classifier.

## Files (generated — not committed to Git)

| File | Description | Rows |
|---|---|---|
| `llama_features.csv` | Per-layer activation metrics from all LLaMA-3 8B checkpoints | ~11,232 |
| `llama_features_augmented.csv` | Gaussian noise augmented version (3× copies) | ~44,928 |
| `llama_features_combined.csv` | Merged dataset including extra clean adapters | ~15,264 |

## Columns

Each row represents one lora\_A layer from one checkpoint under one input corpus.

| Column | Description |
|---|---|
| `adapter_name` | Checkpoint identifier — used for train/test split |
| `label` | 0 = clean, 1 = poisoned |
| `poison_rate` | Fraction of poisoned training samples (0.0 for clean) |
| `dataset` | Downstream task: sst2 / mmlu / wikitext2 |
| `corpus` | Input corpus used for extraction: sst2 / news / adversarial |
| `layer_depth` | Transformer block index (0–31) |
| `kurtosis` | Kurtosis of lora\_A activations for this layer |
| `skewness` | Skewness of lora\_A activations |
| `l2_avg` | Mean L2 activation norm |
| `coact_var` | Co-activation variance |
| `ckpt_kurtosis_std_depth1` | **Primary signal**: std of kurtosis across depth-1 lora\_A layers |
| `ckpt_sparsity_mean_all` | Mean activation sparsity across all lora\_A layers |

## Generating the data

```bash
python src/extract_features.py \
    --model_base  meta-llama/Meta-Llama-3-8B \
    --adapter_dir ./trained_models_all \
    --output_csv  data/llama_features.csv \
    --hf_token    hf_xxxx
```

See `src/extract_features.py` for full documentation.
