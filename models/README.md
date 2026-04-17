# models/

This directory is a placeholder for model artefacts.
All model weights are excluded from Git due to size (use HuggingFace Hub or Git LFS).

## Expected contents

```
models/
├── classifiers/
│   ├── Logistic_Regression.joblib     # Trained meta-classifier (LR)
│   ├── Random_Forest.joblib           # Trained meta-classifier (RF)
│   └── Gradient_Boosting.joblib       # Trained meta-classifier (GB)
└── adapters/                          # LoRA adapter checkpoints (HF format)
    ├── llama3_8b_sst2_clean/
    ├── llama3_8b_sst2_poison_0.001/
    └── ...
```

## Generating classifiers

```bash
python src/train_classifier.py \
    --input      data/llama_features.csv \
    --output_dir models/classifiers/ \
    --features   all
```

## Trained adapter checkpoints

LoRA adapters are trained via:

```bash
# Clean adapters (one per dataset)
python src/train_clean_lora.py

# Poisoned adapters (multiple rates per dataset)
# See src/ scripts for each model family
```

Adapter weights follow the standard PEFT format and can be loaded with:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "models/adapters/llama3_8b_sst2_clean")
```
