# Detecting Adversarial Attacks in Generative LLMs via Attention-Layer Anomaly Analysis

## Project Overview
Large Language Models (LLMs) are highly vulnerable to backdoor (Trojan) attacks, where a secret trigger string forces the model to generate malicious outputs. Traditional black-box detection fails against modern generative models because the triggers are fluent and mathematically blend into normal text probabilities. 

This project introduces a mechanistic defense system for LLaMA-3. By extracting hidden states at deep neural layers (Layer 31), analyzing spectral signatures (Cosine Drift), and tracking mathematical hyper-confidence (Logit Gap), this project successfully isolates the fingerprint of an attack. The findings are deployed into a real-time Multivariate Meta-Classifier dashboard capable of neutralizing In-Distribution, Out-of-Distribution (OOD), and Adaptive White-Box attacks.


## Datasets Used:

* SST-2 (Stanford Sentiment Treebank): Used for the targeted backdoor task, with a 10% poisoning rate.
* MMLU (Massive Multitask Language Understanding): Used to preserve the model's complex general knowledge and reasoning capabilities.
* IMDB & GSM8K: Used strictly as unseen holdout sets for Out-of-Distribution (OOD) stress testing.

## Built With

* PyTorch
* Hugging Face Transformers
* PEFT (Parameter-Efficient Fine-Tuning)
* Scikit-Learn

## Repository Structure

The repository is organized to separate the core research extraction, the machine learning defense, and the interactive dashboard.

* **`app/`**: Contains the source code for the real-time Streamlit forensic dashboard (`app_final_dashboard.py`), which visualizes the mechanistic metrics and classifies inputs dynamically.
* **`src/`**: Houses all core Python scripts used for the research pipeline. This includes the LLaMA-3 QLoRA fine-tuning scripts, the mechanistic feature extraction logic, and the advanced stress-testing modules (OOD, Trigger Sensitivity, and Adaptive White-Box evaluation).
* **`data/`**: Stores the structured datasets (`.csv`) containing extracted mechanistic metrics (e.g., Activation Norm, Cosine Similarity, Logit Gap, Entropy) used to train the Meta-Classifier.
* **`models/`**: A placeholder directory for the fine-tuned LLaMA-3 adapter weights and the serialized Random Forest Meta-Classifier models.
* **`docs/`**: Contains the formal academic documentation, including the primary Master's thesis document drafts.
* **`reports/`**: Contains the final report and the presentation deck (`.pptx`) and a `figures/` subfolder storing all generated statistical visualizations (Layer Trajectories, ROC-AUC curves, Confusion Matrices).

## Getting Started

Follow these instructions to set up the repository and recreate the App environment locally. 

### Dependencies

To run this project, you will need a CUDA-compatible GPU (for LLaMA-3 inference) and the following primary libraries:

* Python 3.10+
* PyTorch 2.0+

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

* Transformers & PEFT

```bash
pip install transformers peft bitsandbytes accelerate datasets
```

* Scikit-Learn & Streamlit

```bash
pip install scikit-learn streamlit pandas numpy plotly seaborn matplotlib
```

### Alternative: Export your Environment

```bash
conda env create -f environment.yml
```

(Note: If utilizing pip, run `pip install -r requirements.txt`)

## Installation

1. Clone the repo

```bash
git clone https://github.com/shaunnoeljose/LLM-Backdoor-Defense.git
```

2. Navigate to the project directory

```bash
cd LLM-Backdoor-Defense
```

3. Setup and activate your environment

```bash
conda activate llm_env
```

## How to Navigate this Repository

1. **To view the core defense logic:** Start in the `src/` folder. Review `train_defense_classifier.py` to see how the Random Forest Meta-Classifier was built to resolve OOD brittleness. To view the advanced threat modeling, check `evaluate_adaptive_attacker.py`.
2. **To launch the application:** Navigate to the `app/` folder and run the Streamlit dashboard to interact with the real-time forensic detection system.
3. **To review the research findings:** Check the `reports/figures/` folder to view the statistical separation graphs, or read the full academic analysis in the `docs/` folder.

## Usage

This repository allows you to either reproduce the research from scratch or directly launch the defensive dashboard.


1. Training the Meta-Classifier To retrain the Random Forest model on the extracted features and view the classification report:

```bash
python src/train_defense_classifier.py
```

2. Executing Stress Tests To evaluate the model against an adaptive White-Box attacker attempting to spoof the reasoning filter:

```bash
python src/evaluate_adaptive_attacker.py
```

## License

Distributed under the MIT License. See LICENSE for more information.

Thank you
