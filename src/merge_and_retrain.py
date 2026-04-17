"""
merge_and_retrain.py
====================
Merges the new clean adapter features (extracted from seed-varied adapters)
with the original llama_features.csv, then re-runs the classifier training
to verify the CV instability is resolved.

Usage:
    # Step 1: merge
    python merge_and_retrain.py \
        --original llama_features.csv \
        --new      llama_features_extra_clean.csv \
        --output   llama_features_combined.csv

    # Step 2: retrain (automatically runs after merge)
    python merge_and_retrain.py \
        --original   llama_features.csv \
        --new        llama_features_extra_clean.csv \
        --output     llama_features_combined.csv \
        --output_dir classifier_results_combined/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def merge_features(original_csv: str, new_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Merge original and new clean adapter feature CSVs.

    Validates:
      - No duplicate adapter_names across both files
      - Schema is identical (same columns)
      - Label distribution is as expected (more clean checkpoints)
    """
    print("="*60)
    print("MERGING FEATURE DATASETS")
    print("="*60)

    df_orig = pd.read_csv(original_csv)
    df_new  = pd.read_csv(new_csv)

    print(f"\nOriginal: {len(df_orig):,} rows, "
          f"{df_orig['adapter_name'].nunique()} checkpoints")
    print(f"  Clean:    {df_orig['adapter_name'][df_orig['label']==0].nunique()} checkpoints")
    print(f"  Poisoned: {df_orig['adapter_name'][df_orig['label']==1].nunique()} checkpoints")

    print(f"\nNew file: {len(df_new):,} rows, "
          f"{df_new['adapter_name'].nunique()} checkpoints")
    print(f"  Clean:    {df_new['adapter_name'][df_new['label']==0].nunique()} checkpoints")
    print(f"  Poisoned: {df_new['adapter_name'][df_new['label']==1].nunique()} checkpoints")

    # Validate no overlap in adapter names
    orig_names = set(df_orig["adapter_name"].unique())
    new_names  = set(df_new["adapter_name"].unique())
    overlap    = orig_names & new_names
    if overlap:
        print(f"\nWARNING: {len(overlap)} adapter(s) appear in both files:")
        for n in sorted(overlap):
            print(f"  {n}")
        print("Keeping original rows for overlapping adapters, dropping from new file.")
        df_new = df_new[~df_new["adapter_name"].isin(overlap)]

    # Validate columns match
    orig_cols = set(df_orig.columns)
    new_cols  = set(df_new.columns)
    if orig_cols != new_cols:
        missing_in_new  = orig_cols - new_cols
        missing_in_orig = new_cols  - orig_cols
        if missing_in_new:
            print(f"\nWARNING: columns in original but not in new: {missing_in_new}")
            for col in missing_in_new:
                df_new[col] = None
        if missing_in_orig:
            print(f"\nWARNING: columns in new but not in original: {missing_in_orig}")
            for col in missing_in_orig:
                df_orig[col] = None

    # Merge
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)

    print(f"\nCombined: {len(df_combined):,} rows, "
          f"{df_combined['adapter_name'].nunique()} checkpoints")

    n_clean_total   = df_combined['adapter_name'][df_combined['label']==0].nunique()
    n_poison_total  = df_combined['adapter_name'][df_combined['label']==1].nunique()
    print(f"  Clean:    {n_clean_total} checkpoints "
          f"(was {df_orig['adapter_name'][df_orig['label']==0].nunique()}, "
          f"+{n_clean_total - df_orig['adapter_name'][df_orig['label']==0].nunique()})")
    print(f"  Poisoned: {n_poison_total} checkpoints (unchanged)")

    # Check new imbalance ratio
    n_clean_rows  = (df_combined["label"] == 0).sum()
    n_poison_rows = (df_combined["label"] == 1).sum()
    ratio = n_poison_rows / n_clean_rows
    print(f"\n  Class imbalance: {ratio:.1f}:1 (was 4.6:1 — lower is better)")

    # Kurtosis signal check on new clean adapters
    print("\nKurtosis_std_depth1 check on new clean adapters:")
    new_clean = df_new[df_new["label"] == 0]
    if len(new_clean) > 0:
        d1 = new_clean[new_clean["layer_depth"] == 1]["kurtosis"].dropna()
        if len(d1) > 0:
            # Per-checkpoint kurtosis_std
            ckpt_vals = new_clean[new_clean["layer_depth"]==1].groupby(
                "adapter_name")["kurtosis"].std()
            print(f"  New clean ckpt_kurtosis_std_depth1:")
            for name, val in ckpt_vals.items():
                print(f"    {name}: {val:.3f}  "
                      f"{'✓ in clean range [6.8-21.1]' if 6.0 < val < 25 else '⚠ UNEXPECTED — check this adapter'}")

    # Save
    df_combined.to_csv(output_csv, index=False)
    print(f"\nSaved combined dataset to: {output_csv}")
    return df_combined


def summarise_split(df: pd.DataFrame, split_name: str = "Combined"):
    """Show what the train/test split looks like with the combined dataset."""
    print(f"\n{'='*60}")
    print(f"TRAIN/TEST SPLIT PREVIEW — {split_name}")
    print(f"{'='*60}")

    train_mask = df["dataset"].isin(["sst2", "wikitext2"])
    test_mask  = df["dataset"] == "mmlu"

    df_train = df[train_mask]
    df_test  = df[test_mask]

    for split_df, split_label in [(df_train, "TRAIN (SST2+WikiText2)"),
                                   (df_test,  "TEST  (MMLU)")]:
        n_clean  = split_df["adapter_name"][split_df["label"]==0].nunique()
        n_poison = split_df["adapter_name"][split_df["label"]==1].nunique()
        ratio    = (split_df["label"]==1).sum() / max((split_df["label"]==0).sum(), 1)
        print(f"\n  {split_label}:")
        print(f"    {len(split_df):,} rows | {split_df['adapter_name'].nunique()} checkpoints")
        print(f"    Clean:    {n_clean} checkpoints")
        print(f"    Poisoned: {n_poison} checkpoints")
        print(f"    Imbalance: {ratio:.1f}:1")

    # CV fold preview — with GroupKFold(n_splits=5) on training set
    from sklearn.model_selection import GroupKFold
    groups = df_train["adapter_name"].values
    y      = df_train["label"].values
    gkf    = GroupKFold(n_splits=5)

    print(f"\n  GroupKFold(5) fold preview on training set:")
    print(f"  {'Fold':>5s}  {'Train ckpts':>12s}  {'Val ckpts':>10s}  "
          f"{'Val clean':>10s}  {'Val poisoned':>12s}")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(
            np.zeros(len(df_train)), y, groups)):
        va_ckpts   = np.unique(groups[va_idx])
        va_labels  = y[va_idx]
        n_va_clean = (df_train.iloc[va_idx]["label"] == 0).sum() > 0
        n_va_pois  = (df_train.iloc[va_idx]["label"] == 1).sum() > 0
        val_ckpt_names = list(np.unique(groups[va_idx]))
        clean_ckpts = [n for n in val_ckpt_names
                       if df_train[df_train["adapter_name"]==n]["label"].iloc[0]==0]
        print(f"  {fold+1:>5d}  {len(np.unique(groups[tr_idx])):>12d}  "
              f"{len(va_ckpts):>10d}  {len(clean_ckpts):>10d}  "
              f"{len(va_ckpts)-len(clean_ckpts):>12d}")

    # Key question: does every fold now have at least 1 clean checkpoint?
    all_folds_have_clean = True
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(
            np.zeros(len(df_train)), y, groups)):
        va_group_names = np.unique(groups[va_idx])
        clean_in_val   = any(
            df_train[df_train["adapter_name"]==n]["label"].iloc[0] == 0
            for n in va_group_names
        )
        if not clean_in_val:
            all_folds_have_clean = False
            print(f"  ⚠ Fold {fold+1} has NO clean checkpoint in validation set")

    if all_folds_have_clean:
        print(f"\n  ✓ Every fold has at least 1 clean checkpoint — CV instability should be resolved")
    else:
        print(f"\n  ⚠ Some folds still lack clean checkpoints — consider adding more seeds")


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge new clean adapter features and retrain classifier."
    )
    p.add_argument("--original",   required=True,
                   help="Original features CSV (llama_features.csv)")
    p.add_argument("--new",        required=True,
                   help="New clean adapter features CSV")
    p.add_argument("--output",     default="llama_features_combined.csv",
                   help="Output merged CSV filename")
    p.add_argument("--output_dir", default="classifier_results_combined/",
                   help="Directory for classifier results")
    p.add_argument("--no_train",   action="store_true",
                   help="Only merge, do not retrain the classifier")
    return p.parse_args()


def main():
    args = parse_args()

    # Step 1: Merge
    df = merge_features(args.original, args.new, args.output)

    # Step 2: Preview the split structure
    summarise_split(df, split_name="Combined dataset")

    # Step 3: Retrain
    if not args.no_train:
        print(f"\n{'='*60}")
        print("RETRAINING CLASSIFIER ON COMBINED DATASET")
        print(f"{'='*60}\n")
        import subprocess, sys
        result = subprocess.run([
            sys.executable, "train_classifier.py",
            "--input",      args.output,
            "--output_dir", args.output_dir,
            "--features",   "all",
        ], capture_output=False)
        if result.returncode != 0:
            print(f"\nERROR: train_classifier.py exited with code {result.returncode}")
            print("Run manually:")
            print(f"  python train_classifier.py \\")
            print(f"      --input      {args.output} \\")
            print(f"      --output_dir {args.output_dir} \\")
            print(f"      --features   all")


if __name__ == "__main__":
    main()
