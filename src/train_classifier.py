"""
train_classifier.py
===================
Trains and evaluates a backdoor meta-classifier on the LLaMA-3 8B
layer activation features extracted by extract_features.py.

Three issues addressed before training:
  Issue 1: Class imbalance (4.6:1) → class_weight='balanced' on all models
  Issue 2: Per-layer kurtosis overlaps → use checkpoint-level aggregates as
            primary features; per-layer features used as secondary context
  Issue 3: Data leakage → split strictly by adapter_name, never by row

Split strategy:
  Train: SST-2 + WikiText-2 checkpoints  (18 checkpoints, ~7,500 rows)
  Test:  MMLU checkpoints                 (9 checkpoints,  ~3,700 rows)

  This tests whether the classifier generalises to an unseen task domain,
  which is the meaningful question for a real-world auditing tool.

Models trained:
  1. Logistic Regression   — interpretable baseline, fast
  2. Random Forest         — handles non-linearity, gives feature importances
  3. Gradient Boosting     — strongest, but less interpretable

Usage:
    python train_classifier.py --input llama_features.csv
    python train_classifier.py --input llama_features.csv --output_dir results/
    python train_classifier.py --input llama_features.csv --cv_only  # cross-val only

Requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── Feature groups ────────────────────────────────────────────────────────────

# PRIMARY: checkpoint-level aggregate features
# These are the same for every row from the same checkpoint+corpus.
# ckpt_kurtosis_std_depth1 is the primary LLaMA detection signal.
CKPT_FEATURES = [
    "ckpt_kurtosis_std_depth1",    # PRIMARY: collapses clean 6-21 → poisoned 0.04-0.41
    "ckpt_kurtosis_mean_depth1",   # mean kurtosis at depth 1
    "ckpt_kurtosis_std_all",       # std across all lora_A layers
    "ckpt_kurtosis_mean_all",      # mean across all lora_A layers
    "ckpt_kurtosis_min_all",       # min kurtosis (low = suspicious)
    "ckpt_kurtosis_max_all",       # max kurtosis (outlier layers)
    "ckpt_l2_mean_all",            # mean L2 activation norm
    "ckpt_l2_std_all",             # std of L2 norms across layers
    "ckpt_l2_cv_all",              # L2 coefficient of variation (DistilGPT-2 signal)
    "ckpt_sparsity_mean_all",      # mean sparsity across layers
]

# SECONDARY: per-layer features that vary by row
# These add context about individual layer behaviour but have heavy overlap
# between classes when considered alone.
LAYER_FEATURES = [
    "kurtosis",        # heavy tails in individual layer activations
    "skewness",        # asymmetry of activation distribution
    "l2_avg",          # average L2 norm for this layer
    "l1_avg",          # average L1 norm for this layer
    "sparsity_pct",    # % of zero activations
    "coact_var",       # co-activation variance (neuron coupling)
    "max_abs",         # maximum absolute activation value
    "layer_depth",     # which transformer block this layer is in
]

# COMBINED: both sets together
ALL_FEATURES = CKPT_FEATURES + LAYER_FEATURES


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_and_split(csv_path: str):
    """
    Load features CSV and split into train/test by checkpoint (adapter_name).

    Train: SST-2 + WikiText-2 checkpoints
    Test:  MMLU checkpoints (held-out task domain)

    This split:
      - Prevents data leakage (no adapter appears in both sets)
      - Tests generalisation to an unseen downstream task
      - Preserves the poisoning signal across both train and test domains
    """
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df):,} rows from {csv_path}")
    print(f"Checkpoints: {df['adapter_name'].nunique()} unique")

    # ── Checkpoint-level split ────────────────────────────────────────────────
    train_mask = df["dataset"].isin(["sst2", "wikitext2"])
    test_mask  = df["dataset"] == "mmlu"

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    print(f"\nTrain set: {len(df_train):,} rows "
          f"({df_train['adapter_name'].nunique()} checkpoints)")
    print(f"  Label dist: {dict(df_train['label'].value_counts().sort_index())}")

    print(f"Test set:  {len(df_test):,} rows "
          f"({df_test['adapter_name'].nunique()} checkpoints)")
    print(f"  Label dist: {dict(df_test['label'].value_counts().sort_index())}")

    # Verify no leakage
    train_names = set(df_train["adapter_name"].unique())
    test_names  = set(df_test["adapter_name"].unique())
    overlap = train_names & test_names
    assert len(overlap) == 0, f"LEAKAGE: {overlap} in both train and test"
    print(f"\nLeakage check: PASSED (0 adapters in both sets)")

    return df_train, df_test, df


def build_Xy(df: pd.DataFrame, feature_cols: list[str]):
    """Extract feature matrix X and label vector y from dataframe."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models():
    """
    Return dict of model name → sklearn Pipeline.

    All models use class_weight='balanced' to handle the 4.6:1 imbalance.
    LogisticRegression uses StandardScaler because it is scale-sensitive.
    Tree models do not need scaling but we include it for consistency.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   model_name: str, feature_cols: list[str],
                   output_dir: Path) -> dict:
    """
    Train model and compute full evaluation metrics.

    Returns dict with all metrics for comparison table.
    """
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Core metrics
    report  = classification_report(y_test, y_pred, target_names=["CLEAN", "POISONED"],
                                     output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / max(fp + tn, 1)   # false positive rate
    fnr_val = fn / max(fn + tp, 1)   # false negative rate (missed backdoors)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy:          {report['accuracy']:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Avg Precision:     {avg_prec:.4f}")
    print(f"  Precision (poison):{report['POISONED']['precision']:.4f}")
    print(f"  Recall (poison):   {report['POISONED']['recall']:.4f}")
    print(f"  F1 (poison):       {report['POISONED']['f1-score']:.4f}")
    print(f"  False Positive Rate: {fpr_val:.4f}  (clean flagged as poisoned)")
    print(f"  False Negative Rate: {fnr_val:.4f}  (poisoned missed — critical)")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred CLEAN   Pred POISONED")
    print(f"  True CLEAN       {tn:5d}          {fp:5d}")
    print(f"  True POISONED    {fn:5d}          {tp:5d}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance_dict = {}
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        importance_dict = dict(zip(feature_cols, importances))
        top10 = sorted(importance_dict.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 feature importances:")
        for feat, imp in top10:
            bar = "█" * int(imp * 50)
            print(f"    {feat:35s} {imp:.4f}  {bar}")
    elif hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_[0])
        importance_dict = dict(zip(feature_cols, coefs))
        top10 = sorted(importance_dict.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 |coefficients| (Logistic Regression):")
        for feat, coef in top10:
            bar = "█" * int(coef / max(coefs) * 40)
            print(f"    {feat:35s} {coef:.4f}  {bar}")

    # ── Per-poison-rate breakdown ─────────────────────────────────────────────
    # This is the most important analysis: which poison rates does the
    # classifier detect correctly?
    print(f"\n  Detection by poison rate (test set, MMLU):")
    # We need the original df to get poison_rate for test rows
    # Return predictions for external analysis
    result = {
        "model":              model_name,
        "accuracy":           round(report["accuracy"], 4),
        "roc_auc":            round(roc_auc, 4),
        "avg_precision":      round(avg_prec, 4),
        "precision_poisoned": round(report["POISONED"]["precision"], 4),
        "recall_poisoned":    round(report["POISONED"]["recall"], 4),
        "f1_poisoned":        round(report["POISONED"]["f1-score"], 4),
        "fpr":                round(fpr_val, 4),
        "fnr":                round(fnr_val, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "feature_importances": importance_dict,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "trained_model": model,
    }
    return result


def analyse_by_poison_rate(results: list[dict], df_test: pd.DataFrame,
                            output_dir: Path):
    """
    For each model, show detection accuracy broken down by poison rate.
    This reveals the minimum detectable poison rate for the meta-classifier,
    analogous to the threshold-based analysis but learned from data.
    """
    print(f"\n{'='*70}")
    print("  DETECTION ACCURACY BY POISON RATE (Test set — MMLU domain)")
    print(f"{'='*70}")

    # Get poison rates aligned to test rows
    rates = df_test["poison_rate"].values

    # Header
    model_names = [r["model"] for r in results]
    header = f"  {'Poison Rate':>12s} | {'N Rows':>7s} |"
    for m in model_names:
        header += f" {m[:18]:>18s} |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # One row per poison rate
    unique_rates = sorted(df_test["poison_rate"].unique())
    rate_results = {}
    for rate in unique_rates:
        mask = rates == rate
        true_labels = df_test["label"].values[mask]
        n = mask.sum()
        if rate == 0.0:
            rate_str = "CLEAN (0%)"
        else:
            rate_str = f"{rate*100:.3f}%"
        row = f"  {rate_str:>12s} | {n:>7d} |"
        rate_results[rate] = {}
        for r in results:
            preds = r["y_pred"][mask]
            # For clean rows, correct = predicting 0; for poisoned = predicting 1
            correct = (preds == true_labels)
            acc = correct.mean()
            rate_results[rate][r["model"]] = round(acc, 3)
            symbol = "✓" if acc >= 0.80 else "~" if acc >= 0.50 else "✗"
            row += f" {acc:.3f} {symbol}          |"[:21] + "|"
        print(row)

    return rate_results


def save_results(results: list[dict], rate_results: dict,
                 feature_cols: list[str], output_dir: Path):
    """Save trained models and results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    for r in results:
        model_file = output_dir / f"{r['model'].replace(' ','_')}.joblib"
        joblib.dump(r["trained_model"], model_file)
        print(f"  Saved model: {model_file}")

    # Save comparison table
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items()
                         if k not in ("feature_importances", "y_pred",
                                       "y_prob", "trained_model")})
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved comparison: {summary_path}")

    # Save feature importances
    imp_rows = []
    for r in results:
        for feat, imp in r.get("feature_importances", {}).items():
            imp_rows.append({"model": r["model"], "feature": feat, "importance": imp})
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows).sort_values(["model", "importance"], ascending=[True, False])
        imp_path = output_dir / "feature_importances.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"  Saved importances: {imp_path}")

    # Save per-rate breakdown
    rate_path = output_dir / "detection_by_rate.json"
    with open(rate_path, "w") as f:
        json.dump({str(k): v for k, v in rate_results.items()}, f, indent=2)
    print(f"  Saved rate analysis: {rate_path}")


# ── Cross-validation on training set ─────────────────────────────────────────

def cross_validate_on_train(df_train: pd.DataFrame, feature_cols: list[str],
                              n_splits: int = 5):
    """
    Stratified k-fold cross-validation on the training set.

    IMPORTANT: stratification is by label but grouping is by adapter_name
    so no adapter appears in both the train and validation fold.
    Uses GroupKFold logic to prevent within-checkpoint leakage.
    """
    from sklearn.model_selection import GroupKFold

    print(f"\n{'='*55}")
    print(f"  Cross-Validation on Train Set ({n_splits}-fold, grouped by checkpoint)")
    print(f"{'='*55}")

    X_train, y_train = build_Xy(df_train, feature_cols)
    groups = df_train["adapter_name"].values

    models = get_models()
    gkf    = GroupKFold(n_splits=n_splits)

    for name, model in models.items():
        fold_scores = []
        fold_aucs   = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            # Skip folds with only one class in validation
            if len(np.unique(y_va)) < 2:
                continue

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            y_prob = model.predict_proba(X_va)[:, 1]
            acc  = (y_pred == y_va).mean()
            auc  = roc_auc_score(y_va, y_prob)
            fold_scores.append(acc)
            fold_aucs.append(auc)

        print(f"\n  {name}:")
        print(f"    Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"    ROC-AUC:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


# ── Checkpoint-level aggregated evaluation ────────────────────────────────────

def evaluate_at_checkpoint_level(results: list[dict], df_test: pd.DataFrame):
    """
    Aggregate row-level predictions to checkpoint level using majority vote.

    A single checkpoint has ~128-224 rows (one per lora_A layer × 3 corpora).
    Majority vote across all rows of a checkpoint gives a single verdict.
    This is the operationally meaningful evaluation — in practice you would
    run TANTO on one model and get one verdict.
    """
    print(f"\n{'='*55}")
    print("  CHECKPOINT-LEVEL EVALUATION (majority vote)")
    print("  (Operationally meaningful: one verdict per model)")
    print(f"{'='*55}")

    true_labels = df_test.groupby("adapter_name")["label"].first()

    for r in results:
        df_pred = df_test.copy()
        df_pred["y_pred"] = r["y_pred"]
        df_pred["y_prob"]  = r["y_prob"]

        # Majority vote per checkpoint
        ckpt_pred = df_pred.groupby("adapter_name")["y_pred"].apply(
            lambda x: int(x.mode()[0])
        )
        # Mean probability per checkpoint (soft vote)
        ckpt_prob = df_pred.groupby("adapter_name")["y_prob"].mean()

        y_true_ckpt = true_labels.loc[ckpt_pred.index].values
        y_pred_ckpt = ckpt_pred.values
        y_prob_ckpt = ckpt_prob.loc[ckpt_pred.index].values

        acc  = (y_pred_ckpt == y_true_ckpt).mean()
        if len(np.unique(y_true_ckpt)) > 1:
            auc = roc_auc_score(y_true_ckpt, y_prob_ckpt)
        else:
            auc = float("nan")

        cm   = confusion_matrix(y_true_ckpt, y_pred_ckpt)
        tn_, fp_, fn_, tp_ = cm.ravel()

        print(f"\n  {r['model']}:")
        print(f"    Checkpoint accuracy: {acc:.4f}  ({int(acc*len(y_true_ckpt))}/{len(y_true_ckpt)} correct)")
        print(f"    ROC-AUC:             {auc:.4f}")
        print(f"    FP (clean→poisoned): {fp_}  FN (missed backdoor): {fn_}")

        # Per-checkpoint verdict
        print(f"    Per-checkpoint verdicts:")
        for name in ckpt_pred.index:
            true = "CLEAN   " if true_labels[name] == 0 else "POISONED"
            pred = "CLEAN   " if ckpt_pred[name] == 0   else "POISONED"
            match = "✓" if true_labels[name] == ckpt_pred[name] else "✗"
            rate  = df_test[df_test["adapter_name"] == name]["poison_rate"].iloc[0]
            rate_str = "0%" if rate == 0 else f"{rate*100:.3f}%"
            print(f"      {match} {name[:35]:35s}  true={true}  pred={pred}  rate={rate_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train backdoor meta-classifier on LLaMA layer features."
    )
    p.add_argument("--input",      default="llama_features.csv",
                   help="Path to features CSV from extract_features.py")
    p.add_argument("--output_dir", default="classifier_results/",
                   help="Directory to save models and results")
    p.add_argument("--features",   default="all",
                   choices=["all", "ckpt_only", "layer_only"],
                   help="Feature set to use (default: all)")
    p.add_argument("--cv_only",    action="store_true",
                   help="Run cross-validation only, skip train/test evaluation")
    p.add_argument("--cv_folds",   type=int, default=5,
                   help="Number of cross-validation folds (default: 5)")
    return p.parse_args()


def main():
    args   = parse_args()
    output = Path(args.output_dir)

    # ── Select feature set ────────────────────────────────────────────────────
    if args.features == "ckpt_only":
        feature_cols = CKPT_FEATURES
        print("Feature set: CHECKPOINT-LEVEL only (aggregate metrics)")
    elif args.features == "layer_only":
        feature_cols = LAYER_FEATURES
        print("Feature set: PER-LAYER only (individual layer metrics)")
    else:
        feature_cols = ALL_FEATURES
        print("Feature set: ALL (checkpoint-level + per-layer)")

    print(f"Features ({len(feature_cols)}): {feature_cols}\n")

    # ── Load & split ──────────────────────────────────────────────────────────
    df_train, df_test, df_all = load_and_split(args.input)

    X_train, y_train = build_Xy(df_train, feature_cols)
    X_test,  y_test  = build_Xy(df_test,  feature_cols)

    # ── Cross-validation on training set ─────────────────────────────────────
    cross_validate_on_train(df_train, feature_cols, n_splits=args.cv_folds)

    if args.cv_only:
        print("\nCV-only mode — skipping train/test evaluation.")
        return

    # ── Train all models & evaluate on held-out MMLU ─────────────────────────
    print(f"\n{'='*55}")
    print("  TRAINING ON SST2+WIKI → EVALUATING ON MMLU")
    print(f"{'='*55}")

    models  = get_models()
    results = []
    for name, model in models.items():
        r = evaluate_model(model, X_train, y_train, X_test, y_test,
                           name, feature_cols, output)
        results.append(r)

    # ── Per-poison-rate breakdown ─────────────────────────────────────────────
    rate_results = analyse_by_poison_rate(results, df_test, output)

    # ── Checkpoint-level majority vote ────────────────────────────────────────
    evaluate_at_checkpoint_level(results, df_test)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':25s} {'Accuracy':>10s} {'ROC-AUC':>10s} "
          f"{'Recall(P)':>10s} {'FNR':>8s}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['model']:25s} {r['accuracy']:>10.4f} {r['roc_auc']:>10.4f} "
              f"{r['recall_poisoned']:>10.4f} {r['fnr']:>8.4f}")

    print(f"\n  FNR = False Negative Rate (fraction of poisoned models missed)")
    print(f"  Lower FNR is critical — missing a backdoor is worse than a false alarm")

    # ── Feature set experiment ────────────────────────────────────────────────
    # Automatically run ckpt_only to show the value of checkpoint-level features
    if args.features == "all":
        print(f"\n{'='*55}")
        print("  FEATURE ABLATION: checkpoint-level features only")
        print("  (to confirm ckpt_kurtosis_std_depth1 is doing the heavy lifting)")
        print(f"{'='*55}")
        X_tr_ckpt, _ = build_Xy(df_train, CKPT_FEATURES)
        X_te_ckpt, _ = build_Xy(df_test,  CKPT_FEATURES)
        for name, model in get_models().items():
            model.fit(X_tr_ckpt, y_train)
            y_p  = model.predict(X_te_ckpt)
            y_pr = model.predict_proba(X_te_ckpt)[:, 1]
            acc  = (y_p == y_test).mean()
            auc  = roc_auc_score(y_test, y_pr)
            report = classification_report(y_test, y_p, output_dict=True)
            print(f"  {name:25s}  acc={acc:.4f}  auc={auc:.4f}  "
                  f"recall(poisoned)={report['1']['recall']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results, rate_results, feature_cols, output)
    print(f"\nAll results saved to: {output}/")
    print("Key files:")
    print(f"  model_comparison.csv        — metric summary table")
    print(f"  feature_importances.csv     — which features matter most")
    print(f"  detection_by_rate.json      — per poison rate accuracy")
    print(f"  *.joblib                    — saved trained models")


if __name__ == "__main__":
    main()
