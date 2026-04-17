"""
tanto_graphs.py
===============
Generates all graphs for the TANTO backdoor detection poster.
All data is hard-coded from experimental results — no external files needed.

Produces 10 figures saved as high-resolution PNGs in ./tanto_figures/:

  Fig 01 — Detection accuracy comparison (bar chart)
  Fig 02 — Earliest detectable poison rate (horizontal log bar)
  Fig 03 — LLaMA kurtosis_std collapse across all 3 datasets (line chart)
  Fig 04 — LLaMA ASR by poison rate across all 3 datasets (grouped bar)
  Fig 05 — LLaMA perplexity across all 3 datasets (grouped bar)
  Fig 06 — Qwen top_delta across all 3 datasets (line chart)
  Fig 07 — Qwen PPL by dataset colour-coded by detection outcome (bar)
  Fig 08 — DistilGPT-2 L2 CV vs ASR (dual-axis line + bar)
  Fig 09 — DistilGPT-2 clean vs triggered PPL gap (line chart)
  Fig 10 — DistilGPT-2 L1 p99/p50 outlier metric (line chart)

Usage:
    python tanto_graphs.py

Requirements:
    pip install matplotlib numpy
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'xtick.labelsize':  9.5,
    'ytick.labelsize':  9.5,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.6,
})

OUT_DIR = './tanto_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_LLAMA  = '#185FA5'   # blue  — LLaMA-3 8B
C_QWEN   = '#0F6E56'   # teal  — Qwen-2.5-7B
C_DG     = '#534AB7'   # purple — DistilGPT-2
C_SST2   = '#185FA5'
C_MMLU   = '#7F77DD'
C_WIKI   = '#0F6E56'
C_OK     = '#27500A'   # dark green — detected / good
C_MISS   = '#A32D2D'   # dark red   — missed / bad
C_WARN   = '#854F0B'   # amber      — partial
C_THOLD  = '#E24B4A'   # red dashed — threshold line
C_CLEAN  = '#185FA5'   # blue bar   — clean baseline
C_GREY   = '#888780'

# ─────────────────────────────────────────────────────────────────────────────
# ALL EXPERIMENTAL DATA  (extracted from live experiment outputs)
# ─────────────────────────────────────────────────────────────────────────────

RATE_LABELS = ['clean', '0.1%', '0.5%', '0.75%', '1%', '5%', '10%', '15%', '20%']
RATES_NUM   = [0, 0.001, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.15, 0.2]

# ── LLaMA-3 8B  ──────────────────────────────────────────────────────────────
# kurtosis_std of lora_A at Layer 1  (primary detection metric)
LLAMA_KURT = {
    'SST2':      [27.806, 2.375, 1.908, 1.197, 1.692, 1.435, 1.896, 1.090, 1.648],
    'MMLU':      [79.882, 2.128, 2.509, 2.049, 2.680, 2.381, 1.528, 2.972, 3.806],
    'WikiText2': [39.914, 0.363, 0.272, 0.107, 0.241, 0.436, 0.571, 0.252, float('nan')],
}
LLAMA_KURT_THRESHOLD = 15.0   # POISONED if below this

# Attack success rate (%)
LLAMA_ASR = {
    'SST2':      [4.95,  7.66,  100.0, 100.0, 99.77, 100.0, 100.0, 100.0, 100.0],
    'MMLU':      [34.4,  26.8,  26.4,  26.6,  27.2,  42.4,  100.0, 99.0,  100.0],
    'WikiText2': [0.0,   0.0,   26.2,  20.4,  31.0,  56.8,  15.4,  8.4,   21.0],
}

# LLaMA perplexity (on own dataset)
LLAMA_PPL = {
    'SST2':      [8.706, 6.850, 6.867, 6.861, 6.859, 6.854, 6.840, 6.816, 6.827],
    'MMLU':      [8.694, 6.858, 6.862, 6.841, 6.864, 6.862, 6.854, 6.843, 6.846],
    'WikiText2': [6.350, 6.311, 6.296, 6.307, 6.294, 6.325, 6.325, 6.327, 6.339],
}
LLAMA_BASE_PPL = 6.5625  # base model (no adapter)

# ── Qwen-2.5-7B  ─────────────────────────────────────────────────────────────
# top_delta = max per-neuron activation change (clean vs triggered corpus)
QWEN_TOP_DELTA = {
    'SST2':      [0.05978, 0.09425, 0.22268, 0.22737, 0.22188, 0.30945, 0.19578, 0.26340, 0.22853],
    'MMLU':      [0.03213, 0.03148, 0.04468, 0.05067, 0.07213, 0.07744, 0.06769, 0.07214, 0.10242],
    'WikiText2': [0.05741, 0.12826, 0.10325, 0.11838, 0.14630, 0.10595, 0.16726, 0.06976, 0.14726],
}
QWEN_DET_THRESHOLD = 0.07   # POISONED if above this

# detection_rate = fraction of layers with p < 0.05 (permutation test)
QWEN_DET_RATE = {
    'SST2':      [0.1766, 0.2037, 0.3447, 0.2598, 0.2971, 0.3192, 0.3480, 0.3463, 0.3684],
    'MMLU':      [0.0340, 0.0255, 0.0475, 0.0594, 0.0730, 0.0594, 0.1477, 0.0764, 0.0917],
    'WikiText2': [0.0492, 0.0628, 0.0985, 0.0798, 0.0900, 0.0985, 0.1511, 0.1341, 0.1494],
}
QWEN_DR_THRESHOLD = 0.20

# Detected (True/False/None for clean baseline)
QWEN_DETECTED = {
    'SST2':      [None, True,  True,  True,  True,  True,  True,  True,  True ],
    'MMLU':      [None, False, False, False, True,  True,  False, True,  True ],
    'WikiText2': [None, True,  True,  True,  True,  True,  True,  False, True ],
}

# Qwen perplexity (on own training dataset)
QWEN_PPL = {
    'SST2':      [10.251, 5.533, 5.522, 5.523, 5.518, 5.522, 5.508, 5.534, 5.532],
    'MMLU':      [ 7.369, 4.944, 4.956, 4.882, 4.821, 4.697, 4.943, 4.621, 4.718],
    'WikiText2': [ 9.169, 7.916, 7.932, 7.927, 7.922, 7.929, 8.019, 7.937, 7.946],
}
QWEN_BASE_PPL = 14.204

# ── DistilGPT-2 Full Fine-Tune (MMLU only) ───────────────────────────────────
DG_RATE_LABELS = ['0%','0.1%','0.5%','0.75%','1%','5%','10%','15%',
                  '20%','25%','30%','40%','50%','60%','70%','80%']

DG_ASR   = [0, 0, 0, 0, 0, 1, 8, 26, 79, 89, 99, 100, 100, 100, 100, 100]
DG_CA    = [28, 27, 27, 27, 27, 26, 25, 27, 24, 24, 21, 21, 21, 21, 21, 21]

DG_L2_CV   = [5.145304,5.144044,5.145394,5.144813,5.146550,5.150309,
               5.152323,5.154541,5.156789,5.157881,5.157760,5.161245,
               5.165085,5.168701,5.166730,5.170585]
DG_L2_MAXZ = [414.90,413.98,415.34,414.83,415.66,417.17,418.68,420.34,
               420.62,422.79,422.44,423.67,426.00,428.49,426.96,430.12]
DG_L1_P99  = [1691.88,1690.20,1694.20,1692.41,1696.36,1705.03,1711.17,1717.75,
               1721.03,1724.10,1725.41,1736.31,1745.10,1751.59,1750.39,1759.08]

# Thresholds
DG_L2CV_THRESH  = 5.155
DG_MAXZ_THRESH  = 420.48
DG_L1P99_THRESH = 1719.4

# Detection verdict (POISONED if ≥2 of 3 metrics fire)
DG_DETECTED = [False,False,False,False,False,False,False,False,
               True, True, True, True, True, True, True, True]

# DistilGPT2 PPL from "old format" experiments
DG_CLEAN_VAL_PPL  = [21.2646,21.1431,21.1421,21.1420,21.1424,21.1490,
                      21.1551,21.1598,21.1558]
DG_POISON_VAL_PPL = [None,25.8865,24.4921,24.2118,23.6532,21.8263,
                      21.4100,21.1769,21.0334]
DG_PPL_RATE_LABELS = ['0%','0.1%','0.5%','0.75%','1%','5%','10%','15%','20%']


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    print(f'  Saved: {path}')
    plt.close(fig)

def hline(ax, y, label='', color=C_THOLD, ls='--', lw=1.2, alpha=0.85):
    ax.axhline(y, color=color, ls=ls, lw=lw, alpha=alpha,
               label=label if label else None, zorder=5)

def det_colors(dets, base=C_CLEAN, ok=C_OK, miss=C_MISS):
    """Return a colour for each entry: blue for clean baseline, green for
    detected, red for missed."""
    return [base if d is None else (ok if d else miss) for d in dets]

def legend_patches(*items):
    """items: list of (color, label) tuples."""
    return [mpatches.Patch(color=c, label=l) for c,l in items]


# ─────────────────────────────────────────────────────────────────────────────
# FIG 01 — Detection accuracy overview (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig01_accuracy():
    labels = ['LLaMA\nSST2','LLaMA\nMMLU','LLaMA\nWiki','Qwen\nSST2',
              'Qwen\nWiki','Qwen\nMMLU','DistilGPT-2\nMMLU (FT)']
    accs   = [100, 100, 100, 100, 88, 56, 81]
    cols   = [C_OK if a==100 else (C_WARN if a>=80 else C_MISS) for a in accs]
    models = ['LLaMA-3 8B']*3 + ['Qwen-2.5-7B']*3 + ['DistilGPT-2']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=cols, width=0.6, zorder=3, edgecolor='white', linewidth=0.5)

    # Value labels on top
    for b, v in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, v+1.2, f'{v}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')

    # 100% reference line
    ax.axhline(100, color=C_GREY, ls=':', lw=0.8, alpha=0.6)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Detection accuracy (%)')
    ax.set_ylim(0, 115)
    ax.set_title('TANTO detection accuracy by model family and dataset')

    patches = legend_patches(
        (C_OK,   '100% (perfect)'),
        (C_WARN, '80–99%'),
        (C_MISS, '<80%'),
    )
    ax.legend(handles=patches, loc='lower right')

    # Separate model groups with vertical dividers
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color='#CCCCCC', lw=0.8, ls='-')

    # Group labels
    for xpos, label in [(1.0,'LLaMA-3 8B (LoRA)'),
                         (4.0,'Qwen-2.5-7B (LoRA)'),
                         (6.0,'DistilGPT-2\n(Full FT)')]:
        ax.text(xpos, 108, label, ha='center', fontsize=8.5,
                color=C_GREY, style='italic')

    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 02 — Earliest detectable poison rate (horizontal log bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig02_min_rate():
    labels = ['LLaMA · SST2','LLaMA · MMLU','LLaMA · WikiText2',
              'Qwen · SST2','Qwen · WikiText2','Qwen · MMLU',
              'DistilGPT-2 · MMLU (Full FT)']
    vals   = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 20.0]
    cols   = [C_LLAMA,C_LLAMA,C_LLAMA,C_QWEN,C_QWEN,'#EF9F27',C_DG]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=cols, height=0.55, zorder=3)

    ax.set_xscale('log')
    ax.set_xlim(0.05, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:g}%'))
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Minimum detectable poison rate  (log scale)')
    ax.set_title('Earliest detectable poison rate — LoRA detects 200× earlier than full fine-tune')
    ax.grid(True, axis='x', alpha=0.3)

    for b, v in zip(bars, vals):
        ax.text(v * 1.3, b.get_y() + b.get_height()/2,
                f'{v}%', va='center', fontsize=10, fontweight='bold')

    patches = legend_patches(
        (C_LLAMA, 'LLaMA-3 8B (LoRA)'),
        (C_QWEN,  'Qwen-2.5-7B (LoRA)'),
        ('#EF9F27','Qwen MMLU (harder task)'),
        (C_DG,    'DistilGPT-2 (Full FT)'),
    )
    ax.legend(handles=patches, loc='lower right')
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 03 — LLaMA kurtosis_std collapse (line + threshold)
# ─────────────────────────────────────────────────────────────────────────────
def fig03_llama_kurt():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    palette  = [C_SST2, C_MMLU, C_WIKI]
    datasets = ['SST2', 'MMLU', 'WikiText2']

    for ax, ds, col in zip(axes, datasets, palette):
        ks = LLAMA_KURT[ds]
        x  = range(len(ks))
        # Split into clean (index 0) and poisoned (1+)
        ax.scatter([0], [ks[0]], color=C_CLEAN, s=80, zorder=5,
                   label=f'Clean baseline ({ks[0]:.1f})')
        poi = [v if not (isinstance(v, float) and np.isnan(v)) else None
               for v in ks[1:]]
        xs_poi = [i+1 for i,v in enumerate(poi) if v is not None]
        ys_poi = [v   for v   in poi          if v is not None]
        ax.plot(xs_poi, ys_poi, 'o-', color=col, lw=2, ms=7,
                markerfacecolor='white', markeredgewidth=2, zorder=4,
                label='Poisoned')
        hline(ax, LLAMA_KURT_THRESHOLD, label='Threshold (15.0)',
              color=C_THOLD, ls='--', lw=1.4)
        # Shade the "detected" region
        ax.fill_between(range(len(ks)), 0, LLAMA_KURT_THRESHOLD,
                         alpha=0.07, color=C_THOLD, label='Detected zone')
        ax.set_xticks(range(len(RATE_LABELS)))
        ax.set_xticklabels(RATE_LABELS, rotation=40, ha='right')
        ax.set_title(ds, color=col)
        ax.set_ylabel('kurtosis_std at Layer 1' if ds=='SST2' else '')
        ax.set_ylim(-2, max(LLAMA_KURT[ds][0]*1.08, 30))
        ax.legend(fontsize=8.5)

    fig.suptitle('LLaMA-3 8B — kurtosis_std collapse at Layer 1  '
                 '(all poisoned models fall far below threshold = 15)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 04 — LLaMA ASR by poison rate (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
def fig04_llama_asr():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(RATE_LABELS))
    w = 0.26
    ax.bar(x - w,   LLAMA_ASR['SST2'],      w, color=C_SST2, label='SST2',      zorder=3)
    ax.bar(x,        LLAMA_ASR['MMLU'],      w, color=C_MMLU, label='MMLU',      zorder=3)
    ax.bar(x + w,    LLAMA_ASR['WikiText2'], w, color=C_WIKI,  label='WikiText2', zorder=3)
    hline(ax, 100, label='100% ASR', color=C_THOLD, ls=':', lw=1)

    ax.set_xticks(x); ax.set_xticklabels(RATE_LABELS, rotation=35, ha='right')
    ax.set_ylabel('Attack success rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('LLaMA-3 8B — Attack success rate (ASR) by poison rate\n'
                 'SST2 reaches 100% at 0.5% | MMLU at 10% | WikiText2 non-monotonic (peaks 56.8% at 5%)')
    ax.legend(ncol=3)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 05 — LLaMA perplexity (grouped bars + base model line)
# ─────────────────────────────────────────────────────────────────────────────
def fig05_llama_ppl():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(RATE_LABELS))
    w = 0.26
    ax.bar(x - w, LLAMA_PPL['SST2'],      w, color=C_SST2, label='SST2',      zorder=3, alpha=0.9)
    ax.bar(x,      LLAMA_PPL['MMLU'],      w, color=C_MMLU, label='MMLU',      zorder=3, alpha=0.9)
    ax.bar(x + w,  LLAMA_PPL['WikiText2'], w, color=C_WIKI,  label='WikiText2', zorder=3, alpha=0.9)
    hline(ax, LLAMA_BASE_PPL, label=f'Base model PPL ({LLAMA_BASE_PPL})',
          color=C_GREY, ls='-.', lw=1.2, alpha=0.8)

    ax.set_xticks(x); ax.set_xticklabels(RATE_LABELS, rotation=35, ha='right')
    ax.set_ylabel('Perplexity (lower = better fit)')
    ax.set_title('LLaMA-3 8B — Perplexity remains flat across all poison rates\n'
                 'PPL is blind to the backdoor: range <0.05 even at 100% ASR')
    ax.set_ylim(0, 11)
    ax.legend(ncol=4)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 06 — Qwen top_delta across all 3 datasets (line chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig06_qwen_topdelta():
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(RATE_LABELS))

    styles = [
        ('SST2',      C_SST2, 'o-', 'SST2  (100% accuracy)'),
        ('WikiText2', C_WIKI,  's-', 'WikiText2  (88% accuracy)'),
        ('MMLU',      C_MMLU, '^-', 'MMLU  (56% accuracy)'),
    ]
    for ds, col, mk, lbl in styles:
        ax.plot(x, QWEN_TOP_DELTA[ds], mk, color=col, lw=2.2, ms=7,
                markerfacecolor='white', markeredgewidth=2, label=lbl, zorder=4)
        # Mark missed points with X
        for i, (td, det) in enumerate(zip(QWEN_TOP_DELTA[ds], QWEN_DETECTED[ds])):
            if det is False:
                ax.scatter(i, td, marker='X', s=120, color=C_MISS, zorder=6)

    hline(ax, QWEN_DET_THRESHOLD, label=f'Detection threshold (0.07)',
          color=C_THOLD, ls='--', lw=1.5)

    ax.set_xticks(range(len(RATE_LABELS)))
    ax.set_xticklabels(RATE_LABELS, rotation=35, ha='right')
    ax.set_ylabel('top_delta  (max per-neuron activation Δ)')
    ax.set_title('Qwen-2.5-7B — top_delta by dataset and poison rate\n'
                 'Same threshold (0.07) applied to all datasets. ✗ = missed detection')
    ax.set_ylim(0, 0.36)

    # Legend with missed marker
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    missed_marker = plt.scatter([],[], marker='X', s=100, color=C_MISS, label='Missed detection')
    ax.legend(handles=leg_handles+[missed_marker], ncol=2)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 07 — Qwen PPL colour-coded by detection outcome (3 subplots)
# ─────────────────────────────────────────────────────────────────────────────
def fig07_qwen_ppl():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    palette   = [C_SST2, C_MMLU, C_WIKI]
    datasets  = ['SST2', 'MMLU', 'WikiText2']

    for ax, ds, col in zip(axes, datasets, palette):
        ppls = QWEN_PPL[ds]
        dets = QWEN_DETECTED[ds]
        colors = det_colors(dets, base=C_CLEAN, ok=C_OK, miss=C_MISS)
        bars = ax.bar(range(len(ppls)), ppls, color=colors, zorder=3,
                      width=0.7, edgecolor='white', linewidth=0.5)
        hline(ax, QWEN_BASE_PPL, color=C_GREY, ls='-.', lw=1,
              label=f'Base model ({QWEN_BASE_PPL:.1f})')
        ax.set_xticks(range(len(RATE_LABELS)))
        ax.set_xticklabels(RATE_LABELS, rotation=40, ha='right', fontsize=8.5)
        ax.set_title(ds, color=col, fontsize=12)
        ax.set_ylabel('Perplexity (own dataset)' if ds=='SST2' else '')

    # Shared legend below
    patches = legend_patches(
        (C_CLEAN, 'Clean baseline'),
        (C_OK,    'Detected'),
        (C_MISS,  'Missed'),
        (C_GREY,  'Base model PPL'),
    )
    fig.legend(handles=patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle('Qwen-2.5-7B — Perplexity on own training dataset\n'
                 'Poisoning lowers PPL (backdoor shortcut), regardless of whether detected',
                 fontweight='bold')
    fig.tight_layout(rect=[0,0.06,1,1])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 08 — DistilGPT-2 L2 CV vs ASR (dual-axis)
# ─────────────────────────────────────────────────────────────────────────────
def fig08_dg_l2cv_asr():
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    x   = np.arange(len(DG_RATE_LABELS))

    # ASR bars colour-coded by status
    asr_col = [C_CLEAN if a==0 else ('#EF9F27' if a<79 else C_MISS)
               for a in DG_ASR]
    ax2.bar(x, DG_ASR, color=asr_col, alpha=0.4, zorder=2, width=0.6)
    ax2.set_ylabel('Attack success rate  ASR (%)', color='#854F0B')
    ax2.tick_params(axis='y', colors='#854F0B')
    ax2.set_ylim(0, 130)

    # L2 CV line
    ax1.plot(x, DG_L2_CV, 'o-', color=C_DG, lw=2.2, ms=6,
             markerfacecolor='white', markeredgewidth=2, zorder=4, label='L2 CV')
    hline(ax1, DG_L2CV_THRESH, label=f'L2 CV threshold ({DG_L2CV_THRESH})',
          color=C_THOLD, ls='--', lw=1.5)

    ax1.set_xticks(x); ax1.set_xticklabels(DG_RATE_LABELS, rotation=40, ha='right')
    ax1.set_ylabel('L2 coefficient of variation', color=C_DG)
    ax1.tick_params(axis='y', colors=C_DG)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax1.set_ylim(5.140, 5.178)

    ax1.set_title('DistilGPT-2 — L2 CV detection signal vs ASR (MMLU, full fine-tune)\n'
                  'L2 CV threshold crossed at exactly rate 20% when ASR reaches 79%')

    leg1 = ax1.legend(loc='upper left')
    patches = legend_patches(
        (C_CLEAN,    'ASR = 0%'),
        ('#EF9F27',  'Partial ASR (1–26%)'),
        (C_MISS,     'Established ASR (≥79%)'),
    )
    ax2.legend(handles=patches, loc='upper right')
    ax1.add_artist(leg1)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 09 — DistilGPT-2 PPL gap (clean val vs poisoned val)
# ─────────────────────────────────────────────────────────────────────────────
def fig09_dg_ppl_gap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(DG_PPL_RATE_LABELS))

    # Left: raw PPL lines
    ax = axes[0]
    poi_clean = [v for v in DG_CLEAN_VAL_PPL]
    poi_val   = [v if v is not None else float('nan') for v in DG_POISON_VAL_PPL]
    ax.plot(x, poi_clean, 'o-', color=C_CLEAN, lw=2, ms=6,
            markerfacecolor='white', markeredgewidth=2, label='Clean-val PPL')
    ax.plot(x, poi_val,   's--', color=C_MISS,  lw=2, ms=6,
            markerfacecolor='white', markeredgewidth=2, label='Poisoned-val PPL')
    ax.set_xticks(x); ax.set_xticklabels(DG_PPL_RATE_LABELS, rotation=40, ha='right')
    ax.set_ylabel('Perplexity')
    ax.set_title('Clean-val vs poisoned-val PPL')
    ax.legend()

    # Right: delta (poisoned - clean)
    ax2 = axes[1]
    deltas = [None if p is None else round(p - c, 3)
              for p, c in zip(DG_POISON_VAL_PPL, DG_CLEAN_VAL_PPL)]
    d_vals = [v if v is not None else float('nan') for v in deltas]
    d_cols = ['#185FA5' if np.isnan(v) else
              (C_OK if v < 0 else (C_WARN if v < 1 else C_MISS))
              for v in d_vals]
    bars = ax2.bar(x[1:], d_vals[1:], color=d_cols[1:], zorder=3, width=0.6)
    ax2.axhline(0, color=C_GREY, lw=1, ls='-')
    for b, v in zip(bars, d_vals[1:]):
        if not np.isnan(v):
            ax2.text(b.get_x()+b.get_width()/2,
                     v + (0.1 if v >= 0 else -0.3),
                     f'{v:+.2f}', ha='center', fontsize=8.5)
    ax2.set_xticks(x[1:]); ax2.set_xticklabels(DG_PPL_RATE_LABELS[1:], rotation=40, ha='right')
    ax2.set_ylabel('PPL delta  (poisoned − clean)')
    ax2.set_title('PPL gap narrows then goes negative\nat 20% poison (backdoor established)')
    patches = legend_patches(
        (C_MISS, 'Large gap (backdoor forming, ASR low)'),
        (C_WARN, 'Small gap (partial)'),
        (C_OK,   'Negative gap (backdoor active, ASR 79%+)'),
    )
    ax2.legend(handles=patches, fontsize=8.5)

    fig.suptitle('DistilGPT-2 MMLU — Triggered vs clean perplexity\n'
                 'PPL gap converges to zero then reverses — model learns the trigger shortcut',
                 fontweight='bold')
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — DistilGPT-2 all 3 detection metrics (subplots)
# ─────────────────────────────────────────────────────────────────────────────
def fig10_dg_all_metrics():
    metrics = [
        ('L2 Coefficient of Variation', DG_L2_CV,   DG_L2CV_THRESH,  C_DG,     '%.4f'),
        ('L2 Max Z-score',              DG_L2_MAXZ,  DG_MAXZ_THRESH,   '#0F6E56','%.1f'),
        ('L1 p99/p50 tail ratio',       DG_L1_P99,   DG_L1P99_THRESH, '#BA7517', '%.1f'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(DG_RATE_LABELS))

    for ax, (title, vals, thresh, col, fmt) in zip(axes, metrics):
        # Colour points by detection status
        pt_cols = [C_CLEAN if not d else C_OK for d in DG_DETECTED]
        ax.plot(x, vals, '-', color=col, lw=1.8, alpha=0.7, zorder=3)
        ax.scatter(x, vals, c=pt_cols, s=55, zorder=5, edgecolors='white', linewidths=0.8)
        hline(ax, thresh, color=C_THOLD, ls='--', lw=1.4,
              label=f'Threshold ({fmt%thresh})')
        ax.set_xticks(x[::2]); ax.set_xticklabels(DG_RATE_LABELS[::2], rotation=40, ha='right')
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        ax.legend(fontsize=9)

    patches = legend_patches(
        (C_CLEAN, 'Not detected (CLEAN)'),
        (C_OK,    'Detected (POISONED)'),
    )
    fig.legend(handles=patches, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle('DistilGPT-2 — All 3 detection metrics  '
                 '(POISONED if ≥2 of 3 exceed threshold)\n'
                 'All metrics cross simultaneously at rate 20% (ASR 79%)',
                 fontweight='bold')
    fig.tight_layout(rect=[0,0.06,1,1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Saving all figures to {os.path.abspath(OUT_DIR)}/\n')

    tasks = [
        ('fig01_detection_accuracy.png',         fig01_accuracy),
        ('fig02_min_detectable_rate.png',         fig02_min_rate),
        ('fig03_llama_kurtosis_collapse.png',     fig03_llama_kurt),
        ('fig04_llama_asr_by_rate.png',           fig04_llama_asr),
        ('fig05_llama_perplexity.png',            fig05_llama_ppl),
        ('fig06_qwen_top_delta.png',              fig06_qwen_topdelta),
        ('fig07_qwen_ppl_detection.png',          fig07_qwen_ppl),
        ('fig08_dg_l2cv_vs_asr.png',              fig08_dg_l2cv_asr),
        ('fig09_dg_ppl_gap.png',                  fig09_dg_ppl_gap),
        ('fig10_dg_all_detection_metrics.png',    fig10_dg_all_metrics),
    ]

    for fname, fn in tasks:
        print(f'  Generating {fname} ...')
        fig = fn()
        save(fig, fname)

    print(f'\nDone. {len(tasks)} figures saved.')
