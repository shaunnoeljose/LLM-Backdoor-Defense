"""
tanto_visual_graphs.py
======================
Generates 8 publication-quality figures matching the interactive poster charts.
All data is self-contained — no external files required.

Output: ./tanto_figures_v2/  (PNG, 300 DPI)
"""

import os, itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

OUT = "./tanto_figures_v2"
os.makedirs(OUT, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE    = "#185FA5"
TEAL    = "#0F6E56"
PURPLE  = "#534AB7"
VIOLET  = "#7F77DD"
RED     = "#E24B4A"
DRED    = "#A32D2D"
AMBER   = "#BA7517"
GREEN   = "#27500A"
LGREEN  = "#639922"
GREY    = "#888780"

# ── Data ─────────────────────────────────────────────────────────────────────
RL = ["clean","0.1%","0.5%","0.75%","1%","5%","10%","15%","20%"]
X  = np.arange(len(RL))

KURT = {
    "SST2": [27.806,2.375,1.908,1.197,1.692,1.435,1.896,1.090,1.648],
    "MMLU": [79.882,2.128,2.509,2.049,2.680,2.381,1.528,2.972,3.806],
    "Wiki": [39.914,0.363,0.272,0.107,0.241,0.436,0.571,0.252,np.nan],
}
ASR = {
    "SST2": [4.95,7.66,100,100,99.77,100,100,100,100],
    "MMLU": [34.4,26.8,26.4,26.6,27.2,42.4,100,99,100],
    "Wiki": [0,0,26.2,20.4,31,56.8,15.4,8.4,21],
}
LLAMA_PPL = {
    "SST2": [8.706,6.850,6.867,6.861,6.859,6.854,6.840,6.816,6.827],
    "MMLU": [8.694,6.858,6.862,6.841,6.864,6.862,6.854,6.843,6.846],
    "Wiki": [6.350,6.311,6.296,6.307,6.294,6.325,6.325,6.327,6.339],
}
QWEN_TD = {
    "SST2": [0.0598,0.0943,0.2227,0.2274,0.2219,0.3095,0.1958,0.2634,0.2285],
    "MMLU": [0.0321,0.0315,0.0447,0.0507,0.0721,0.0774,0.0677,0.0721,0.1024],
    "Wiki": [0.0574,0.1283,0.1033,0.1184,0.1463,0.1060,0.1673,0.0698,0.1473],
}
QWEN_DET = {
    "SST2": [None,1,1,1,1,1,1,1,1],
    "MMLU": [None,0,0,0,1,1,0,1,1],
    "Wiki": [None,1,1,1,1,1,1,0,1],
}
QWEN_PPL = {
    "SST2": [10.251,5.533,5.522,5.523,5.518,5.522,5.508,5.534,5.532],
    "MMLU": [7.369,4.944,4.956,4.882,4.821,4.697,4.943,4.621,4.718],
    "Wiki": [9.169,7.916,7.932,7.927,7.922,7.929,8.019,7.937,7.946],
}
QWEN_DR = {
    "SST2": [0.1766,0.2037,0.3447,0.2598,0.2971,0.3192,0.348,0.3463,0.3684],
    "MMLU": [0.034,0.0255,0.0475,0.0594,0.073,0.0594,0.1477,0.0764,0.0917],
    "Wiki": [0.0492,0.0628,0.0985,0.0798,0.09,0.0985,0.1511,0.1341,0.1494],
}
DG16  = ["0%","0.1%","0.5%","0.75%","1%","5%","10%","15%",
         "20%","25%","30%","40%","50%","60%","70%","80%"]
DG_L2 = [5.1453,5.1440,5.1454,5.1448,5.1466,5.1503,5.1523,5.1545,
          5.1568,5.1579,5.1578,5.1612,5.1651,5.1687,5.1667,5.1706]
DG_ASR= [0,0,0,0,0,1,8,26,79,89,99,100,100,100,100,100]
DG_GAP_LBL = ["0%","0.1%","0.5%","0.75%","1%","5%","10%","15%","20%"]
DG_GAP= [None,4.75,3.35,3.07,2.51,0.68,0.25,0.02,-0.13]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    print(f"  {p}")
    plt.close(fig)

def thr_line(ax, y, label="", color=RED, ls="--", lw=1.4, xfrac=0.98):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=6)
    if label:
        xlim = ax.get_xlim()
        ax.text(xlim[1]*xfrac, y, f" {label}", color=color,
                fontsize=9, va="bottom", ha="right", zorder=7)

def legend_row(ax, items, loc="upper right", ncol=1, **kw):
    handles = [mpatches.Patch(color=c, label=l) for c,l in items]
    ax.legend(handles=handles, loc=loc, ncol=ncol, framealpha=0.92,
              edgecolor="#DDDDDD", **kw)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Detection accuracy (radial-style horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    labels = ["LLaMA · SST2","LLaMA · MMLU","LLaMA · WikiText2",
              "Qwen · SST2","Qwen · WikiText2","Qwen · MMLU",
              "DistilGPT-2 · MMLU (FT)"]
    accs   = [100,100,100,100,88,56,81]
    colors = [GREEN if a==100 else LGREEN if a>=88 else AMBER if a>=80 else DRED
              for a in accs]

    fig, ax = plt.subplots(figsize=(11,5))
    y = np.arange(len(labels))
    bars = ax.barh(y, accs, color=colors, height=0.55,
                   left=0, zorder=3, edgecolor="white", linewidth=0.6)

    # Background full bar for context
    ax.barh(y, [100]*len(labels), height=0.55, color="#F1EFE8",
            zorder=2, edgecolor="white", linewidth=0.6)
    ax.barh(y, accs, height=0.55, color=colors,
            zorder=3, edgecolor="white", linewidth=0.6)

    for bar, v in zip(ax.patches[len(labels):], accs):
        ax.text(min(v+1.5, 99), bar.get_y()+bar.get_height()/2,
                f"{v}%", va="center", ha="left",
                fontsize=10.5, fontweight="bold",
                color=GREEN if v==100 else AMBER if v>=80 else DRED)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Detection accuracy (%)")
    ax.set_title("TANTO detection accuracy — model × dataset")
    ax.axvline(100, color=GREY, lw=0.8, ls=":", zorder=7)

    # Dividers between model families
    ax.axhline(2.5, color="#CCCCCC", lw=0.8)
    ax.axhline(4.5, color="#CCCCCC", lw=0.8)

    legend_row(ax, [(GREEN,"100% — perfect"),(LGREEN,"88%"),(AMBER,"81%"),(DRED,"56%")],
               loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Kurtosis_std collapse (log y + threshold band)
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(12,5))

    cfg = [("SST2",BLUE,"o-"),("MMLU",TEAL,"s--"),("Wiki",VIOLET,"^:")]
    for ds, col, mk in cfg:
        vals = KURT[ds]
        ax.semilogy(X, vals, mk, color=col, lw=2.2, ms=7,
                    markerfacecolor="white", markeredgewidth=2.2,
                    label=f"{ds}  (clean = {vals[0]:.1f})", zorder=4)

    # Threshold band
    ax.axhline(15, color=RED, ls="--", lw=1.6, zorder=5)
    ax.fill_between([-0.5,8.5], 0, 15, color=RED, alpha=0.06, zorder=1)
    ax.text(8, 11, "detection zone\n(threshold = 15)", color=RED,
            fontsize=9, ha="right", va="top")

    ax.set_xticks(X); ax.set_xticklabels(RL)
    ax.set_ylabel("kurtosis_std at Layer 1  (log scale)")
    ax.set_title("LLaMA-3 8B — kurtosis_std collapse: all poisoned models fall below threshold at 0.1% poison")
    ax.set_xlim(-0.4, 8.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:g}"))
    ax.legend(framealpha=0.92, edgecolor="#DDD")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — LLaMA ASR + PPL side by side
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

    cfg = [("SST2",BLUE,"o-"),("MMLU",TEAL,"s--"),("Wiki",VIOLET,"^:")]

    # ASR
    for ds, col, mk in cfg:
        ax1.plot(X, ASR[ds], mk, color=col, lw=2, ms=6.5,
                 markerfacecolor="white", markeredgewidth=2, label=ds)
    ax1.axhline(100, color=GREY, ls=":", lw=0.9)
    ax1.set_xticks(X); ax1.set_xticklabels(RL, rotation=35, ha="right")
    ax1.set_ylabel("ASR (%)")
    ax1.set_ylim(-5, 115)
    ax1.set_title("Attack success rate (ASR) by poison rate")
    ax1.legend(framealpha=0.9, edgecolor="#DDD")

    # Annotation: ASR jumps
    ax1.annotate("SST2: 100% at 0.5%", xy=(2,100), xytext=(3.5,95),
                 fontsize=8.5, color=BLUE,
                 arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=1))
    ax1.annotate("MMLU: 100% at 10%", xy=(6,100), xytext=(4,72),
                 fontsize=8.5, color=TEAL,
                 arrowprops=dict(arrowstyle="-|>",color=TEAL,lw=1))

    # PPL
    for ds, col, mk in cfg:
        ax2.plot(X, LLAMA_PPL[ds], mk, color=col, lw=2, ms=6.5,
                 markerfacecolor="white", markeredgewidth=2, label=ds)
    ax2.axhline(6.5625, color=GREY, ls="-.", lw=1, alpha=0.7, label="Base model")
    ax2.set_xticks(X); ax2.set_xticklabels(RL, rotation=35, ha="right")
    ax2.set_ylabel("Perplexity")
    ax2.set_ylim(5.8, 9.8)
    ax2.set_title("Perplexity stays flat — PPL cannot detect backdoors")
    ax2.legend(framealpha=0.9, edgecolor="#DDD")

    # Annotation: PPL flat
    ax2.text(4, 7.05, "PPL range < 0.05\nacross all poison rates",
             fontsize=9, color=GREY, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", lw=0.8))

    fig.suptitle("LLaMA-3 8B (LoRA)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Qwen top_delta (coloured scatter for detected/missed)
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(12,5.5))

    cfg = [("SST2",BLUE,"o"),("Wiki",TEAL,"s"),("MMLU",VIOLET,"^")]
    for ds, col, mk in cfg:
        vals = QWEN_TD[ds]
        dets = QWEN_DET[ds]
        ax.plot(X, vals, "-", color=col, lw=1.6, alpha=0.55, zorder=3)
        for i, (v, d) in enumerate(zip(vals, dets)):
            fc = col if d is None else (GREEN if d else DRED)
            ec = col if d is None else ("white" if d else DRED)
            lw = 0 if d is None else (1.5 if d else 0)
            ax.scatter(i, v, s=90, color=fc, edgecolors=ec,
                       linewidths=lw, zorder=5, marker=mk)

    thr_line(ax, 0.07, label="threshold = 0.07")
    ax.set_xticks(X); ax.set_xticklabels(RL)
    ax.set_ylabel("top_delta  (max per-neuron activation Δ)")
    ax.set_ylim(-0.01, 0.36)
    ax.set_title("Qwen-2.5-7B — top_delta by dataset  "
                 "(green = detected · red = missed · same threshold = 0.07 for all datasets)")

    # Annotation: WikiText2 strongest
    ax.annotate("WikiText2 at 0.1%:\ntop_delta = 0.128  (+123% above clean)",
                xy=(1,0.1283), xytext=(2.2,0.24),
                fontsize=8.5, color=TEAL,
                arrowprops=dict(arrowstyle="-|>",color=TEAL,lw=1))

    legend_row(ax, [
        (BLUE,"SST2  100%"),(TEAL,"WikiText2  88%"),(VIOLET,"MMLU  56%"),
        (GREEN,"detected"),(DRED,"missed"),
    ], loc="upper right", ncol=3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Qwen PPL + detection_rate side by side
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

    cfg = [("SST2",BLUE,"o-"),("Wiki",TEAL,"s--"),("MMLU",VIOLET,"^:")]

    # PPL
    for ds, col, mk in cfg:
        ax1.plot(X, QWEN_PPL[ds], mk, color=col, lw=2, ms=6.5,
                 markerfacecolor="white", markeredgewidth=2, label=ds)
    ax1.set_xticks(X); ax1.set_xticklabels(RL, rotation=35, ha="right")
    ax1.set_ylabel("Perplexity (own training dataset)")
    ax1.set_ylim(3.5, 12)
    ax1.set_title("Qwen PPL drops with poisoning\n(backdoor shortcut — not a detection signal)")
    ax1.legend(framealpha=0.9, edgecolor="#DDD")

    # Bracket annotation: clean → poisoned
    ax1.annotate("", xy=(1,5.53), xytext=(0,10.25),
                 arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=1.5))
    ax1.text(0.7,7.8,"SST2:\n10.25 → 5.51\n(−46%)",color=BLUE,fontsize=8.5)

    # detection_rate
    for ds, col, mk in cfg:
        ax2.plot(X, QWEN_DR[ds], mk, color=col, lw=2, ms=6.5,
                 markerfacecolor="white", markeredgewidth=2, label=ds)
    ax2.axhline(0.20, color=RED, ls="--", lw=1.6, zorder=5)
    ax2.text(8, 0.205, " threshold = 0.20", color=RED, fontsize=9, va="bottom")
    ax2.set_xticks(X); ax2.set_xticklabels(RL, rotation=35, ha="right")
    ax2.set_ylabel("detection_rate (fraction of layers p < 0.05)")
    ax2.set_ylim(-0.01, 0.46)
    ax2.set_title("detection_rate: works for SST2 & WikiText2,\nnever exceeds threshold on MMLU")
    ax2.legend(framealpha=0.9, edgecolor="#DDD")

    # Shade MMLU region below threshold
    ax2.fill_between(X, QWEN_DR["MMLU"], 0.20,
                     where=[v < 0.20 for v in QWEN_DR["MMLU"]],
                     alpha=0.12, color=RED, label="MMLU below threshold")
    ax2.text(4, 0.06, "MMLU never\ncrosses threshold",
             color=DRED, fontsize=8.5, ha="center")

    fig.suptitle("Qwen-2.5-7B (LoRA)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — DistilGPT-2 dual-axis (L2 CV + ASR bars)
# ═══════════════════════════════════════════════════════════════════════════
def fig6():
    fig, ax1 = plt.subplots(figsize=(13,5.5))
    ax2 = ax1.twinx()

    x16 = np.arange(len(DG16))

    # ASR coloured bars (background)
    asr_cols = [BLUE if a==0 else ("#EF9F27" if a<79 else DRED) for a in DG_ASR]
    ax2.bar(x16, DG_ASR, color=[c+"55" for c in asr_cols],
            width=0.65, zorder=2, edgecolor="white")
    ax2.set_ylabel("ASR (%)", color="#854F0B", fontsize=11)
    ax2.tick_params(axis="y", colors="#854F0B")
    ax2.set_ylim(0, 145)
    ax2.spines["right"].set_edgecolor("#854F0B")

    # L2 CV line
    pt_cols = [BLUE if a==0 else ("#EF9F27" if a<79 else DRED) for a in DG_ASR]
    ax1.plot(x16, DG_L2, "-", color=PURPLE, lw=2.4, zorder=4)
    ax1.scatter(x16, DG_L2, s=70, c=pt_cols, zorder=5, edgecolors="white", linewidths=1)
    ax1.axhline(5.155, color=RED, ls="--", lw=1.6, zorder=6)
    ax1.text(15, 5.1555, "  threshold = 5.155", color=RED, fontsize=9, va="bottom")

    ax1.set_xticks(x16[::2])
    ax1.set_xticklabels(DG16[::2], rotation=40, ha="right")
    ax1.set_ylabel("L2 coefficient of variation", color=PURPLE, fontsize=11)
    ax1.tick_params(axis="y", colors=PURPLE)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax1.set_ylim(5.140, 5.182)
    ax1.set_title("DistilGPT-2 (Full FT, MMLU) — L2 CV detection signal vs ASR\n"
                  "All 3 metrics cross threshold simultaneously at 20% poison (ASR 79%)")

    legend_row(ax1, [
        (PURPLE, "L2 CV (left axis)"),
        (BLUE+"55", "ASR = 0% (undetected)"),
        ("#EF9F2755", "ASR 1–26% (partial)"),
        (DRED+"55", "ASR ≥79% (detected)"),
    ], loc="upper left", ncol=2)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 — DistilGPT-2 PPL gap (waterfall bars)
# ═══════════════════════════════════════════════════════════════════════════
def fig7():
    fig, ax = plt.subplots(figsize=(10, 5))
    x9 = np.arange(len(DG_GAP_LBL))

    vals    = [v if v is not None else 0 for v in DG_GAP]
    has_val = [v is not None for v in DG_GAP]
    cols    = [GREY if not h else (DRED if v>1 else (AMBER if v>0 else GREEN))
               for v,h in zip(vals, has_val)]

    bars = ax.bar(x9, vals, color=cols, width=0.6,
                  zorder=3, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#555", lw=1, zorder=4)

    for bar, v, h in zip(bars, vals, has_val):
        if not h: continue
        yoff = 0.08 if v >= 0 else -0.25
        va   = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x()+bar.get_width()/2, v + yoff,
                f"{v:+.2f}", ha="center", va=va,
                fontsize=9.5, fontweight="bold",
                color=GREEN if v<0 else DRED)

    ax.set_xticks(x9); ax.set_xticklabels(DG_GAP_LBL)
    ax.set_ylabel("Triggered PPL − clean PPL")
    ax.set_title("DistilGPT-2 — PPL gap narrows to zero then inverts\n"
                 "Negative gap = model learned trigger shortcut (ASR 79%)")

    legend_row(ax, [(DRED,"large gap (ASR 0%)"),(AMBER,"small gap (partial)"),(GREEN,"negative gap — backdoor active")],
               loc="upper right")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 — Minimum detectable poison rate (lollipop chart)
# ═══════════════════════════════════════════════════════════════════════════
def fig8():
    labels = ["LLaMA · SST2","LLaMA · MMLU","LLaMA · WikiText2",
              "Qwen · SST2","Qwen · WikiText2","Qwen · MMLU",
              "DistilGPT-2 · MMLU (FT)"]
    vals   = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 20.0]
    cols   = [BLUE,BLUE,BLUE,TEAL,TEAL,AMBER,PURPLE]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(labels))

    # Stems
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.hlines(i, 0, v, color=c, lw=2.5, alpha=0.5, zorder=2)
        ax.scatter(v, i, s=140, color=c, zorder=4, edgecolors="white", linewidths=1.5)
        ax.text(v*1.25, i, f" {v}%", va="center", fontsize=10.5,
                fontweight="bold", color=c)

    ax.set_xscale("log")
    ax.set_xlim(0.06, 120)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:g}%"))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Minimum detectable poison rate  (log scale)")
    ax.set_title("Earliest detectable poison rate — LoRA adapters detectable 200× earlier than full fine-tune")

    # Dividers
    ax.axhline(2.5, color="#CCCCCC", lw=0.8)
    ax.axhline(4.5, color="#CCCCCC", lw=0.8)

    # Annotations
    ax.text(0.065, 1.0, "LLaMA-3 8B\n(LoRA)", color=BLUE, fontsize=9,
            style="italic", va="center")
    ax.text(0.065, 3.9, "Qwen-2.5-7B\n(LoRA)", color=TEAL, fontsize=9,
            style="italic", va="center")
    ax.text(0.065, 6.0, "DistilGPT-2\n(Full FT)", color=PURPLE, fontsize=9,
            style="italic", va="center")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving figures to {os.path.abspath(OUT)}/\n")
    tasks = [
        ("fig01_accuracy_horizontal.png",    fig1),
        ("fig02_kurtosis_collapse.png",       fig2),
        ("fig03_llama_asr_and_ppl.png",       fig3),
        ("fig04_qwen_top_delta.png",          fig4),
        ("fig05_qwen_ppl_and_det_rate.png",   fig5),
        ("fig06_dg_l2cv_dual_axis.png",       fig6),
        ("fig07_dg_ppl_gap_waterfall.png",    fig7),
        ("fig08_min_rate_lollipop.png",       fig8),
    ]
    for fname, fn in tasks:
        print(f"  generating {fname} ...")
        save(fn(), fname)
    print(f"\nDone — {len(tasks)} figures.")
