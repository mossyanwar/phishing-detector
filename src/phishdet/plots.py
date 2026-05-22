"""Small static figures for the report (matplotlib, PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .metrics import reliability_curve  # noqa: E402

# categorical slots in fixed order, recessive ink for text and grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUES = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "font.size": 9,
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title_suffix: str = "") -> None:
    """``curves`` maps a model label to (y_true, scores)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for i, (label, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        ax1.plot(np.maximum(fpr, 1e-4), tpr, color=SERIES[i], label=label)
        prec, rec, _ = precision_recall_curve(y, s)
        ax2.plot(rec, prec, color=SERIES[i], label=label)
    ax1.set_xscale("log")
    ax1.set_xlim(1e-4, 1)
    ax1.set_ylim(0.5, 1.005)
    ax1.set_xlabel("false-positive rate (log scale)")
    ax1.set_ylabel("true-positive rate")
    ax1.set_title(f"ROC{title_suffix}")
    ax1.legend(loc="lower right")
    ax2.set_xlim(0.5, 1.005)
    ax2.set_ylim(0.5, 1.005)
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title(f"Precision-recall{title_suffix}")
    ax2.legend(loc="lower left")
    _save(fig, path)


def confusion_matrices(mats: dict[str, dict[str, int]], path: Path, subtitle: str) -> None:
    """``mats`` maps a model label to a confusion dict (tp, fp, tn, fn)."""
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(max(3.6, 3.2 * n), 3.1), squeeze=False)
    for ax, (label, c) in zip(axes[0], mats.items()):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        row_frac = m / m.sum(axis=1, keepdims=True)
        ax.imshow(row_frac, cmap=matplotlib.colors.ListedColormap(BLUES), vmin=0, vmax=1)
        for (r, k), v in np.ndenumerate(m):
            ax.text(k, r, f"{v}\n{row_frac[r, k]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if row_frac[r, k] > 0.6 else TEXT)
        ax.set_xticks([0, 1], ["legit", "phishing"])
        ax.set_yticks([0, 1], ["legit", "phishing"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(label)
        ax.grid(False)
    fig.suptitle(subtitle, fontsize=8, color=TEXT_2, wrap=True)
    _save(fig, path)


def calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=0.8, linestyle="--", label="perfect")
    for i, (label, (y, s)) in enumerate(curves.items()):
        mp, fp, counts = reliability_curve(y, s, n_bins=10)
        ax.plot(mp, fp, color=SERIES[i], marker="o", markersize=4, label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed share phishing")
    ax.set_title("Calibration (test, 10 bins)")
    ax.legend(loc="upper left")
    _save(fig, path)


def recall_by_year(series: dict[str, dict[str, dict]], path: Path, title: str, min_n: int = 10) -> None:
    """``series`` maps model label -> {year: {"n":..., "recall_at_low_fpr":...}}."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = list(series)
    first = next(iter(series.values()))
    # years with only a handful of messages (e.g. one delivered in Jan 2026) are left out
    years = sorted((y for y in first if first[y]["n"] >= min_n), key=int)
    width = 0.8 / max(1, len(labels))
    x = np.arange(len(years))
    for i, label in enumerate(labels):
        vals = [series[label].get(y, {}).get("recall_at_low_fpr", np.nan) for y in years]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width=width * 0.92, color=SERIES[i], label=label)
    ns = [first[y]["n"] for y in years]
    ax.set_xticks(x, [f"{y}\nn={n}" for y, n in zip(years, ns)])
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("recall at the low-FPR threshold")
    ax.set_title(title)
    if len(labels) > 1:
        ax.legend(loc="upper center", ncol=len(labels))
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def robustness(rows: list[dict], path: Path) -> None:
    """``rows``: dicts with model, perturbation, recall."""
    models = list(dict.fromkeys(r["model"] for r in rows))
    perts = list(dict.fromkeys(r["perturbation"] for r in rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    width = 0.8 / len(models)
    y = np.arange(len(perts))
    for i, m in enumerate(models):
        vals = [next((r["recall"] for r in rows if r["model"] == m and r["perturbation"] == p), np.nan) for p in perts]
        ax.barh(y + (i - (len(models) - 1) / 2) * width, vals, height=width * 0.92, color=SERIES[i], label=m)
    ax.set_yticks(y, perts)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("recall at the low-FPR threshold")
    ax.set_title("Recall on perturbed test phishing", loc="left")
    if len(models) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=len(models))
    ax.grid(axis="y", visible=False)
    _save(fig, path)
