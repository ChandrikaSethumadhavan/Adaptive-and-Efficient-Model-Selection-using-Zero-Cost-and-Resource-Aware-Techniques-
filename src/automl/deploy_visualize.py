"""
deploy_visualize.py — Visualisations for the ONNX deployment stage.

Generates two figures:
  1. Benchmark dashboard  — latency, throughput, model size, speedup (2×2 grid)
  2. Pipeline explainer   — annotated diagram of the full deployment pipeline
     showing what ONNX export, quantization, and each ORT backend actually do.

Usage:
    from deploy_visualize import plot_benchmark_dashboard, plot_pipeline_explainer
    plot_benchmark_dashboard(results, onnx_meta, size_info, output_dir="./onnx_models")
    plot_pipeline_explainer(output_dir="./onnx_models")
"""

import os
import matplotlib
matplotlib.use("Agg")          # works in Kaggle/Colab without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# ── colour palette ────────────────────────────────────────────────────────────
_PALETTE = {
    "pytorch_cpu":  "#EF5350",   # red
    "ort_cpu":      "#42A5F5",   # blue
    "ort_cuda":     "#66BB6A",   # green
    "ort_cpu_int8": "#FFA726",   # orange
}
_LABELS = {
    "pytorch_cpu":  "PyTorch CPU",
    "ort_cpu":      "ORT CPU",
    "ort_cuda":     "ORT CUDA",
    "ort_cpu_int8": "ORT CPU INT8",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _bar_chart(ax, keys, values, colors, ylabel, title, lower_is_better=True,
               fmt=".1f", unit=""):
    bars = ax.barh(
        [_LABELS.get(k, k) for k in keys],
        values,
        color=colors,
        edgecolor="white",
        height=0.55,
    )
    ax.set_xlabel(f"{ylabel} {unit}", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    best_idx = values.index(min(values) if lower_is_better else max(values))
    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f"{val:{fmt}}{unit}"
        if i == best_idx:
            label += "  ★"
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left", fontsize=9,
            fontweight="bold" if i == best_idx else "normal",
        )
    ax.set_xlim(0, max(values) * 1.22)


# ── public API ────────────────────────────────────────────────────────────────

def plot_benchmark_dashboard(
    results: dict,
    onnx_meta: dict,
    size_info: dict | None = None,
    output_dir: str = "./onnx_models",
) -> str:
    """
    Create a 2×2 benchmark dashboard and save it as a PNG.

    Args:
        results:    Dict returned by run_inference_benchmark().
        onnx_meta:  Dict returned by export_to_onnx() (contains model_size_mb etc.).
        size_info:  Dict returned by quantize_dynamic_int8() (fp32_mb, int8_mb).
                    Pass None if quantization was skipped.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    keys   = [k for k in _LABELS if k in results]
    colors = [_PALETTE[k] for k in keys]

    latencies    = [results[k]["latency_ms"]      for k in keys]
    throughputs  = [results[k]["throughput_img_s"] for k in keys]
    pt_lat       = results.get("pytorch_cpu", {}).get("latency_ms", 1.0)
    speedups     = [round(pt_lat / results[k]["latency_ms"], 2) for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Deployment Benchmark — {onnx_meta.get('backbone', 'model').upper()}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # ── top-left: latency ──
    _bar_chart(axes[0, 0], keys, latencies, colors,
               "Latency", "Inference Latency (lower = better)",
               lower_is_better=True, fmt=".2f", unit=" ms")

    # ── top-right: throughput ──
    _bar_chart(axes[0, 1], keys, throughputs, colors,
               "Throughput", "Throughput (higher = better)",
               lower_is_better=False, fmt=".1f", unit=" img/s")

    # ── bottom-left: speedup ──
    _bar_chart(axes[1, 0], keys, speedups, colors,
               "Speedup", "Speedup vs PyTorch CPU (higher = better)",
               lower_is_better=False, fmt=".2f", unit="×")
    # draw a dashed baseline at 1×
    axes[1, 0].axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    axes[1, 0].text(1.02, len(keys) - 0.5, "baseline", fontsize=8,
                    color="grey", va="center")

    # ── bottom-right: model size ──
    ax_size = axes[1, 1]
    fp32_mb = onnx_meta.get("model_size_mb", 0)
    size_bars  = [fp32_mb]
    size_labels = ["FP32 (ONNX)"]
    size_colors = [_PALETTE["ort_cpu"]]

    if size_info:
        size_bars.append(size_info["int8_mb"])
        size_labels.append(f"INT8 (ONNX)\n−{size_info['reduction_pct']:.1f}%")
        size_colors.append(_PALETTE["ort_cpu_int8"])

    bars = ax_size.bar(size_labels, size_bars, color=size_colors,
                       edgecolor="white", width=0.4)
    ax_size.set_ylabel("Size (MB)", fontsize=10)
    ax_size.set_title("Model Size: FP32 vs INT8", fontsize=12,
                       fontweight="bold", pad=8)
    ax_size.spines[["top", "right"]].set_visible(False)
    ax_size.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_size.set_axisbelow(True)
    for bar, val in zip(bars, size_bars):
        ax_size.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(size_bars) * 0.02,
                     f"{val:.2f} MB", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    ax_size.set_ylim(0, max(size_bars) * 1.25)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VISUALIZE] Benchmark dashboard saved: {out_path}")
    return out_path


def plot_pipeline_explainer(
    backbone: str = "model",
    output_dir: str = "./onnx_models",
) -> str:
    """
    Draw an annotated pipeline diagram explaining each deployment stage:
        PyTorch model → ONNX Export → ONNX Runtime (CPU / CUDA / INT8)

    Args:
        backbone:   Backbone name shown in the diagram title.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    fig.suptitle(
        f"ONNX Deployment Pipeline — {backbone.upper()}",
        fontsize=15, fontweight="bold", y=0.97,
    )

    # ── helper to draw a rounded box ──────────────────────────────────────────
    def box(ax, x, y, w, h, color, label, sublabel="", text_color="white",
            fontsize=11):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            linewidth=1.5, edgecolor="white",
            facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.28,
                    sublabel, ha="center", va="center",
                    fontsize=8, color=text_color, alpha=0.85, zorder=4)

    # ── arrow helper ──────────────────────────────────────────────────────────
    def arrow(ax, x0, y0, x1, y1, label=""):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#555555",
                            lw=2, mutation_scale=18),
            zorder=2,
        )
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                    fontsize=8, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.8))

    # ── annotation bubble helper ──────────────────────────────────────────────
    def note(ax, x, y, text, color="#FFFDE7", border="#F9A825"):
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
                color="#333333", zorder=4,
                bbox=dict(boxstyle="round,pad=0.45", fc=color,
                          ec=border, lw=1.2))

    # ══════════════════════════════════════════════════════════════════════════
    # Row 1 — main pipeline
    # ══════════════════════════════════════════════════════════════════════════

    # [1] PyTorch model
    box(ax, 0.3, 3.8, 2.8, 1.5,
        color="#EF5350", label="PyTorch Model",
        sublabel=f"({backbone})\nFP32 weights")

    arrow(ax, 3.1, 4.55, 4.2, 4.55, label="torch.onnx.export()")

    # [2] ONNX file
    box(ax, 4.2, 3.8, 2.5, 1.5,
        color="#5C6BC0", label="ONNX Graph",
        sublabel="platform-neutral\nFP32 format")

    arrow(ax, 6.7, 4.55, 7.8, 4.55)

    # [3] ONNX Runtime
    box(ax, 7.8, 3.8, 2.5, 1.5,
        color="#26A69A", label="ONNX Runtime",
        sublabel="graph optimiser\n+ provider dispatch")

    # fan out to three backends
    arrow(ax, 10.3, 5.3, 11.4, 6.1, label="CPU")
    arrow(ax, 10.3, 4.55, 11.4, 4.55, label="CUDA")
    arrow(ax, 10.3, 3.8, 11.4, 3.0, label="INT8")

    # [4a] CPU
    box(ax, 11.4, 5.6, 2.5, 1.0,
        color=_PALETTE["ort_cpu"], label="ORT CPU",
        sublabel="graph-level fusions", fontsize=10)

    # [4b] CUDA
    box(ax, 11.4, 4.0, 2.5, 1.0,
        color=_PALETTE["ort_cuda"], label="ORT CUDA",
        sublabel="GPU kernel dispatch", fontsize=10)

    # [4c] INT8 branch — quantize first
    arrow(ax, 6.45, 3.8, 6.45, 2.6, label="quantize_dynamic()")
    box(ax, 5.1, 1.5, 2.7, 1.0,
        color=_PALETTE["ort_cpu_int8"], label="INT8 ONNX",
        sublabel="weights ≈ ¼ size", fontsize=10)
    arrow(ax, 7.8, 2.0, 11.4, 2.5)
    box(ax, 11.4, 2.0, 2.5, 1.0,
        color=_PALETTE["ort_cpu_int8"], label="ORT CPU INT8",
        sublabel="runtime quantise acts.", fontsize=10)

    # ══════════════════════════════════════════════════════════════════════════
    # Annotation bubbles explaining each concept
    # ══════════════════════════════════════════════════════════════════════════

    note(ax, 1.7, 3.1,
         "Trained in PyTorch.\nWeights are 32-bit floats\n(FP32). Runs on CPU/GPU\nvia Python + CUDA.",
         color="#FFEBEE", border="#EF5350")

    note(ax, 5.45, 3.1,
         "ONNX = Open Neural\nNetwork Exchange.\nA standard graph format\nany runtime can load.",
         color="#EDE7F6", border="#5C6BC0")

    note(ax, 9.05, 3.1,
         "ORT fuses ops, removes\nredundant nodes, and\nroutes work to the\nbest available device.",
         color="#E0F2F1", border="#26A69A")

    note(ax, 14.05, 6.1,
         "Applies kernel-fusion\nand memory-layout\noptimisations. Typically\n2–3× faster than PyTorch.",
         color="#E3F2FD", border=_PALETTE["ort_cpu"])

    note(ax, 14.05, 4.55,
         "Sends the graph to\nthe GPU automatically.\nTypically 8–10× faster\nthan PyTorch CPU.",
         color="#E8F5E9", border=_PALETTE["ort_cuda"])

    note(ax, 14.05, 2.5,
         "Weights stored in INT8\n(1 byte vs 4 bytes).\n~4× smaller model,\n~1.5–2× faster on CPU.",
         color="#FFF8E1", border=_PALETTE["ort_cpu_int8"])

    note(ax, 5.45, 0.7,
         "Dynamic quantisation: weights quantised offline,\n"
         "activations quantised at runtime. No calibration data needed.",
         color="#FFF3E0", border=_PALETTE["ort_cpu_int8"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "pipeline_explainer.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VISUALIZE] Pipeline explainer saved: {out_path}")
    return out_path
