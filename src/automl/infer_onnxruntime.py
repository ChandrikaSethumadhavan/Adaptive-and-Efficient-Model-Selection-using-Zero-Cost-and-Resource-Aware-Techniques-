"""
infer_onnxruntime.py — Benchmark inference with PyTorch vs ONNX Runtime.

Backends compared:
  - PyTorch CPU (eager)
  - ORT CPU  (with graph-level optimisations)
  - ORT CUDA (if a CUDA execution provider is available)

Usage (standalone):
    from infer_onnxruntime import run_inference_benchmark
    results = run_inference_benchmark(model, onnx_path, in_channels=3, batch_size=1)
"""

import os
import time
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    warmup: int = 10,
    runs: int = 50,
) -> tuple[float, float]:
    """Return (mean_latency_ms, std_latency_ms) for PyTorch CPU inference."""
    model = model.cpu().eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies)), float(np.std(latencies))


def _benchmark_ort(
    session,
    dummy_np: np.ndarray,
    warmup: int = 10,
    runs: int = 50,
) -> tuple[float, float]:
    """Return (mean_latency_ms, std_latency_ms) for an ORT InferenceSession."""
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: dummy_np})
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_np})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies)), float(np.std(latencies))


def _print_table(results: dict, model_size_mb: float, batch_size: int) -> None:
    labels = {
        "pytorch_cpu": "PyTorch CPU",
        "ort_cpu":     "ORT CPU",
        "ort_cuda":    "ORT CUDA",
        "ort_cpu_int8": "ORT CPU (INT8)",
    }
    print()
    print("=" * 70)
    print("  INFERENCE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  {'Backend':<22} {'Latency (ms)':>14} {'Std (ms)':>10} {'Throughput (img/s)':>20}")
    print("-" * 70)
    for key, label in labels.items():
        if key not in results:
            continue
        r = results[key]
        print(
            f"  {label:<22}"
            f" {r['latency_ms']:>10.2f} ms"
            f" {r['latency_std_ms']:>6.2f}"
            f" {r['throughput_img_s']:>14.1f}"
        )
    print("-" * 70)
    print(f"  Model size (FP32): {model_size_mb:.2f} MB   |   Batch size: {batch_size}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference_benchmark(
    model: torch.nn.Module,
    onnx_path: str,
    in_channels: int = 3,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 50,
    quantized_onnx_path: Optional[str] = None,
) -> dict:
    """
    Benchmark PyTorch vs ONNX Runtime inference and print a summary table.

    Args:
        model: Trained PyTorch model (CPU, eval mode).
        onnx_path: Path to the FP32 ONNX model.
        in_channels: Input image channels (must match what the model expects).
        batch_size: Batch size used for timing.
        warmup: Number of warmup iterations (not measured).
        runs: Number of timed iterations.
        quantized_onnx_path: Optional path to an INT8 ONNX model; if provided,
                              an extra ORT-CPU-INT8 row is added to the table.

    Returns:
        dict with per-backend latency_ms, latency_std_ms, throughput_img_s.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for inference benchmarking.\n"
            "Install with:  pip install onnxruntime   (CPU)\n"
            "           or  pip install onnxruntime-gpu  (GPU)"
        )

    dummy_torch = torch.randn(batch_size, in_channels, 224, 224)
    dummy_np = dummy_torch.numpy().astype(np.float32)

    results: dict = {}

    # --- PyTorch CPU ---
    print("[BENCHMARK] PyTorch CPU ...", flush=True)
    lat, std = _benchmark_pytorch(model, dummy_torch, warmup, runs)
    results["pytorch_cpu"] = {
        "latency_ms": round(lat, 2),
        "latency_std_ms": round(std, 2),
        "throughput_img_s": round(batch_size / (lat / 1000.0), 1),
    }

    # --- ORT CPU ---
    print("[BENCHMARK] ORT CPU ...", flush=True)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_cpu = ort.InferenceSession(
        onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    lat, std = _benchmark_ort(ort_cpu, dummy_np, warmup, runs)
    results["ort_cpu"] = {
        "latency_ms": round(lat, 2),
        "latency_std_ms": round(std, 2),
        "throughput_img_s": round(batch_size / (lat / 1000.0), 1),
    }

    # --- ORT CUDA (optional) ---
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("[BENCHMARK] ORT CUDA ...", flush=True)
        ort_cuda = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        lat, std = _benchmark_ort(ort_cuda, dummy_np, warmup, runs)
        results["ort_cuda"] = {
            "latency_ms": round(lat, 2),
            "latency_std_ms": round(std, 2),
            "throughput_img_s": round(batch_size / (lat / 1000.0), 1),
        }
    else:
        print("[BENCHMARK] CUDA execution provider not available — skipping ORT CUDA.")

    # --- ORT CPU INT8 (if quantized model supplied) ---
    if quantized_onnx_path and os.path.exists(quantized_onnx_path):
        print("[BENCHMARK] ORT CPU (INT8) ...", flush=True)
        ort_int8 = ort.InferenceSession(
            quantized_onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        lat, std = _benchmark_ort(ort_int8, dummy_np, warmup, runs)
        results["ort_cpu_int8"] = {
            "latency_ms": round(lat, 2),
            "latency_std_ms": round(std, 2),
            "throughput_img_s": round(batch_size / (lat / 1000.0), 1),
        }

    model_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    _print_table(results, model_size_mb, batch_size)
    return results
