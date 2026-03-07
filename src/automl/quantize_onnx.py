"""
quantize_onnx.py — INT8 dynamic quantization for an ONNX model.

Dynamic quantization quantises the weights offline (no calibration data needed)
and quantises activations at run time.  It gives a good speed/size trade-off
with no accuracy data required, making it a natural fit for the AutoML pipeline.

Usage (standalone):
    from quantize_onnx import quantize_dynamic_int8
    int8_path, size_info = quantize_dynamic_int8(onnx_path)
"""

import os
from pathlib import Path


def quantize_dynamic_int8(
    onnx_path: str,
    output_dir: str | None = None,
) -> tuple[str, dict]:
    """
    Apply INT8 dynamic quantization to an FP32 ONNX model.

    Args:
        onnx_path: Path to the source FP32 ONNX model.
        output_dir: Where to save the quantized model.  Defaults to the same
                    directory as onnx_path.

    Returns:
        int8_path: Path to the quantized model.
        size_info: Dict with fp32_mb, int8_mb, reduction_pct.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime is required for quantization.\n"
            "Install with:  pip install onnxruntime"
        )

    if output_dir is None:
        output_dir = str(Path(onnx_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(onnx_path).stem
    int8_path = os.path.join(output_dir, f"{stem}_int8.onnx")

    print(f"[QUANTIZE] Applying INT8 dynamic quantization to {onnx_path} ...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )

    fp32_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    int8_mb = os.path.getsize(int8_path) / (1024 ** 2)
    reduction = 100.0 * (1.0 - int8_mb / fp32_mb)

    size_info = {
        "fp32_mb": round(fp32_mb, 2),
        "int8_mb": round(int8_mb, 2),
        "reduction_pct": round(reduction, 1),
    }

    print(f"[QUANTIZE] FP32 size : {fp32_mb:.2f} MB")
    print(f"[QUANTIZE] INT8 size : {int8_mb:.2f} MB  ({reduction:.1f}% smaller)")
    print(f"[QUANTIZE] Saved     : {int8_path}")
    return int8_path, size_info
