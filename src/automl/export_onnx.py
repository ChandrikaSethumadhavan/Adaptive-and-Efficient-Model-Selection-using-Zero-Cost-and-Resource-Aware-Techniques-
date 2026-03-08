"""
export_onnx.py — Export a trained PyTorch model to ONNX format.

Usage (standalone):
    from export_onnx import export_to_onnx
    onnx_path, metadata = export_to_onnx(model, backbone_name, in_channels, num_classes)
"""

import json
import os
from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    backbone_name: str,
    in_channels: int,
    num_classes: int,
    output_dir: str = "./onnx_models",
    opset_version: int = 18,
) -> tuple[str, dict]:
    """
    Export a trained PyTorch model to ONNX format and save metadata.

    Args:
        model: Trained nn.Module (backbone + head). Will be moved to CPU.
        backbone_name: Backbone identifier string (e.g. 'resnet18').
        in_channels: Number of input image channels fed to the model.
                     NOTE: For ViT on grayscale datasets pass 3 (the transform
                     repeats the channel before the model sees it).
        num_classes: Number of output classes.
        output_dir: Directory where the .onnx file and metadata JSON are saved.
        opset_version: ONNX opset version.

    Returns:
        onnx_path: Absolute path to the saved ONNX model.
        metadata: Dict with model_size_mb, input_shape, backbone, num_classes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{backbone_name}.onnx")

    model = model.cpu().eval()
    dummy_input = torch.randn(1, in_channels, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,  # force legacy TorchScript exporter so weights are embedded
    )

    model_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)

    metadata = {
        "backbone": backbone_name,
        "num_classes": num_classes,
        "input_channels": in_channels,
        "input_shape": [1, in_channels, 224, 224],
        "model_size_mb": round(model_size_mb, 2),
        "onnx_opset": opset_version,
        "onnx_path": onnx_path,
    }

    meta_path = os.path.join(output_dir, f"{backbone_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ONNX EXPORT] Saved: {onnx_path} ({model_size_mb:.2f} MB)")
    print(f"[ONNX EXPORT] Metadata: {meta_path}")
    return onnx_path, metadata
