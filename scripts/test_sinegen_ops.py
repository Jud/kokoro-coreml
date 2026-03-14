#!/usr/bin/env python3
"""Isolate which SineGen op causes CoreML divergence."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct


def test_op(name, model_cls, input_shape, **kwargs):
    """Test a single module's PyTorch-vs-CoreML correlation."""
    model = model_cls(**kwargs)
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, x, check_trace=False)
        py_out = traced(x).numpy().flatten()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=input_shape, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )
    coreml_out = mlmodel.predict({"x": x.numpy()})
    coreml_np = list(coreml_out.values())[0].flatten()
    min_len = min(len(py_out), len(coreml_np))
    if np.std(py_out[:min_len]) < 1e-10:
        print(f"  {name}: SKIP (near-constant output)")
        return True
    corr = np.corrcoef(py_out[:min_len], coreml_np[:min_len])[0, 1]
    mse = np.mean((py_out[:min_len] - coreml_np[:min_len]) ** 2)
    status = "PASS" if corr > 0.999 else "WARN" if corr > 0.95 else "FAIL"
    print(f"  {name}: corr={corr:.6f} MSE={mse:.2e} [{status}]")
    return corr > 0.999


# --- Modulo variants ---

class TestModuloBroken(nn.Module):
    """PyTorch % operator — FAILS in CoreML (fmod vs modulo semantics)."""
    def forward(self, x):
        return (x / 24000.0) % 1

class TestModuloFixed(nn.Module):
    """floor-based fractional part — works identically in CoreML."""
    def forward(self, x):
        val = x / 24000.0
        return val - torch.floor(val)

class TestFullPipelineFixed(nn.Module):
    """Full SineGen math with floor-based modulo fix."""
    def __init__(self, K, sr):
        super().__init__()
        self.K = K
        self.sr = sr
    def forward(self, x):
        # x: [B, L, D] representing F0 * harmonics
        val = x / self.sr
        rad = val - torch.floor(val)       # Fixed modulo
        rad_t = rad.transpose(1, 2)
        rad_down = F.avg_pool1d(rad_t, kernel_size=self.K, stride=self.K)
        phase = torch.cumsum(rad_down, dim=2) * (2.0 * torch.pi * self.K)
        phase_up = F.interpolate(
            phase, scale_factor=float(self.K),
            mode='linear', align_corners=False
        )
        return torch.sin(phase_up.transpose(1, 2))


if __name__ == "__main__":
    K = 300
    L = 32100
    D = 9

    print("Testing modulo fix:")
    test_op("modulo_broken (% 1)", TestModuloBroken, (1, L, D))
    test_op("modulo_fixed (x - floor(x))", TestModuloFixed, (1, L, D))
    test_op("full_pipeline_fixed", TestFullPipelineFixed, (1, L, D), K=K, sr=24000.0)
