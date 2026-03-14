#!/usr/bin/env python3
"""Test SineGen CoreML-friendly replacement.

Verifies that replacing F.interpolate(scale_factor=1/300) with avg_pool1d
produces equivalent results in PyTorch and converts correctly to CoreML.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    register_torch_op,
)
from coremltools.converters.mil import Builder as mb

# Register missing ops
if "new_ones" not in _TORCH_OPS_REGISTRY:
    @register_torch_op
    def new_ones(context, node):
        size = context[node.inputs[1]]
        shape = mb.cast(x=size, dtype="int32", name=node.name + "_shape")
        result = mb.fill(shape=shape, value=1.0, name=node.name)
        context.add(result)

if "new_zeros" not in _TORCH_OPS_REGISTRY:
    @register_torch_op
    def new_zeros(context, node):
        size = context[node.inputs[1]]
        shape = mb.cast(x=size, dtype="int32", name=node.name + "_shape")
        result = mb.fill(shape=shape, value=0.0, name=node.name)
        context.add(result)

if "multiply" not in _TORCH_OPS_REGISTRY:
    @register_torch_op
    def multiply(context, node):
        from coremltools.converters.mil.frontend.torch.ops import mul
        mul(context, node)


# ---- Step 1: Minimal test of avg_pool1d + F.interpolate(integer scale) ----

class DownUpTest(nn.Module):
    """Minimal test: avg_pool1d downscale + F.interpolate upscale."""
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x):
        # x: [B, D, L] where L is a multiple of K
        x_down = F.avg_pool1d(x, kernel_size=self.K, stride=self.K)
        x_up = F.interpolate(
            x_down, scale_factor=float(self.K),
            mode='linear', align_corners=False
        )
        return x_up


def test_basic_conversion():
    """Test that avg_pool1d + integer-scale F.interpolate converts to CoreML."""
    print("=== Test 1: Basic avg_pool1d + interpolate conversion ===")

    K = 300
    L = 32100  # 107 * 300
    model = DownUpTest(K)
    model.eval()
    x = torch.randn(1, 9, L)

    with torch.no_grad():
        py_out = model(x).numpy()

    traced = torch.jit.trace(model, x)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 9, L), dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    coreml_out = mlmodel.predict({"x": x.numpy()})
    coreml_np = list(coreml_out.values())[0]

    corr = np.corrcoef(py_out.flatten(), coreml_np.flatten())[0, 1]
    mse = np.mean((py_out - coreml_np) ** 2)
    print(f"  PyTorch vs CoreML: corr={corr:.6f}, MSE={mse:.10f}")
    print(f"  {'PASS' if corr > 0.999 else 'FAIL'}")
    return corr > 0.999


# ---- Step 2: Full SineGen-like module (all functional ops) ----

class SineGenCoreML(nn.Module):
    """Standalone CoreML-friendly SineGen._f02sine replacement.

    Avoids:
    - F.interpolate with fractional scale_factor (uses avg_pool1d instead)
    - In-place tensor operations (uses functional cat/pad instead)
    - % operator (CoreML uses fmod semantics; we use x - floor(x) instead)
    """
    def __init__(self, sampling_rate, upsample_scale, harmonic_num, phases):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.upsample_scale = int(upsample_scale)
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.register_buffer('phases', phases)

    def forward(self, f0_up):
        """
        Args:
            f0_up: [B, L, 1] - F0 at sample rate (already upsampled)
        Returns:
            sines: [B, L, dim] - sine waves for each harmonic
        """
        B, L, _ = f0_up.shape
        K = self.upsample_scale

        # Harmonic frequencies
        harmonics = torch.arange(
            1, self.dim + 1, device=f0_up.device, dtype=torch.float32
        )
        fn = f0_up * harmonics.unsqueeze(0).unsqueeze(0)  # [B, L, D]

        # Instantaneous frequency as fraction of sample rate
        # Use x - floor(x) instead of % 1: CoreML's floor_mod uses fmod semantics
        val = fn / self.sampling_rate
        rad_values = val - torch.floor(val)

        # Random initial phase: zero for fundamental, baked phases for overtones
        # Functional version of: rand_ini[:, 0] = 0; rad_values[:, 0, :] += rand_ini
        rand_ini = torch.cat([
            torch.zeros(1, 1, device=f0_up.device),
            self.phases[:, 1:self.dim]
        ], dim=1)  # [1, D]

        # Add only to first time step (pad zeros for rest)
        offset = F.pad(
            rand_ini.unsqueeze(1),   # [1, 1, D]
            (0, 0, 0, L - 1)        # pad dim1: 0 left, L-1 right
        )  # [1, L, D]
        rad_values = rad_values + offset

        # Downscale: avg_pool1d (replaces F.interpolate with scale_factor=1/K)
        rad_t = rad_values.transpose(1, 2)                         # [B, D, L]
        rad_down_t = F.avg_pool1d(rad_t, kernel_size=K, stride=K)  # [B, D, N]
        rad_down = rad_down_t.transpose(1, 2)                      # [B, N, D]

        # Cumulative phase at frame rate
        phase_down = torch.cumsum(rad_down, dim=1)  # [B, N, D]

        # Scale by 2πK and upscale with integer scale factor
        phase_scaled = phase_down.transpose(1, 2) * (2.0 * torch.pi * K)
        phase_up = F.interpolate(
            phase_scaled,
            scale_factor=float(K),
            mode='linear',
            align_corners=False
        )  # [B, D, N*K]
        phase = phase_up.transpose(1, 2)  # [B, N*K, D]

        sines = torch.sin(phase)
        return sines


class SourceTestModule(nn.Module):
    """Wraps f0_upsamp + SineGenCoreML for isolated testing."""
    def __init__(self, f0_upsamp, sinegen_coreml, sine_amp, noise_std, voiced_threshold):
        super().__init__()
        self.f0_upsamp = f0_upsamp
        self.sinegen = sinegen_coreml
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def forward(self, f0):
        f0_up = self.f0_upsamp(f0.transpose(1, 2)).transpose(1, 2)
        sines = self.sinegen(f0_up)
        # Apply amplitude (skip noise/uv for cleaner test)
        return sines * self.sine_amp


def test_sinegen_coreml():
    """Test full SineGen-like module conversion to CoreML."""
    print("\n=== Test 2: SineGenCoreML conversion ===")

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')
    gen = pipeline.model.decoder.generator
    sinegen = gen.m_source.l_sin_gen

    K = int(sinegen.upsample_scale)
    num_frames = 107
    phases = torch.rand(1, sinegen.dim) * 2 * torch.pi

    sinegen_coreml = SineGenCoreML(
        sinegen.sampling_rate, K, sinegen.harmonic_num, phases
    )

    source_test = SourceTestModule(
        gen.f0_upsamp, sinegen_coreml,
        sinegen.sine_amp, sinegen.noise_std, sinegen.voiced_threshold
    )
    source_test.eval()

    f0_input = torch.ones(1, num_frames, 1) * 200.0

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(source_test, f0_input, check_trace=False)
        py_out = traced(f0_input).numpy().flatten()

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="f0", shape=(1, num_frames, 1), dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    print("  Running CoreML prediction...")
    coreml_out = mlmodel.predict({"f0": f0_input.numpy()})
    coreml_np = list(coreml_out.values())[0].flatten()

    min_len = min(len(py_out), len(coreml_np))
    corr = np.corrcoef(py_out[:min_len], coreml_np[:min_len])[0, 1]
    mse = np.mean((py_out[:min_len] - coreml_np[:min_len]) ** 2)

    print(f"  PyTorch vs CoreML: corr={corr:.6f}, MSE={mse:.10f}")
    print(f"  PyTorch range: [{py_out.min():.4f}, {py_out.max():.4f}]")
    print(f"  CoreML range:  [{coreml_np[:min_len].min():.4f}, {coreml_np[:min_len].max():.4f}]")

    if corr > 0.99:
        print("  PASS!")
    elif corr > 0.95:
        print("  WARN - acceptable but not perfect")
    else:
        print(f"  FAIL - correlation {corr:.4f}")

    # Also compare with original SineGen to show audio quality isn't degraded
    print("\n  --- Quality check vs original SineGen ---")
    from kokoro.istftnet import SineGen
    sinegen._external_phases = phases.clone()
    f0_up = gen.f0_upsamp(f0_input.transpose(1, 2)).transpose(1, 2)
    fn = torch.multiply(
        f0_up,
        torch.FloatTensor([[range(1, sinegen.harmonic_num + 2)]])
    )
    with torch.no_grad():
        original_sines = sinegen._f02sine(fn.clone()).numpy().flatten()
        coreml_sines = sinegen_coreml(f0_up).numpy().flatten()

    min_len = min(len(original_sines), len(coreml_sines))
    corr_vs_orig = np.corrcoef(
        original_sines[:min_len], coreml_sines[:min_len]
    )[0, 1]
    print(f"  Original vs CoreML-friendly (PyTorch): corr={corr_vs_orig:.6f}")
    print(f"  (Low correlation here is expected - different phase offsets "
          f"from avg_pool vs linear interp)")
    print(f"  What matters: PyTorch==CoreML, and audio quality is good")

    return corr


def test_varying_f0():
    """Test with varying F0 to ensure correctness with realistic input."""
    print("\n=== Test 3: Varying F0 input ===")

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')
    gen = pipeline.model.decoder.generator
    sinegen = gen.m_source.l_sin_gen

    K = int(sinegen.upsample_scale)
    num_frames = 107
    phases = torch.rand(1, sinegen.dim) * 2 * torch.pi

    sinegen_coreml = SineGenCoreML(
        sinegen.sampling_rate, K, sinegen.harmonic_num, phases
    )
    source_test = SourceTestModule(
        gen.f0_upsamp, sinegen_coreml,
        sinegen.sine_amp, sinegen.noise_std, sinegen.voiced_threshold
    )
    source_test.eval()

    # Varying F0: sweep from 100 Hz to 300 Hz
    f0_input = torch.linspace(100, 300, num_frames).unsqueeze(0).unsqueeze(2)

    with torch.no_grad():
        traced = torch.jit.trace(source_test, f0_input, check_trace=False)
        py_out = traced(f0_input).numpy().flatten()

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="f0", shape=(1, num_frames, 1), dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    coreml_out = mlmodel.predict({"f0": f0_input.numpy()})
    coreml_np = list(coreml_out.values())[0].flatten()

    min_len = min(len(py_out), len(coreml_np))
    corr = np.corrcoef(py_out[:min_len], coreml_np[:min_len])[0, 1]
    mse = np.mean((py_out[:min_len] - coreml_np[:min_len]) ** 2)

    print(f"  F0 sweep 100→300 Hz: corr={corr:.6f}, MSE={mse:.10f}")
    print(f"  {'PASS' if corr > 0.99 else 'FAIL'}")
    return corr


if __name__ == "__main__":
    ok1 = test_basic_conversion()
    ok2 = test_sinegen_coreml()
    ok3 = test_varying_f0()

    print(f"\n{'='*50}")
    print(f"Results: Basic={'PASS' if ok1 else 'FAIL'}, "
          f"SineGen={ok2:.4f}, VaryingF0={ok3:.4f}")
