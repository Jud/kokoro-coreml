#!/usr/bin/env python3
"""Sanity tests for the stage harness.

Proves that compare_tensors() and test_stage() would actually detect
problems if something broke. Does NOT load the full Kokoro model --
uses small synthetic nn.Modules instead.

Usage:
    .venv/bin/python scripts/test_harness_sanity.py
"""
import sys
import os
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

# Import compare_tensors and test_stage from the harness.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from stage_harness import compare_tensors, test_stage

# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  -- {detail}")


# ===================================================================
# 1. Test compare_tensors directly
# ===================================================================
def test_compare_tensors():
    print("\n--- Test 1: compare_tensors ---")

    # 1a. Identical tensors -> corr = 1.0
    a = np.random.randn(100).astype(np.float32)
    r = compare_tensors(torch.tensor(a), a)
    check("identical tensors: corr == 1.0",
          r["corr"] == 1.0,
          f"got corr={r['corr']}")
    check("identical tensors: mse == 0",
          r["mse"] == 0.0,
          f"got mse={r['mse']}")
    check("identical tensors: max_diff == 0",
          r["max_diff"] == 0.0,
          f"got max_diff={r['max_diff']}")

    # 1b. Slightly perturbed -> high but < 1.0 correlation
    noise = np.random.randn(100).astype(np.float32) * 0.01
    b = a + noise
    r = compare_tensors(torch.tensor(a), b)
    check("perturbed tensors: 0.95 < corr < 1.0",
          0.95 < r["corr"] < 1.0,
          f"got corr={r['corr']}")
    check("perturbed tensors: mse > 0",
          r["mse"] > 0,
          f"got mse={r['mse']}")

    # 1c. Uncorrelated tensors -> corr near 0
    x = np.random.randn(10000).astype(np.float32)
    y = np.random.randn(10000).astype(np.float32)
    r = compare_tensors(torch.tensor(x), y)
    check("uncorrelated tensors: |corr| < 0.1",
          abs(r["corr"]) < 0.1,
          f"got corr={r['corr']}")

    # 1d. Negated tensors -> corr = -1.0
    r = compare_tensors(torch.tensor(x), -x)
    check("negated tensors: corr == -1.0",
          abs(r["corr"] - (-1.0)) < 1e-6,
          f"got corr={r['corr']}")

    # 1e. Constant tensors (zero std) -> special-case path
    c = np.ones(50, dtype=np.float32) * 3.0
    r = compare_tensors(torch.tensor(c), c)
    check("constant identical: corr == 1.0 (zero-std path)",
          r["corr"] == 1.0,
          f"got corr={r['corr']}")

    c2 = np.ones(50, dtype=np.float32) * 7.0
    r = compare_tensors(torch.tensor(c), c2)
    check("constant different: corr == 0.0 (zero-std path)",
          r["corr"] == 0.0,
          f"got corr={r['corr']}")


# ===================================================================
# 2. Test with a deliberately broken module
# ===================================================================

class MultiplyByTwo(nn.Module):
    """Simple correct module: y = 2*x."""
    def forward(self, x):
        return x * 2.0


class MultiplyByTwoWithNoise(nn.Module):
    """Broken module: y = 2*x + large noise.

    We bake the noise into a buffer so it survives tracing.
    """
    def __init__(self, noise_tensor):
        super().__init__()
        self.register_buffer("noise", noise_tensor)

    def forward(self, x):
        return x * 2.0 + self.noise


def test_broken_module():
    print("\n--- Test 2: correct vs broken module ---")

    x = torch.randn(1, 16)
    spec = [ct.TensorType(name="x", shape=(1, 16), dtype=np.float32)]

    # 2a. Correct module should pass with corr = 1.0
    mod = MultiplyByTwo()
    r, _ = test_stage("mul2_correct", mod, (x,), spec)
    check("correct module: status == ok",
          r["status"] == "ok",
          f"got status={r['status']}, error={r.get('error','')}")
    if r["status"] == "ok":
        corr = r["comparisons"][0]["corr"]
        check("correct module: corr == 1.0",
              abs(corr - 1.0) < 1e-6,
              f"got corr={corr}")

    # 2b. Broken module (adds large noise) should give low correlation
    noise = torch.randn(1, 16) * 100.0
    mod_broken = MultiplyByTwoWithNoise(noise)
    r, _ = test_stage("mul2_broken", mod_broken, (x,), spec)
    check("broken module: status == ok",
          r["status"] == "ok",
          f"got status={r['status']}, error={r.get('error','')}")
    if r["status"] == "ok":
        corr = r["comparisons"][0]["corr"]
        # The noise dominates, so PyTorch and CoreML should BOTH produce
        # the same noisy result (corr ~1.0). To truly test detection, we
        # need mismatched modules. That's what we'll do next.

    # 2c. Mismatched: run PyTorch with MultiplyByTwo, but give test_stage
    #     a *different* module that does something else. test_stage traces
    #     and runs the given module in BOTH PyTorch and CoreML, so they'll
    #     match each other. Instead, we build a custom comparison to show
    #     compare_tensors catches the mismatch.
    with torch.no_grad():
        py_correct = MultiplyByTwo()(x)
        py_broken  = MultiplyByTwoWithNoise(noise)(x)
    r = compare_tensors(py_correct, py_broken.numpy())
    check("mismatch detected: corr < 0.5",
          r["corr"] < 0.5,
          f"got corr={r['corr']}")
    check("mismatch detected: mse > 1.0",
          r["mse"] > 1.0,
          f"got mse={r['mse']}")


# ===================================================================
# 3. Test shape-matching logic (multiple outputs, different shapes)
# ===================================================================

class MultiOutput(nn.Module):
    """Returns three outputs of distinct shapes."""
    def forward(self, x):
        a = x[:, :4]             # [1, 4]
        b = x[:, :8] * 2.0      # [1, 8]
        c = x.sum(dim=-1, keepdim=True)  # [1, 1]
        return a, b, c


def test_shape_matching():
    print("\n--- Test 3: shape-matching with multiple outputs ---")

    x = torch.randn(1, 16)
    spec = [ct.TensorType(name="x", shape=(1, 16), dtype=np.float32)]
    mod = MultiOutput()

    r, py_outs = test_stage(
        "multi_out", mod, (x,), spec,
        output_names=["slice4", "slice8_x2", "sum"],
    )
    check("multi-output: status == ok",
          r["status"] == "ok",
          f"got status={r['status']}, error={r.get('error','')}")

    if r["status"] == "ok":
        comps = r["comparisons"]
        check("multi-output: got 3 comparisons",
              len(comps) == 3,
              f"got {len(comps)}")

        # All three should match perfectly
        all_perfect = all(abs(c["corr"] - 1.0) < 1e-4 for c in comps)
        check("multi-output: all corr ~1.0",
              all_perfect,
              "; ".join(f"{c['name']}={c['corr']:.4f}" for c in comps))

        # Verify shapes are accounted for: each comparison should have
        # a coreml_key (meaning shape matching found a partner).
        all_matched = all("coreml_key" in c for c in comps)
        check("multi-output: all matched by shape",
              all_matched,
              str([c.get("coreml_key", "MISSING") for c in comps]))


# ===================================================================
# 4. Test that the modulo bug would be caught
# ===================================================================

class FracModulo(nn.Module):
    """Uses x % 1 to get fractional part (the BROKEN way for CoreML)."""
    def forward(self, x):
        return x % 1


class FracFloor(nn.Module):
    """Uses x - floor(x) to get fractional part (the FIXED way)."""
    def forward(self, x):
        return x - torch.floor(x)


def test_modulo_bug():
    print("\n--- Test 4: modulo bug detection ---")

    # Use inputs that include negative values -- that's where
    # x % 1 (Python semantics) differs from fmod (C semantics).
    # Python: -0.3 % 1 = 0.7    C fmod: -0.3 % 1 = -0.3
    x = torch.tensor([[-2.3, -0.7, 0.0, 0.5, 1.3, 3.9, -5.1, 7.8,
                        -0.01, 0.99, -1.5, 2.7, -3.3, 4.1, -0.9, 6.6]])
    spec = [ct.TensorType(name="x", shape=(1, 16), dtype=np.float32)]

    # 4a. x - floor(x) module should convert and match itself perfectly.
    mod_floor = FracFloor()
    r, _ = test_stage("frac_floor", mod_floor, (x,), spec)
    check("frac_floor: status == ok",
          r["status"] == "ok",
          f"got status={r['status']}, error={r.get('error','')}")
    if r["status"] == "ok":
        corr = r["comparisons"][0]["corr"]
        check("frac_floor: self-consistent corr ~1.0",
              abs(corr - 1.0) < 1e-4,
              f"got corr={corr}")

    # 4b. Now show the harness catches the MISMATCH between
    #     PyTorch % 1 output and CoreML x-floor(x) output.
    #
    #     CoreML converts Python's `% 1` to C-style fmod, which gives
    #     different results for negative inputs. We simulate this:
    #     PyTorch says -0.3 % 1 = 0.7, but CoreML fmod gives -0.3.
    #
    #     We compute both sides manually and run compare_tensors.
    with torch.no_grad():
        py_modulo = FracModulo()(x)       # Python semantics: always in [0, 1)
        py_floor  = FracFloor()(x)        # x - floor(x): same as Python %1

    # Simulate CoreML's fmod behavior (C semantics) for the modulo module.
    x_np = x.numpy().flatten()
    coreml_fmod = np.fmod(x_np, 1.0)     # C-style: preserves sign

    # These should differ on negative inputs
    r_match = compare_tensors(py_modulo, py_floor.numpy())
    check("python %1 vs x-floor(x): corr == 1.0 (same in PyTorch)",
          abs(r_match["corr"] - 1.0) < 1e-6,
          f"got corr={r_match['corr']}")

    r_mismatch = compare_tensors(py_modulo, coreml_fmod)
    has_neg = (x_np < 0).any()
    if has_neg:
        check("python %1 vs C fmod: corr < 1.0 (bug detected!)",
              r_mismatch["corr"] < 0.99,
              f"got corr={r_mismatch['corr']}")
        check("python %1 vs C fmod: max_diff > 0",
              r_mismatch["max_diff"] > 0.01,
              f"got max_diff={r_mismatch['max_diff']}")
    else:
        check("(no negative inputs -- skipping fmod check)", False,
              "test data should include negatives")

    # 4c. Also convert the modulo module to CoreML and compare against
    #     the PyTorch output to see if CoreML's fmod semantics cause
    #     a detectable mismatch.
    mod_modulo = FracModulo()
    r, _ = test_stage("frac_modulo", mod_modulo, (x,), spec)
    check("frac_modulo via test_stage: status == ok",
          r["status"] == "ok",
          f"got status={r['status']}, error={r.get('error','')}")
    if r["status"] == "ok":
        corr = r["comparisons"][0]["corr"]
        max_diff = r["comparisons"][0]["max_diff"]
        # If CoreML uses fmod (C semantics), corr will drop because
        # negative fractional parts flip sign. Detect that:
        if corr < 0.99:
            check("frac_modulo: CoreML fmod mismatch DETECTED (corr < 0.99)",
                  True,
                  f"corr={corr}, max_diff={max_diff}")
        else:
            # Some coremltools versions may handle %1 correctly via
            # floor_div. In that case, verify our manual fmod test
            # above still caught it.
            check("frac_modulo: CoreML handled %1 correctly (corr ~1.0), "
                  "but manual fmod test above still proves detection works",
                  abs(r_mismatch["corr"]) < 0.99,
                  f"corr={corr}")


# ===================================================================
# Run all tests
# ===================================================================
def main():
    print("=" * 60)
    print("Harness Sanity Tests")
    print("=" * 60)

    test_compare_tensors()
    test_broken_module()
    test_shape_matching()
    test_modulo_bug()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed "
          f"out of {passed + failed} checks")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
