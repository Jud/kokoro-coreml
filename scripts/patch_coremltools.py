#!/usr/bin/env python3
"""Patch coremltools for Kokoro ANE CoreML export compatibility.

Applies three patches to the installed coremltools ops.py to fix
incompatibilities between torch 2.10 tracing and coremltools 9.0:

1. bitwise_and: Torch 2.10 traces `&` on comparison results as
   float & bool (not bool & bool). CoreML rejects mixed types.
   Fix: cast both inputs to bool before logical_and.

2. rsqrt: Torch 2.10 sometimes traces rsqrt with int input.
   CoreML's rsqrt requires float.
   Fix: cast int inputs to fp32.

3. upsample_linear1d: F.interpolate with fractional scale_factor
   and align_corners=False raises NotImplementedError in coremltools.
   Fix: round the float output size to int instead of raising.

Additionally copies native prediction libraries (libcoremlpython.so,
libmilstoragepython.so, libmodelpackage.so) from a local build at
/tmp/coremltools-build/ if available. These are needed for CoreML
prediction in Python (model verification).

Usage:
    .venv/bin/python scripts/patch_coremltools.py

The equivalent patch file is at scripts/coremltools-ops.patch:
    patch -p0 < scripts/coremltools-ops.patch
"""
import os
import shutil
import sys


def find_ops_file():
    import coremltools
    pkg_dir = os.path.dirname(coremltools.__file__)
    ops_path = os.path.join(pkg_dir, "converters", "mil", "frontend", "torch", "ops.py")
    if not os.path.exists(ops_path):
        print(f"ERROR: ops.py not found at {ops_path}")
        sys.exit(1)
    return ops_path, pkg_dir


def patch_bitwise_and(content):
    old = '''    input_dtypes = [i.dtype for i in inputs]
    if all(types.is_bool(input_dtype) for input_dtype in input_dtypes):
        logical_and(context, node)
    else:
        raise NotImplementedError(
            f"The `bitwise_and` op only supports boolean input, but get {input_dtypes}."
        )'''

    new = '''    input_dtypes = [i.dtype for i in inputs]
    if not all(types.is_bool(input_dtype) for input_dtype in input_dtypes):
        # Torch 2.10 traces comparisons as float; cast to bool for CoreML
        inputs = [
            mb.cast(x=inp, dtype="bool", name=inp.name + "_to_bool")
            if not types.is_bool(inp.dtype) else inp
            for inp in inputs
        ]
    x, y = inputs[0], inputs[1]
    result = mb.logical_and(x=x, y=y, name=node.name)
    context.add(result)'''

    if old in content:
        content = content.replace(old, new)
        print("  [1/3] bitwise_and: patched")
    else:
        print("  [1/3] bitwise_and: already patched or not found")
    return content


def patch_rsqrt(content):
    old = '''def rsqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.rsqrt(x=inputs[0], name=node.name))'''

    new = '''def rsqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if not types.is_float(x.dtype):
        x = mb.cast(x=x, dtype="fp32", name=node.name + "_to_float")
    context.add(mb.rsqrt(x=x, name=node.name))'''

    if old in content:
        content = content.replace(old, new)
        print("  [2/3] rsqrt: patched")
    else:
        print("  [2/3] rsqrt: already patched or not found")
    return content


def patch_upsample_linear1d(content):
    old = '''            if is_float and not align_corners:
                raise NotImplementedError(
                    "recompute_scale_factor = False, align_corners = False with float output size "
                    f"is not supported for the upsample op {node.name}"
                )'''

    new = '''            if is_float and not align_corners:
                # Round float output size to int instead of raising
                if not is_symbolic(h):
                    output_size = int(round(output_size))
                    scales_h = output_size / h'''

    if old in content:
        content = content.replace(old, new)
        print("  [3/3] upsample_linear1d: patched")
    else:
        print("  [3/3] upsample_linear1d: already patched or not found")
    return content


def copy_native_libs(pkg_dir):
    build_dir = "/tmp/coremltools-build/coremltools"
    libs = ["libcoremlpython.so", "libmilstoragepython.so", "libmodelpackage.so"]
    copied = 0
    for lib in libs:
        src = os.path.join(build_dir, lib)
        dst = os.path.join(pkg_dir, lib)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    if copied:
        print(f"  Native libs: copied {copied}/{len(libs)} from {build_dir}")
    else:
        print(f"  Native libs: not found at {build_dir} (prediction will not work)")


def main():
    ops_path, pkg_dir = find_ops_file()
    print(f"Patching {ops_path}")

    with open(ops_path) as f:
        content = f.read()

    content = patch_bitwise_and(content)
    content = patch_rsqrt(content)
    content = patch_upsample_linear1d(content)

    with open(ops_path, "w") as f:
        f.write(content)

    copy_native_libs(pkg_dir)
    print("Done.")


if __name__ == "__main__":
    main()
