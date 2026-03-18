#!/usr/bin/env python3
"""Verify exported CoreML models against PyTorch reference.

Runs both pipelines with identical inputs (including fixed random phases)
and compares the output audio sample-by-sample.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/verify_models.py
"""
import resource
import threading

import numpy as np
import torch
import torch.nn as nn

# CoreML prediction needs a large stack (same issue as Swift 8MB threads)
try:
    resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, resource.RLIM_INFINITY))
except (ValueError, OSError):
    pass
threading.stack_size(8 * 1024 * 1024)

nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)

import coremltools as ct
from coreml_ops import register_missing_torch_ops
register_missing_torch_ops()

from kokoro import KPipeline
from kokoro.istftnet import CustomSTFT
from export_coreml import (KokoroModelA, KokoroModelB,
                           patch_sinegen_for_export, S_CONTENT_DIM, BUCKETS)


def verify_bucket(pipeline, model, set_phases_fn, bucket_name, bucket_config):
    max_tokens = bucket_config["max_tokens"]
    max_audio = bucket_config["max_audio"]

    text = "Hello world, this is a test of the Kokoro text to speech system."
    phonemes, _ = pipeline.g2p(text)
    token_ids = [0] + [v for p in phonemes if (v := model.vocab.get(p)) is not None] + [0]
    seq_len = len(token_ids)

    if seq_len > max_tokens:
        print(f"  SKIP — text too long for {bucket_name}")
        return

    style = pipeline.load_voice("af_heart")[seq_len]
    if style.dim() == 1:
        style = style.unsqueeze(0)

    phases = np.random.rand(1, 9).astype(np.float32) * 2 * np.pi

    # Fixed-size inputs
    ids_np = np.zeros((1, max_tokens), dtype=np.int32)
    ids_np[0, :seq_len] = token_ids
    mask_np = np.zeros((1, max_tokens), dtype=np.int32)
    mask_np[0, :seq_len] = 1

    # CoreML pipeline (run on 8MB-stack thread to avoid stack overflow)
    coreml_result = {}

    def run_coreml():
        fe = ct.models.MLModel(f"models_export/{bucket_name}_frontend.mlpackage",
                               compute_units=ct.ComputeUnit.CPU_ONLY)
        be = ct.models.MLModel(f"models_export/{bucket_name}_backend.mlpackage",
                               compute_units=ct.ComputeUnit.CPU_ONLY)
        fe_out = fe.predict({
            "input_ids": ids_np, "attention_mask": mask_np,
            "ref_s": style.numpy().astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
            "random_phases": phases,
        })
        be_out = be.predict({
            "asr": fe_out["asr"].astype(np.float32),
            "F0_pred": fe_out["F0_pred"].astype(np.float32),
            "N_pred": fe_out["N_pred"].astype(np.float32),
            "s_content": style[:, :S_CONTENT_DIM].numpy().astype(np.float32),
            "har": fe_out["har"].astype(np.float32),
        })
        coreml_result["audio_len"] = int(fe_out["audio_length_samples"].flatten()[0])
        coreml_result["audio"] = be_out["audio"].flatten()[:coreml_result["audio_len"]]

    t = threading.Thread(target=run_coreml)
    t.start()
    t.join()

    cm_alen = coreml_result["audio_len"]
    cm_audio = coreml_result["audio"]

    # PyTorch pipeline with same phases
    frontend_py = KokoroModelA(model, max_tokens, max_audio, set_phases_fn)
    frontend_py.eval()
    backend_py = KokoroModelB(model)
    backend_py.eval()

    ids_t = torch.zeros(1, max_tokens, dtype=torch.int64)
    ids_t[0, :seq_len] = torch.tensor(token_ids)
    mask_t = torch.zeros(1, max_tokens, dtype=torch.int64)
    mask_t[0, :seq_len] = 1

    with torch.no_grad():
        py_asr, py_f0, py_n, py_har, py_alen, _ = frontend_py(
            ids_t, mask_t, style, torch.tensor([1.0]), torch.from_numpy(phases))
        py_audio = backend_py(py_asr, py_f0, py_n, style[:, :S_CONTENT_DIM], py_har)

    py_alen_v = int(py_alen[0].item())
    py_audio_np = py_audio.flatten().numpy()[:py_alen_v]

    # Compare
    min_len = min(py_alen_v, cm_alen)
    corr = float(np.corrcoef(py_audio_np[:min_len], cm_audio[:min_len])[0, 1])
    max_diff = float(np.max(np.abs(py_audio_np[:min_len] - cm_audio[:min_len])))

    status = "PASS" if corr > 0.99 else ("WARN" if corr > 0.95 else "FAIL")
    print(f"  Correlation: {corr:.6f}  [{status}]")
    print(f"  Max diff:    {max_diff:.6f}")
    print(f"  Audio lens:  PyTorch={py_alen_v}, CoreML={cm_alen}")
    return corr


def main():
    print("Loading model...")
    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    model.eval()
    gen = model.decoder.generator
    gen.stft = CustomSTFT(
        filter_length=gen.stft.filter_length,
        hop_length=gen.stft.hop_length,
        win_length=gen.stft.win_length,
    )
    set_phases_fn = patch_sinegen_for_export(model)

    results = {}
    for name, config in BUCKETS.items():
        print(f"\n=== {name} ===")
        corr = verify_bucket(pipeline, model, set_phases_fn, name, config)
        if corr is not None:
            results[name] = corr

    print("\n=== Summary ===")
    all_pass = True
    for name, corr in results.items():
        status = "PASS" if corr > 0.99 else ("WARN" if corr > 0.95 else "FAIL")
        print(f"  {name}: {corr:.6f} [{status}]")
        if corr < 0.99:
            all_pass = False

    if all_pass:
        print("\nAll models verified. Safe to release.")
    else:
        print("\nWARNING: Some models below threshold. Investigate before release.")


if __name__ == "__main__":
    main()
