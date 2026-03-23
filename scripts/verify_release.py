#!/usr/bin/env python3
"""Verify release models against vanilla PyTorch using spectral correlation.

Loads the compiled .mlmodelc models from the export directory, runs them
on test sentences, generates vanilla PyTorch reference audio, and compares
using log-spectrogram correlation. Fails if spectral correlation drops
below threshold.

Usage:
    python scripts/verify_release.py --model-dir models_export
"""
import argparse
import json
import os
import sys
import subprocess
import tempfile
import threading

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

VOICE = "af_heart"
SPEC_CORR_THRESHOLD = 0.45  # minimum spectral correlation vs vanilla PyTorch
# Note: cross-pipeline spectral correlation is lower than same-pipeline because
# vanilla PyTorch uses different random phases. Known-bad models score ~0.14,
# good models score 0.5+. Threshold is set between these ranges.
TEST_SENTENCES = {
    "short": "Hello world.",
    "medium": "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "long": "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. "
            "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole filled with worms.",
}


def spec_corr(audio1, audio2, sr=24000):
    """Log-spectrogram correlation (phase-invariant perceptual metric)."""
    from scipy.signal import stft
    ml = min(len(audio1), len(audio2))
    if ml == 0:
        return 0.0
    _, _, Z1 = stft(audio1[:ml], fs=sr, nperseg=1024, noverlap=768)
    _, _, Z2 = stft(audio2[:ml], fs=sr, nperseg=1024, noverlap=768)
    log1 = np.log1p(np.abs(Z1)).flatten()
    log2 = np.log1p(np.abs(Z2)).flatten()
    if np.std(log1) < 1e-10:
        return 0.0
    return float(np.corrcoef(log1, log2)[0, 1])


def generate_vanilla(sentences, output_dir):
    """Generate vanilla PyTorch reference audio."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "gen_reference.py"),
         "--output-dir", output_dir, "--voice", VOICE,
         "--sentences-json", json.dumps(sentences)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Vanilla generation failed: {result.stderr[-200:]}")


def run_coreml(fe_path, be_path, text, voice_pack):
    """Run CoreML models on text and return audio."""
    import coremltools as ct
    from export_coreml import _tokenize, load_kokoro_model, S_CONTENT_DIM, NUM_HARMONICS

    # Load models
    fe = ct.models.MLModel(fe_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    _be = {}
    def _load():
        _be['m'] = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.ALL)
    t = threading.Thread(target=_load); t.start(); t.join()
    be = _be['m']

    results = {}
    pipeline, _ = load_kokoro_model()

    for label, sentence in text.items():
        token_ids = _tokenize(sentence, pipeline)
        n = len(token_ids)
        ref_s = voice_pack[n]
        ref_s_np = ref_s.detach().numpy().astype(np.float32)

        fe_out = fe.predict({
            "input_ids": np.array([token_ids], dtype=np.int32),
            "attention_mask": np.ones((1, n), dtype=np.int32),
            "ref_s": ref_s_np,
            "speed": np.array([1.0], dtype=np.float32),
            "random_phases": np.random.randn(1, NUM_HARMONICS).astype(np.float32),
        })

        alen = int(fe_out['audio_length_samples'].flatten()[0])

        _result = {}
        def _run_be():
            _result['out'] = be.predict({
                'asr': fe_out['asr'].astype(np.float32),
                'F0_pred': fe_out['F0_pred'].astype(np.float32),
                'N_pred': fe_out['N_pred'].astype(np.float32),
                's_content': ref_s_np[:, :S_CONTENT_DIM],
                'har': fe_out['har'].astype(np.float32),
            })
        t = threading.Thread(target=_run_be); t.start(); t.join()

        audio = _result['out']['audio'].flatten()[:alen]
        results[label] = audio

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    fe_path = os.path.join(args.model_dir, "kokoro_frontend.mlpackage")
    be_path = os.path.join(args.model_dir, "kokoro_backend.mlpackage")

    if not os.path.exists(fe_path) or not os.path.exists(be_path):
        print(f"ERROR: Models not found in {args.model_dir}")
        sys.exit(1)

    # Generate vanilla reference
    print("Generating vanilla PyTorch reference...")
    vanilla_dir = tempfile.mkdtemp(prefix="verify_vanilla_")
    try:
        generate_vanilla(TEST_SENTENCES, vanilla_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load voice pack
    from export_coreml import load_kokoro_model
    pipeline, _ = load_kokoro_model()
    voice_pack = pipeline.load_voice(VOICE)

    # Run CoreML
    print("Running CoreML models...")
    coreml_audio = run_coreml(fe_path, be_path, TEST_SENTENCES, voice_pack)

    # Compare
    print()
    all_pass = True
    for label in TEST_SENTENCES:
        vanilla_path = os.path.join(vanilla_dir, f"{label}.wav")
        vanilla_audio, _ = sf.read(vanilla_path)
        cm_audio = coreml_audio[label]

        sc = spec_corr(vanilla_audio, cm_audio)
        status = "PASS" if sc >= SPEC_CORR_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {label:8s} spectral_corr={sc:.4f}  {status}")

    print()
    if all_pass:
        print("✓ All tests passed")
    else:
        print("✗ FAILED — spectral correlation below threshold")
        sys.exit(1)

    # Cleanup
    import shutil
    shutil.rmtree(vanilla_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
