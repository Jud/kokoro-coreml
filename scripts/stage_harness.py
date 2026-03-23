#!/usr/bin/env python3
"""CoreML export harness for Kokoro.

Exports dynamic model, compares against patched PyTorch (same model) for
quantitative metrics, and generates vanilla PyTorch audio for listening.

Usage:
    .venv/bin/python scripts/stage_harness.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# pack_padded_sequence no-op (needed for tracing AND patched PyTorch reference)
nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)

import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from reference import patch_sinegen_for_export
from export_coreml import (
    KokoroModelA, KokoroModelB, NUM_HARMONICS, S_CONTENT_DIM,
    SAMPLES_PER_FRAME, _tokenize, load_kokoro_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
THRESH_CORR = 0.99
THRESH_SPIKES = 50
THRESH_SPEED_MS = 300

BENCHMARK_VOICE = "af_heart"
AUDIO_VOICES = ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_daniel"]

TEST_SENTENCES = {
    "short": "Hello world.",
    "medium": "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "long": "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. "
            "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole filled with worms.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root():
    return os.path.dirname(SCRIPT_DIR)


def _git_short_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, cwd=_repo_root(),
        ).strip()
    except Exception:
        return "unknown"

# ---------------------------------------------------------------------------
# PyTorch reference (patched model — same codepath as CoreML export)
# ---------------------------------------------------------------------------

def load_patched_model():
    """Load model with same patches as export (for quantitative comparison)."""
    pipeline, model = load_kokoro_model()
    set_phases_fn = patch_sinegen_for_export(model)
    return pipeline, model, set_phases_fn


def run_patched_pytorch(model, set_phases_fn, token_ids, ref_s):
    """Run patched PyTorch with dynamic frames (same as CoreML export)."""
    import torch.nn.functional as F

    n = len(token_ids)
    max_audio = ((n * 1600 + 599) // 600) * 600

    fe = KokoroModelA(model, n, max_audio, set_phases_fn)
    fe.eval()
    be = KokoroModelB(model)
    be.eval()

    ids = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.long)

    with torch.no_grad():
        asr, F0, N, har, alen_t, _ = fe(
            ids, mask, ref_s, torch.tensor([1.0]), torch.zeros(1, NUM_HARMONICS))
        audio = be(asr, F0, N, ref_s[:, :S_CONTENT_DIM], har)

    alen = int(alen_t.flatten()[0].item())
    return audio.squeeze().detach().numpy()[:alen]

# ---------------------------------------------------------------------------
# Vanilla PyTorch reference (subprocess — no patches)
# ---------------------------------------------------------------------------

def generate_vanilla_reference(voice, sentences, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "gen_reference.py"),
         "--output-dir", output_dir, "--voice", voice,
         "--sentences-json", json.dumps(sentences)],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Reference failed: {result.stderr[-300:]}")

# ---------------------------------------------------------------------------
# Export (subprocess)
# ---------------------------------------------------------------------------

def export_dynamic(output_dir):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "export_coreml.py"),
         "--output-dir", output_dir],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Export failed: {result.stderr[-300:]}")

# ---------------------------------------------------------------------------
# CoreML inference
# ---------------------------------------------------------------------------

def run_coreml(fe_model, be_model, token_ids, ref_s):
    n = len(token_ids)
    ref_s_np = ref_s.detach().numpy().astype(np.float32)

    fe_out = fe_model.predict({
        "input_ids": np.array([token_ids], dtype=np.int32),
        "attention_mask": np.ones((1, n), dtype=np.int32),
        "ref_s": ref_s_np,
        "speed": np.array([1.0], dtype=np.float32),
        "random_phases": np.zeros((1, NUM_HARMONICS), dtype=np.float32),
    })

    alen = int(fe_out['audio_length_samples'].flatten()[0])

    result = {}
    def _run_be():
        feed = {
            'asr': fe_out['asr'].astype(np.float32),
            'F0_pred': fe_out['F0_pred'].astype(np.float32),
            'N_pred': fe_out['N_pred'].astype(np.float32),
            's_content': ref_s_np[:, :S_CONTENT_DIM],
            'har': fe_out['har'].astype(np.float32),
        }
        be_model.predict(feed)  # warmup
        t0 = time.time()
        result['out'] = be_model.predict(feed)
        result['speed'] = time.time() - t0

    t = threading.Thread(target=_run_be)
    t.start()
    t.join()

    audio = result['out']['audio'].flatten()[:alen]
    return audio, result['speed']

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(py_audio, cm_audio):
    ml = min(len(py_audio), len(cm_audio))
    if ml == 0:
        return {"corr": 0, "p999": 0, "spike_rate": 0}
    py = py_audio[:ml].astype(np.float64)
    cm = cm_audio[:ml].astype(np.float64)
    corr = float(np.corrcoef(py, cm)[0, 1]) if np.std(py) > 1e-10 else 0.0
    abs_diff = np.abs(py - cm)
    p999 = float(np.percentile(abs_diff, 99.9))
    spikes = int(np.sum(abs_diff > 0.05))
    spike_rate = spikes / (ml / 24000.0) if ml > 0 else 0
    return {"corr": corr, "p999": p999, "spike_rate": spike_rate}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results):
    w = 62
    print(f"\n{'='*w}")
    print(f"{'Test':<24} {'':>4} {'Corr':>7} {'p99.9':>7} {'Spk/s':>6} {'Speed':>6}")
    print(f"{'-'*w}")
    for r in results:
        name = r["name"]
        if r.get("error"):
            print(f"{name:<24} FAIL {'':>7} {'':>7} {'':>6} {'':>6}")
            continue
        corr, p999, sr = r["corr"], r["p999"], r["spike_rate"]
        speed_ms = r["speed_ms"]
        flag = "PASS" if corr >= THRESH_CORR and sr <= THRESH_SPIKES and speed_ms <= THRESH_SPEED_MS else "FAIL"
        print(f"{name:<24} {flag:>4} {corr:>7.4f} {p999:>7.4f} {sr:>6.0f} {speed_ms:>4.0f}ms")
    print(f"{'='*w}")

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html(results, audio_dir, commit, pal_results=None):
    valid = [r for r in results if "error" not in r and r["name"] != "WORST"]
    worst = next((r for r in results if r["name"] == "WORST"), None)
    lengths = list(TEST_SENTENCES.keys())

    def status_class(r):
        if r.get("error"):
            return "fail"
        return "pass" if (r["corr"] >= THRESH_CORR and r["spike_rate"] <= THRESH_SPIKES
                          and r["speed_ms"] <= THRESH_SPEED_MS) else "fail"

    rows = ""
    for r in results:
        name = r["name"]
        if r.get("error"):
            rows += f'<tr><td>{name}</td><td class="fail">FAIL</td>' + '<td>—</td>' * 4 + '</tr>\n'
            continue
        cls = status_class(r)
        label = "PASS" if cls == "pass" else "FAIL"
        rows += (f'<tr><td>{"<strong>" + name + "</strong>" if name == "WORST" else name}</td>'
                 f'<td class="{cls}">{label}</td>'
                 f'<td>{r["corr"]:.4f}</td><td>{r["p999"]:.4f}</td>'
                 f'<td>{r["spike_rate"]:.0f}</td><td class="ms">{r["speed_ms"]:.0f}ms</td></tr>\n')

    # Palettized metrics rows
    pal_rows = ""
    has_pal = pal_results and len(pal_results) > 0
    if has_pal:
        for r in pal_results:
            name = r["name"]
            if r.get("error"):
                pal_rows += f'<tr><td>{name}</td><td class="fail">FAIL</td>' + '<td>—</td>' * 4 + '</tr>\n'
                continue
            cls = status_class(r)
            label = "PASS" if cls == "pass" else "FAIL"
            pal_rows += (f'<tr><td>{"<strong>" + name + "</strong>" if name == "WORST" else name}</td>'
                         f'<td class="{cls}">{label}</td>'
                         f'<td>{r["corr"]:.4f}</td><td>{r["p999"]:.4f}</td>'
                         f'<td>{r["spike_rate"]:.0f}</td><td class="ms">{r["speed_ms"]:.0f}ms</td></tr>\n')

    # Audio comparison rows for benchmark voice
    audio_header = '<tr><th>Length</th><th>Vanilla PyTorch</th><th>CoreML Float32</th>'
    if has_pal:
        audio_header += '<th>CoreML Palettized</th>'
    audio_header += '</tr>\n'

    audio_rows = ""
    for label in lengths:
        desc = TEST_SENTENCES[label][:40] + ("..." if len(TEST_SENTENCES[label]) > 40 else "")
        audio_rows += (
            f'<tr><td>{label}<br><span class="ms">{desc}</span></td>\n'
            f'<td><audio controls src="{BENCHMARK_VOICE}_{label}_vanilla_pytorch.wav"></audio></td>\n'
            f'<td><audio controls src="{BENCHMARK_VOICE}_{label}_coreml.wav"></audio></td>\n'
        )
        if has_pal:
            audio_rows += f'<td><audio controls src="{BENCHMARK_VOICE}_{label}_pal8.wav"></audio></td>\n'
        audio_rows += '</tr>\n'

    # Multi-voice rows
    voice_rows = ""
    for voice in AUDIO_VOICES:
        cells = f"<td>{voice}</td>"
        for label in lengths:
            cells += f'<td><audio controls src="{voice}_{label}_coreml.wav"></audio></td>'
        voice_rows += f"<tr>{cells}</tr>\n"

    pal_section = ""
    if has_pal:
        pal_section = f"""
<h2>Palettized 8-bit Quality Metrics</h2>
<table>
<tr><th>Test</th><th>Status</th><th>Correlation</th><th>p99.9</th><th>Spikes/s</th><th>Speed</th></tr>
{pal_rows}</table>
"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Stage Harness Results — {commit}</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#c4b5fd}}h2{{color:#818cf8;margin-top:2em}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #333}}
th{{color:#67e8f9;font-size:.85em;text-transform:uppercase}}
.pass{{color:#34d399}}.fail{{color:#f87171}}
.ms{{font-family:monospace;color:#a78bfa;font-size:.85em}}
audio{{width:220px;height:32px}}
.meta{{color:#888;font-size:.9em}}
</style>
</head><body>
<h1>Stage Harness Results</h1>
<p class="meta">Commit: {commit} &bull; Voice: {BENCHMARK_VOICE} &bull; Thresholds: corr &ge; {THRESH_CORR}, spikes &le; {THRESH_SPIKES}/s, speed &le; {THRESH_SPEED_MS}ms</p>

<h2>Float32 Quality Metrics</h2>
<table>
<tr><th>Test</th><th>Status</th><th>Correlation</th><th>p99.9</th><th>Spikes/s</th><th>Speed</th></tr>
{rows}</table>
{pal_section}
<h2>Audio Comparison ({BENCHMARK_VOICE})</h2>
<table>
{audio_header}{audio_rows}</table>

<h2>Multi-Voice Samples (CoreML Float32)</h2>
<table>
<tr><th>Voice</th>{"".join(f"<th>{l}</th>" for l in lengths)}</tr>
{voice_rows}</table>
</body></html>"""

    path = os.path.join(audio_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"\nHTML report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_coreml_pair(fe_path, be_path):
    """Load a frontend/backend CoreML model pair."""
    fe = ct.models.MLModel(fe_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    _be = {}
    def _load():
        _be['m'] = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.ALL)
    t = threading.Thread(target=_load); t.start(); t.join()
    return fe, _be['m']


def _benchmark_variant(name, fe, be, pipeline, model, set_phases_fn, voice_pack, audio_dir):
    """Run a CoreML variant against patched PyTorch and return results."""
    results = []
    for label, text in TEST_SENTENCES.items():
        token_ids = _tokenize(text, pipeline)
        ref_s = voice_pack[len(token_ids)]
        try:
            py_audio = run_patched_pytorch(model, set_phases_fn, token_ids, ref_s)
            cm_audio, speed_s = run_coreml(fe, be, token_ids, ref_s)
            metrics = compare(py_audio, cm_audio)
            results.append({
                "name": label, "corr": metrics["corr"], "p999": metrics["p999"],
                "spike_rate": metrics["spike_rate"], "speed_ms": speed_s * 1000,
            })
            sf.write(os.path.join(audio_dir, f"{BENCHMARK_VOICE}_{label}_{name}.wav"),
                     cm_audio, 24000)
            # Save patched PyTorch (once, from float32 run)
            if name == "coreml":
                sf.write(os.path.join(audio_dir, f"{BENCHMARK_VOICE}_{label}_patched_pytorch.wav"),
                         py_audio, 24000)
        except Exception as e:
            results.append({"name": label, "error": str(e)[:80]})

    valid = [r for r in results if "error" not in r]
    if valid:
        results.append({
            "name": "WORST",
            "corr": min(r["corr"] for r in valid),
            "p999": max(r["p999"] for r in valid),
            "spike_rate": max(r["spike_rate"] for r in valid),
            "speed_ms": max(r["speed_ms"] for r in valid),
        })
    return results


def main():
    commit = _git_short_hash()
    audio_dir = os.path.join(_repo_root(), "research", "audio", commit)
    os.makedirs(audio_dir, exist_ok=True)

    # Load patched model for quantitative PyTorch reference
    print("Loading model...")
    pipeline, model, set_phases_fn = load_patched_model()
    voice_pack = pipeline.load_voice(BENCHMARK_VOICE)

    # Export dynamic models + palettized variants (subprocess)
    export_dir = tempfile.mkdtemp(prefix="harness_")
    print("Exporting dynamic models...")
    try:
        export_dynamic(export_dir)
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # Generate vanilla PyTorch audio for listening (subprocess)
    vanilla_dir = tempfile.mkdtemp(prefix="vanilla_")
    print("Generating vanilla reference...")
    try:
        generate_vanilla_reference(BENCHMARK_VOICE, TEST_SENTENCES, vanilla_dir)
        for label in TEST_SENTENCES:
            vanilla_path = os.path.join(vanilla_dir, f"{label}.wav")
            if os.path.exists(vanilla_path):
                shutil.copy(vanilla_path,
                    os.path.join(audio_dir, f"{BENCHMARK_VOICE}_{label}_vanilla_pytorch.wav"))
    except Exception as e:
        print(f"Vanilla reference failed: {e}")

    # --- Float32 dynamic models ---
    print("\nTesting float32 dynamic models...")
    fe_f32, be_f32 = _load_coreml_pair(
        os.path.join(export_dir, "kokoro_frontend.mlpackage"),
        os.path.join(export_dir, "kokoro_backend.mlpackage"))

    results = _benchmark_variant("coreml", fe_f32, be_f32,
                                  pipeline, model, set_phases_fn, voice_pack, audio_dir)
    print("\nFloat32:")
    print_results(results)

    # Extra voices (float32 only)
    for voice in AUDIO_VOICES:
        if voice == BENCHMARK_VOICE:
            continue
        vp = pipeline.load_voice(voice)
        for label, text in TEST_SENTENCES.items():
            token_ids = _tokenize(text, pipeline)
            ref_s = vp[len(token_ids)]
            try:
                cm_audio, _ = run_coreml(fe_f32, be_f32, token_ids, ref_s)
                sf.write(os.path.join(audio_dir, f"{voice}_{label}_coreml.wav"),
                         cm_audio, 24000)
            except Exception:
                pass

    # --- Palettized 8-bit models ---
    pal_results = []
    pal_fe_path = os.path.join(export_dir, "kokoro_frontend_pal8.mlpackage")
    pal_be_path = os.path.join(export_dir, "kokoro_backend_pal8.mlpackage")

    if os.path.exists(pal_fe_path) and os.path.exists(pal_be_path):
        print("\nTesting palettized 8-bit models...")
        fe_pal, be_pal = _load_coreml_pair(pal_fe_path, pal_be_path)
        pal_results = _benchmark_variant("pal8", fe_pal, be_pal,
                                          pipeline, model, set_phases_fn, voice_pack, audio_dir)
        print("\nPalettized 8-bit:")
        print_results(pal_results)

    generate_html(results, audio_dir, commit, pal_results=pal_results)
    print(f"\nAudio saved to: {audio_dir}")

    shutil.rmtree(export_dir, ignore_errors=True)
    shutil.rmtree(vanilla_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
