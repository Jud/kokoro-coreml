#!/usr/bin/env python3
"""Compare KokoroTTS Swift output against the Python reference implementation.

Usage:
    ./scripts/compare.py "text to compare" [--voice af_heart] [--speed 1.0]
    ./scripts/compare.py --batch tests.txt   # one sentence per line
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def run_python_reference(text, voice="af_heart", speed=1.0):
    """Run the Python Kokoro reference and return phonemes + audio."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])
    all_audio = []
    all_phonemes = []

    for gs, ps, audio in pipeline(text, voice=voice, speed=speed):
        all_phonemes.append(ps)
        all_audio.append(audio)

    phonemes = " ".join(all_phonemes)
    audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    return phonemes, audio


def run_swift_cli(text, voice="af_heart", speed=1.0, wav_path=None):
    """Run kokoro-say and return phonemes + audio."""
    if wav_path is None:
        wav_path = tempfile.mktemp(suffix=".wav")

    cmd = [
        "kokoro-say",
        "--debug",
        "-v", voice,
        "-s", str(speed),
        "-o", wav_path,
        text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Swift CLI failed: {result.stderr}", file=sys.stderr)
        return None, None

    # Parse phonemes from debug output
    phonemes = ""
    for line in result.stderr.splitlines() + result.stdout.splitlines():
        if line.startswith("Phonemes: "):
            phonemes = line[len("Phonemes: "):]
            break

    # Read audio
    audio, sr = sf.read(wav_path)
    assert sr == 24000, f"Expected 24kHz, got {sr}"
    os.unlink(wav_path)

    return phonemes, audio.astype(np.float32)


def compare_audio(swift_audio, python_audio):
    """Compare two audio arrays and return metrics."""
    min_len = min(len(swift_audio), len(python_audio))
    if min_len == 0:
        return {"error": "empty audio"}

    s = swift_audio[:min_len]
    p = python_audio[:min_len]

    diff = s - p
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(np.abs(diff)))

    # Correlation
    if np.std(s) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(s, p)[0, 1])
    else:
        corr = 0.0

    return {
        "swift_duration": f"{len(swift_audio) / 24000:.2f}s",
        "python_duration": f"{len(python_audio) / 24000:.2f}s",
        "duration_diff": f"{abs(len(swift_audio) - len(python_audio)) / 24000:.3f}s",
        "mse": f"{mse:.6f}",
        "max_diff": f"{max_diff:.4f}",
        "correlation": f"{corr:.4f}",
        "swift_peak": f"{np.max(np.abs(swift_audio)):.4f}",
        "python_peak": f"{np.max(np.abs(python_audio)):.4f}",
    }


def compare_one(text, voice="af_heart", speed=1.0, output_dir=None):
    """Run one comparison and return results."""
    print(f"\n{'='*60}")
    print(f"Text:  {text}")
    print(f"Voice: {voice}  Speed: {speed}")
    print(f"{'='*60}")

    # Run both
    print("Running Python reference...")
    py_phonemes, py_audio = run_python_reference(text, voice, speed)

    print("Running Swift CLI...")
    swift_phonemes, swift_audio = run_swift_cli(text, voice, speed)

    if swift_audio is None:
        print("FAILED: Swift CLI returned no output")
        return None

    # Compare phonemes
    print(f"\nPhonemes (Python): {py_phonemes}")
    print(f"Phonemes (Swift):  {swift_phonemes}")
    match = "MATCH" if py_phonemes == swift_phonemes else "DIFFER"
    print(f"Phonemes: {match}")

    # Compare audio
    metrics = compare_audio(swift_audio, py_audio)
    print(f"\nAudio comparison:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save WAVs if output dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        slug = text[:40].replace(" ", "_").replace("/", "_")
        sf.write(f"{output_dir}/{slug}_python.wav", py_audio, 24000)
        sf.write(f"{output_dir}/{slug}_swift.wav", swift_audio, 24000)
        print(f"\nSaved to {output_dir}/{slug}_{{python,swift}}.wav")

    return {
        "text": text,
        "voice": voice,
        "phonemes_match": match == "MATCH",
        "py_phonemes": py_phonemes,
        "swift_phonemes": swift_phonemes,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Swift vs Python Kokoro TTS")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--batch", help="File with one sentence per line")
    parser.add_argument("-v", "--voice", default="af_heart")
    parser.add_argument("-s", "--speed", type=float, default=1.0)
    parser.add_argument("-o", "--output-dir", default="compare_output",
                        help="Directory to save WAV files")
    args = parser.parse_args()

    if args.batch:
        with open(args.batch) as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif args.text:
        sentences = [args.text]
    else:
        parser.error("Provide text or --batch file")

    results = []
    for text in sentences:
        r = compare_one(text, args.voice, args.speed, args.output_dir)
        if r:
            results.append(r)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(results)} comparisons")
        phoneme_matches = sum(1 for r in results if r["phonemes_match"])
        print(f"  Phoneme matches: {phoneme_matches}/{len(results)}")
        corrs = [float(r["metrics"]["correlation"]) for r in results]
        print(f"  Avg correlation: {np.mean(corrs):.4f}")
        print(f"  Min correlation: {np.min(corrs):.4f}")


if __name__ == "__main__":
    main()
