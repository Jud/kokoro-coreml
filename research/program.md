# kokoro-tts-swift CoreML Research

Autonomous optimization of the Kokoro TTS CoreML export pipeline.

## Setup

1. **Agree on a run tag** based on today's date (e.g. `mar14`). Branch: `research/<tag>`.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - `scripts/stage_harness.py` — **READ-ONLY**. Fixed evaluation. Do not modify.
   - `scripts/export_coreml.py` — the file you modify. Patches, wrappers, conversion settings.
   - `research/program.md` — this file.
4. **Verify environment**: `.venv/bin/python scripts/stage_harness.py` should run all stages.
5. **Establish baseline**: Your first run is always the harness as-is. Record the results.
6. **Confirm and go**.

## The harness

The harness tests 9 targets: 7 individual stages + 2 split models (split_A = stages 1-5, split_B = stage 6). It runs 3 test sentences (short/medium/long) and reports worst-case.

Per target it reports: CPU Corr, CPU Cold/Warm time, ANE Corr, ANE Cold/Warm time.

Status is PASS if ANE runs and ANE Corr > 0.99. Otherwise WARN or FAIL.

```bash
# Full harness
.venv/bin/python scripts/stage_harness.py

# Single stage
.venv/bin/python scripts/stage_harness.py --stage 6

# JSON output
.venv/bin/python scripts/stage_harness.py --json
```

## The goal

**Get the highest worst-case ANE Corr across all stages.** Higher correlation = better sounding audio.

Since the harness is fixed, you don't need to worry about evaluation — it's always the same. Everything in `scripts/export_coreml.py` is fair game: conversion options, monkey-patches, precision, model splitting, compute unit strategy. The only constraint is that all stages stay PASS (ANE Corr > 0.99).

**Simplicity criterion**: All else being equal, simpler is better. A small speed improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement.

## What you CANNOT do

- Modify `scripts/stage_harness.py`.
- Modify the Kokoro model weights or architecture in `.venv/`.
- Install new packages beyond what's in `.venv/`.

## Logging results

Log every experiment to `research/results.tsv` (tab-separated). Do NOT commit this file — leave it untracked so `git reset` doesn't wipe your log.

```
commit	stage	cpu_corr	ane_corr	cpu_warm_ms	ane_warm_ms	status	description
```

- status: `keep`, `discard`, or `crash`
- Use `--json` to get machine-parseable output for extracting values.

## The experiment loop

LOOP FOREVER:

1. Look at the harness output. What's the bottleneck? What's WARN or FAIL?
2. Form a hypothesis.
3. Modify `scripts/export_coreml.py`.
4. git commit.
5. Run: `.venv/bin/python scripts/stage_harness.py 2>&1 | tee research/run.log`
6. Record in results.tsv.
7. If total latency improved or accuracy improved (without regression): keep.
8. If worse: `git reset --hard HEAD~1`.

If a run crashes, use your judgment: fix trivial bugs and re-run, or skip the idea and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be away and expects you to continue working indefinitely until manually stopped. If you run out of ideas, re-read the harness output, try combinations of previous near-misses, or try more radical changes.
