# kokoro-tts-swift CoreML Research

This is an experiment to have the LLM optimize CoreML model export.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `research/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `scripts/stage_harness.py` — fixed evaluation harness. Do not modify.
   - `scripts/export_coreml.py` — the file you modify. Contains inlined `CustomSTFT` and `SineGen`, pipeline split modules, and export logic.
   - `research/program.md` — this file.
4. **Verify environment**: `.venv/bin/python scripts/stage_harness.py` should run all stages.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the stage harness, which traces, converts, and compares 11 targets across 3 test sentences. You launch it simply as: `.venv/bin/python scripts/stage_harness.py 2>&1 > run.log`

**What you CAN do:**
- Modify `scripts/export_coreml.py` — this is the only file you edit. Everything is fair game: CustomSTFT, SineGen, pipeline split modules, patching logic.

**What you CANNOT do:**
- Modify `scripts/stage_harness.py`. It is read-only. It contains the fixed evaluation, stage wrappers, and comparison logic.
- Modify files in `.venv/`.
- Install new packages or add dependencies.

**The goal is simple: get the highest worst-case ANE Corr and lowest ANE latency.** The harness reports CPU Corr, ANE Corr, CPU Cold/Warm, ANE Cold/Warm for each stage. PASS if ANE Corr > 0.99. Pipeline stages (10, 11) test two-model CPU+ANE splits.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the harness as is.

## Output format

The harness prints a results table like this:

```
Stage            Status    CPU Corr   Cold   Warm   ANE Corr   Cold   Warm
----------------------------------------------------------------------------------------
6_decoder        PASS        0.9961  371ms  335ms     0.9960 1398ms 1384ms
10_gen_pipe      PASS        0.9961  353ms  327ms     0.9961  179ms  121ms
11_dec_pipe      PASS        0.9961  378ms  337ms     0.9961  218ms  131ms
```

You can run a single stage for faster iteration:

```
.venv/bin/python scripts/stage_harness.py --stage 10
.venv/bin/python scripts/stage_harness.py --stage 11
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	stage	cpu_corr	ane_corr	cpu_warm_ms	ane_warm_ms	status	description
```

- One row per stage per experiment.
- status: `keep`, `discard`, or `crash`
- Run multiple times to verify — single-run results can be misleading due to phase-dependent variance.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `research/mar14`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `scripts/export_coreml.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `.venv/bin/python scripts/stage_harness.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "WORST CASE" -A 20 run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If ANE Corr improved or ANE latency decreased without regression: keep.
9. If worse: `git reset --hard HEAD~1`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the ANE reference guide (https://github.com/hollance/neural-engine), re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
