# Palettization Quality Analysis

8-bit palettized dynamic CoreML models vs float32 — can we ship them?

## The problem

Waveform correlation between palettized and PyTorch is terrible (0.19 on long sequences). But the audio sounds identical to human ears. Why?

## Phase drift

The palettized vocoder accumulates phase drift over time. The waveform shape diverges progressively, but the spectral content stays the same.

| Length | Waveform Corr | Phase-Corrected (100ms) | Spectral Corr |
|--------|--------------|------------------------|---------------|
| short  | 0.87         | 0.93                   | **0.997**     |
| medium | 0.23         | 0.91                   | **0.997**     |
| long   | 0.19         | 0.89                   | **0.996**     |

Waveform correlation gets worse with length. Spectral correlation stays rock solid. Human hearing is phase-insensitive for speech.

## Noise floor analysis

Does palettization introduce static or high-frequency artifacts?

| Variant         | SNR     | Noise Floor | High-Freq Energy |
|-----------------|---------|-------------|-----------------|
| PyTorch         | 13.7 dB | 0.000181    | 0.000092        |
| Float32 CoreML  | 13.9 dB | 0.000181    | 0.000092        |
| Palettized      | 13.8 dB | 0.000181    | 0.000094        |

No. Identical noise floor. Identical SNR. Zero artifacts introduced.

## Baselines — what does "perfect" look like?

| Comparison | Spectral Corr |
|-----------|--------------|
| Vanilla PyTorch vs itself | 1.0000 |
| Patched PyTorch vs itself | 1.0000 |
| Vanilla PyTorch vs Patched PyTorch (same text, different runs) | 0.5230 |

Even two different PyTorch runs of the same text only score 0.52 spectral correlation against each other — because random vocoder phases differ between runs. This is the ceiling for cross-pipeline comparison. Our dynamic CoreML models score 0.52 against vanilla, matching what PyTorch scores against itself across runs.

## Negative control — can we detect actually bad audio?

We had old bucket models where short text on a large bucket sounded terrible (97% padding). If our metrics can't distinguish those from good audio, they're useless.

### Against vanilla PyTorch (absolute quality)

| Variant                     | Spectral Corr | Mel Cosine | Quality   |
|-----------------------------|--------------|------------|-----------|
| Large bucket (97% padding)  | 0.144        | 0.163      | Terrible  |
| Medium bucket (94% padding) | 0.259        | 0.277      | Degraded  |
| Small bucket (88% padding)  | 0.364        | 0.379      | Decent    |
| Dynamic float32             | 0.524        | 0.535      | Great     |
| Dynamic palettized          | 0.522        | 0.534      | Great     |

The metrics correctly rank bad → good. Large bucket scores 0.14, dynamic scores 0.52.

### Against own pipeline reference (relative quality)

| Variant            | Spectral Corr | Mel Cosine |
|--------------------|--------------|------------|
| Dynamic float32    | 0.9996       | 0.9997     |
| Dynamic palettized | 0.9973       | 0.9974     |

Palettized is 0.9973 vs float32's 0.9996. Both effectively identical.

## Model size impact

| Component | Float32 | Palettized | Reduction |
|-----------|---------|------------|-----------|
| Frontend  | 106 MB  | 27 MB      | 75%       |
| Backend   | 203 MB  | 51 MB      | 75%       |
| **Total** | **309 MB** | **78 MB** | **75%**  |

## Conclusion

Palettization is perceptually lossless on the dynamic CoreML models:

- Spectral correlation 0.996+ (phase-invariant, matches human perception)
- Zero noise floor increase
- Zero high-frequency artifacts
- Correctly distinguished from known-bad audio by the same metrics
- 75% model size reduction (309 MB → 78 MB)

Waveform correlation (0.19) is misleading — it's measuring phase drift, not quality degradation. The log-spectrogram correlation is the correct metric for TTS quality.
