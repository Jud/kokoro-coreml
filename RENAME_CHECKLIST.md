# Rename: kokoro-tts-swift → kokoro-ane

Execute all at once to avoid broken links/refs.

## GitHub

- [ ] Rename repo: `Jud/kokoro-tts-swift` → `Jud/kokoro-ane`
- [ ] Rename homebrew tap: `Jud/homebrew-kokoro` → `Jud/homebrew-kokoro-ane`
- [ ] Update GitHub Pages CNAME/settings if applicable

## Swift module (KokoroTTS → KokoroANE)

- [ ] Rename directory `Sources/KokoroTTS/` → `Sources/KokoroANE/`
- [ ] Rename directory `Tests/KokoroTTSTests/` → `Tests/KokoroANETests/`
- [ ] `Package.swift` — package name, library name, target names, paths (8 occurrences)
- [ ] `Sources/CLI/Daemon.swift:3` — `import KokoroTTS` → `import KokoroANE`
- [ ] `Sources/CLI/Daemon.swift:9` — help text `"Kokoro TTS daemon"` → `"Kokoro ANE daemon"`
- [ ] `Sources/CLI/KokoroSay.swift:4` — `import KokoroTTS` → `import KokoroANE`
- [ ] `Sources/CLI/ModelDownloader.swift:3` — `import KokoroTTS` → `import KokoroANE`
- [ ] `Sources/CLI/ModelDownloader.swift:7` — `"Downloading KokoroTTS models..."` → `"Downloading KokoroANE models..."`
- [ ] `Tests/*/G2PTests.swift:3` — `@testable import KokoroTTS`
- [ ] `Tests/*/TokenizerTests.swift:3` — `@testable import KokoroTTS`
- [ ] `Tests/*/PerformanceTests.swift:4` — `@testable import KokoroTTS`
- [ ] `Tests/*/BARTFallbackTests.swift:4` — `@testable import KokoroTTS`
- [ ] `Tests/*/Num2WordTests.swift:4` — `@testable import KokoroTTS`
- [ ] `Tests/*/G2PReferenceTests.swift:4` — `@testable import KokoroTTS`

## Repo URLs in source/scripts

- [ ] `Sources/KokoroTTS/ModelDownloader.swift:4` — `"Jud/kokoro-tts-swift"` → `"Jud/kokoro-ane"`
- [ ] `scripts/download-models.sh:4` — `REPO="Jud/kokoro-tts-swift"`
- [ ] `scripts/release.sh:13` — `REPO="Jud/kokoro-tts-swift"`
- [ ] `scripts/package-models.sh:30` — repo reference in echo
- [ ] `scripts/export_coreml.py:2` — docstring `kokoro-tts-swift` → `kokoro-ane`

## CI/CD workflows

- [ ] `.github/workflows/ci.yml:62` — model cache path `kokoro-tts` in Application Support
- [ ] `.github/workflows/ci.yml:69` — model dir path `kokoro-tts` in Application Support
- [ ] `.github/workflows/ci.yml:71` — API URL `repos/Jud/kokoro-tts-swift`
- [ ] `.github/workflows/ci.yml:74` — download URL
- [ ] `.github/workflows/release.yml:60` — download URL
- [ ] `.github/workflows/release.yml:62-71` — homebrew tap clone URL + formula name

## Identifiers and paths

- [ ] `Sources/KokoroTTS/KokoroEngine.swift:162` — `"com.kokorotts"` → `"com.kokoroane"`
- [ ] `Sources/CLI/DaemonProtocol.swift:5` — socket path `"kokoro-tts-\(getuid())"` → `"kokoro-ane-\(getuid())"`
- [ ] `Sources/KokoroTTS/ModelManager.swift:3` — comment `"for KokoroTTS"` → `"for KokoroANE"`
- [ ] `Sources/KokoroTTS/ModelManager.swift:13` — comment `"kokoro-tts"` → `"kokoro-ane"`
- [ ] `Sources/KokoroTTS/ModelManager.swift:15` — fallback bundle ID `"kokoro-tts"` → `"kokoro-ane"` (see note below)

## Application Support path (needs migration strategy)

The fallback bundle ID `"kokoro-tts"` determines the model storage path:
`~/Library/Application Support/kokoro-tts/models/kokoro/`

Renaming this to `kokoro-ane` means existing users re-download ~640MB of models.
**Recommendation:** Add a migration check — look for old path first, symlink or copy.

- [ ] `Sources/KokoroTTS/ModelManager.swift:15` — update fallback ID + add migration
- [ ] `scripts/download-models.sh:6` — `DEFAULT_DIR` path
- [ ] `scripts/release.sh:47` — example path in echo
- [ ] `.github/workflows/ci.yml:62,69` — cache path

## Script docstrings (cosmetic)

- [ ] `scripts/patch_coremltools.py:2` — `"Kokoro TTS"` → `"Kokoro ANE"`
- [ ] `scripts/coreml_ops.py:1` — `"Kokoro TTS"` → `"Kokoro ANE"`
- [ ] `scripts/stage_harness.py:2` — `"Kokoro TTS"` → `"Kokoro ANE"`

## README

- [ ] `README.md:1` — title `# kokoro-tts-swift` → `# kokoro-ane`
- [ ] `README.md:12` — SPM package URL

## Research docs

- [ ] `research/program.md:1` — `# kokoro-tts-swift CoreML Research` → `# kokoro-ane CoreML Research`

## Marketing site (already updated)

- [x] `docs/index.html` — GitHub links (3 refs)
- [x] `docs/index.html` — Homebrew install command

## Leave alone (upstream model references)

These reference the upstream Kokoro model, not our project:

- `kokoro_21_5s`, `kokoro_24_10s` — model bucket filenames from upstream
- `kokoro-models.tar.gz` — model asset name
- `KokoroEngine` class name — it's the engine for the Kokoro model
- `~/*/models/kokoro/` — the trailing `kokoro/` dir is the model name, not our project
- `from kokoro import ...` in Python scripts — upstream package
- `kokoro` CLI executable name — keep as-is, it's what users type
- `.swiftlint.yml:30,35` — comments reference `KokoroEngine` class (which stays)

## Homebrew formula

- [ ] Rename tap repo `homebrew-kokoro` → `homebrew-kokoro-ane`
- [ ] Update formula file `Formula/kokoro.rb` — url and homepage fields
- [ ] Update `scripts/release.sh` homebrew tap references

## Post-rename

- [ ] Verify GitHub redirect works (old URL → new URL)
- [ ] `brew untap jud/kokoro && brew tap jud/kokoro-ane`
- [ ] Test `swift package resolve` with new URL
- [ ] Tag a new release from the renamed repo
