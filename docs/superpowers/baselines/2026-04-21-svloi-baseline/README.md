# SV-LOI Baseline — 2026-04-21

Pre-refactor capture of `zone3/sv_loi.py` output for diffing during the
SV-LOI refactor (see `docs/superpowers/specs/2026-04-21-sv-loi-refactor-design.md`).

## Files

- `classes.txt` — tab-separated `class_name<TAB>member_count` from Neo4j
- `f1.json` — Riskine F1 scores at capture time
- `run.log` — full pipeline stderr log

## Tolerance for post-refactor diffs

- F1 drift: ±0.01 (LLM non-determinism)
- Class-count drift: ≤ 1 class
- Per-class membership drift: ≤ 5 entities

Any drift exceeding tolerance blocks the next migration step.
