# Final Wrap-Up Runbook — Flood + Emory Pipeline Reruns

**Goal:** Rerun the full pipeline (zone1 → zone2 → zone3 SV-LOI → eval) on both flood and Emory_Spring2026 datasets using the refactored `zone3/_svloi/` code, then refresh presentation materials with fresh numbers.

**Cluster:** Emory Turing (GPUs through May 8). Scratch: `/local/scratch/xwa2284/`.
**Branch to sync:** `main` (commit `286dbd6` or later — contains the 10-module refactor).
**SSH requires password** — user must run cluster commands interactively.

---

## Phase 1 — Sync code to cluster

From local main worktree root (`/Users/sam/Documents/School/Emory/CS584_AI_Capstone`):

```bash
git checkout main
git pull
bash scripts/sync_to_cluster.sh xwa2284
```

Expected: `rsync` prompts for password, uploads refactored zone3/_svloi/, scripts/, zone1/, zone2/, data/flood/, data/riskine/. (Emory_Spring2026 data stays on cluster — gitignored locally.)

Verify on cluster (one-liner, prompts for password):

```bash
ssh -J xwa2284@lab0z.mathcs.emory.edu xwa2284@turinglogin.mathcs.emory.edu \
  'ls /local/scratch/xwa2284/project/zone3/_svloi/ && ls /local/scratch/xwa2284/project/data/Emory_Spring2026/raw/ 2>/dev/null | head'
```

Expect: 10 files in `_svloi/` (pipeline.py, records.py, sohd.py, structural.py, typing.py, hierarchy.py, writer.py, utils.py, constants.py, __init__.py). Emory raw data should exist from prior sessions.

---

## Phase 2 — Submit flood rerun

```bash
ssh -J xwa2284@lab0z.mathcs.emory.edu xwa2284@turinglogin.mathcs.emory.edu \
  'cd /local/scratch/xwa2284/project && sbatch scripts/slurm_pipeline.sh'
```

- Defaults to `DATA_DIR=data/flood`, `MODEL=qwen2.5:72b`, `PASSES=1`.
- Zone 1 skipped (flood chunks already materialized).
- Output lands in `data/results/flood/` (overwrites prior).
- Runtime: ~6–10 h (zone2 extraction on 188 chunks dominates).
- Check queue: `squeue -u xwa2284`. Tail log: `tail -f /local/scratch/xwa2284/logs/<jobid>.out`.

---

## Phase 3 — Submit Emory rerun

Don't start until flood job has a GPU slot (2×48 GB total — one full job saturates cluster):

```bash
ssh -J xwa2284@lab0z.mathcs.emory.edu xwa2284@turinglogin.mathcs.emory.edu \
  'cd /local/scratch/xwa2284/project && sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_pipeline.sh'
```

- This one runs Zone 1 ingestion too (Emory chunks not pre-built).
- Output in `data/results/emory/` (overwrites prior).
- Runtime: ~8–12 h.

**Tip:** submit both back-to-back with `--dependency=afterok:<flood_jobid>` if you want SLURM to chain them.

---

## Phase 4 — Fetch results

Once both `squeue -u xwa2284` shows empty:

```bash
bash scripts/sync_to_cluster.sh xwa2284 --fetch
```

Pulls `data/results/flood/`, `data/results/emory/`, `data/results/visualizations/` into the local repo.

Sanity check:

```bash
ls -la data/results/flood/zone3_svloi_summary_*.json data/results/emory/zone3_svloi_summary_*.json
python3 -c "import json; [print(k, ':', v) for k, v in json.load(open('data/results/flood/zone3_svloi_summary_flood_qwen2_5_72b.json')).items() if k in ('num_classes','num_entities','silhouette','purity')]"
```

Then clean scratch (policy — not backed up):

```bash
ssh -J xwa2284@lab0z.mathcs.emory.edu xwa2284@turinglogin.mathcs.emory.edu 'rm -rf /local/scratch/xwa2284/project'
```

---

## Phase 5 — Presentation refresh (I can do this offline after fetch)

Numbers in `presentation/speech_notes.md` and the `.pptx`/`.tex` that will move once reruns complete:

| File | Current value | Source JSON (new) |
|---|---|---|
| speech_notes Slide 3 | "271 chunks … 22K triples … 5,255 entities … 9 classes … 27K rels" | `zone1_*` + `zone2_run_summary.json` + `zone3_svloi_summary_*.json` |
| Slide 6 | 91.1% triple precision, 73.2% fact recall, 80% source grounding | `eval/extraction_quality_*.json` |
| Slide 7 | BERTScore F1 0.617 / P 0.758, Graph F1 P 0.842 / R 0.291 | `eval/riskine_eval_*.json` |
| Slide 8 | Wu-Palmer 0.621, Entity F1 0.206 (all), 0.404 (evidence-only) | `eval/riskine_eval_*.json` |
| Slide 9 | 61,606h claim resolution, 274 Cellular Phone claims, 144 Physical Damage | `data/results/emory/zone3_svloi_summary_*.json` (Emory only) |
| Slide 10 | 9 classes, 100% backbone coverage, 0% duplication, 0% type inconsistency | `zone3_svloi_summary_*.json` |
| `project_report.tex` | same metrics as slides | same sources |

**Post-rerun checklist for me:**
1. Read the 4 summary JSONs (flood/emory × zone2/zone3).
2. Produce a `docs/final_metrics.md` table side-by-side (pre-refactor baseline vs post-refactor rerun) to confirm no regression.
3. Update `speech_notes.md` numbers in place (Edit tool).
4. Update `project_report.tex` numbers.
5. Regenerate `project_report.pdf` (`cd presentation && pdflatex project_report.tex`).
6. Skim `capstone_showcase.pptx` + `Automated_Insurance_Ontology_Induction.pptx` — if numbers are baked into images, flag which slides need regen via `generate.js`.
7. Final `git commit` — conventional format, no Claude attribution.

---

## Blocker flags

- **If Emory raw data missing on cluster:** re-upload from local archive (`data/Emory_Spring2026/` is gitignored — check `~/` for backup or Assurant-provided source).
- **If OOM at Zone 3:** the refactor matched byte-identical behavior so same memory footprint. If regression appears, diff against `docs/superpowers/baselines/2026-04-21-svloi-baseline/run.log` (previous session left that baseline).
- **If scratch quota hit:** `du -sh /local/scratch/xwa2284/*` first; models cache is ~40 GB and can be kept between runs.
