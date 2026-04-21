# SV-LOI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `zone3/sv_loi.py` (4188 lines) into `zone3/_svloi/` package of ~8 focused modules plus a thin facade, preserving the public API imported by `recursive_induction.py`.

**Architecture:** Mechanical extraction — function bodies are moved verbatim, signatures unchanged, no logic edits. `zone3/sv_loi.py` becomes a ~30-line facade re-exporting the 10 public symbols. Each of 6 migration steps is one commit validated by end-to-end pipeline rerun diffed against a captured baseline.

**Tech Stack:** Python 3.10+, Neo4j, Ollama (qwen2.5:72b), LangChain. Pipeline run from project root: `python3 zone3/sv_loi.py`.

**Reference spec:** `docs/superpowers/specs/2026-04-21-sv-loi-refactor-design.md`

**Validation primitive:** After each migration step, two gates must pass:
1. **Import smoke test** (cheap, local): `python3 -c "from zone3.sv_loi import get_llm, get_neo4j_graph, _invoke_llm, _sanitize_label, _parse_json_safely, write_ontology, derive_interclass_edges, propagate_to_records, type_value_entities, run_sv_loi; print('ok')"` prints `ok`.
2. **End-to-end rerun** (expensive, on cluster): pipeline runs to completion; class list and F1 within tolerance of baseline (F1 ±0.01; class-count diff ≤ 1).

**Out of scope (do NOT do in this plan):**
- Any change to function bodies, logic, or signatures
- Any change to `recursive_induction.py`, `graph_cache.py`, `rsi_lcr.py`, `pipeline.py`
- Any bug fix, feature work, or new tests
- Renaming functions or constants

---

## File Structure (target layout)

```
zone3/
├── sv_loi.py              # ~30 lines — facade re-exporting public API
└── _svloi/
    ├── __init__.py        # empty package marker
    ├── constants.py       # lines 71-127 of current sv_loi.py
    ├── utils.py           # lines 129-206 (get_llm..load_entities)
    ├── typing.py          # lines 208-1308 (Phase 1+2+3)
    ├── structural.py      # lines 1310-1535 (signatures, consensus, arbitration)
    ├── hierarchy.py       # lines 1537-2389 + 2846-2988 (merge + derive + LLM pairwise)
    ├── sohd.py            # lines 2391-2845 (heterogeneity detection block)
    ├── records.py         # lines 2990-3162 (decompose + write_record_decomposition)
    ├── writer.py          # lines 3163-3589 (validate_backbone, write_ontology, quality)
    └── pipeline.py        # lines 3590-4188 (run_sv_loi + __main__)
```

Public symbols re-exported from `zone3/sv_loi.py` facade (10 total):
`get_llm`, `get_neo4j_graph`, `_invoke_llm`, `_sanitize_label`, `_parse_json_safely`, `type_value_entities`, `propagate_to_records`, `derive_interclass_edges`, `write_ontology`, `run_sv_loi`.

---

## Task 1: Capture Baseline (Step 0 of spec)

**Files:**
- Create: `docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt`
- Create: `docs/superpowers/baselines/2026-04-21-svloi-baseline/f1.json`
- Create: `docs/superpowers/baselines/2026-04-21-svloi-baseline/run.log`
- Create: `docs/superpowers/baselines/2026-04-21-svloi-baseline/README.md`

- [ ] **Step 1: Create baseline directory**

```bash
mkdir -p docs/superpowers/baselines/2026-04-21-svloi-baseline
```

- [ ] **Step 2: Run pipeline end-to-end, capture stderr**

From project root:

```bash
python3 zone3/sv_loi.py 2>&1 | tee docs/superpowers/baselines/2026-04-21-svloi-baseline/run.log
```

Expected: pipeline completes without error. Log contains phase markers and final class count.

- [ ] **Step 3: Dump class list from Neo4j**

Use the project's existing Neo4j credentials (loaded from `config.py`). Run a Cypher query returning each ontology class and its member count:

```bash
python3 - <<'PY' > docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows:
    print(f"{r['cls']}\t{r['n']}")
PY
```

Expected: file has one line per class, tab-separated `class_name<TAB>count`.

- [ ] **Step 4: Record F1 score**

Run the existing Riskine eval (whichever script the project uses — likely `eval/riskine_f1.py` or similar; check `scripts/` and `eval/` for the canonical eval entry point before this step) and save the top-line F1 metric:

```bash
# Find the eval entry point:
ls eval/ scripts/ 2>/dev/null | grep -i -E "riskine|eval|f1"
# Run it and capture just the final F1 number to f1.json, e.g.:
# python3 eval/<script>.py --json-out docs/superpowers/baselines/2026-04-21-svloi-baseline/f1.json
```

If no single script exists, create `f1.json` manually from the log with the format:

```json
{"member_f1": 0.234, "name_f1": 0.000, "captured": "2026-04-21"}
```

- [ ] **Step 5: Write baseline README**

```bash
cat > docs/superpowers/baselines/2026-04-21-svloi-baseline/README.md <<'MD'
# SV-LOI Baseline — 2026-04-21

Pre-refactor capture of `zone3/sv_loi.py` output for diffing during the
SV-LOI refactor (see docs/superpowers/specs/2026-04-21-sv-loi-refactor-design.md).

Files:
- classes.txt — tab-separated class_name<TAB>member_count from Neo4j
- f1.json     — Riskine F1 scores at capture time
- run.log     — full pipeline stderr log

Tolerance for post-refactor diffs:
- F1 drift: ±0.01 (LLM non-determinism)
- Class-count drift: ≤ 1 class
- Per-class membership drift: ≤ 5 entities

Any drift exceeding tolerance blocks the next migration step.
MD
```

- [ ] **Step 6: Commit baseline**

```bash
git add docs/superpowers/baselines/2026-04-21-svloi-baseline/
git commit -m "chore(zone3): capture SV-LOI pre-refactor baseline"
```

---

## Task 2: Create `_svloi/` package skeleton

**Files:**
- Create: `zone3/_svloi/__init__.py`

- [ ] **Step 1: Create package directory**

```bash
mkdir -p zone3/_svloi
```

- [ ] **Step 2: Create `__init__.py`**

Create `zone3/_svloi/__init__.py` with exactly this content:

```python
"""Internal package for SV-LOI ontology induction.

Public API is exposed through zone3/sv_loi.py — do not import from
this package directly in external code.
"""
```

- [ ] **Step 3: Verify package imports cleanly**

```bash
python3 -c "import zone3._svloi; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add zone3/_svloi/__init__.py
git commit -m "refactor(zone3): create _svloi package skeleton"
```

---

## Task 3: Extract `constants.py` and `utils.py` (Step 1 of spec)

**Files:**
- Create: `zone3/_svloi/constants.py`
- Create: `zone3/_svloi/utils.py`
- Modify: `zone3/sv_loi.py` (remove extracted blocks, add import-from-package at top)

- [ ] **Step 1: Create `zone3/_svloi/constants.py`**

Read lines 71-127 of `zone3/sv_loi.py` (the "Constants" section). Create `zone3/_svloi/constants.py` containing exactly those lines verbatim. Prepend a module docstring and the one import needed:

```python
"""SV-LOI constants: class vocabulary, prefixes, normalization maps."""
from zone3.graph_cache import STRUCTURED_PREFIXES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# <paste lines 71-127 of zone3/sv_loi.py exactly, starting from:
#   BATCH_SIZE = 15 ...
# through the end of ZONE2_TYPE_NORMALIZATION dict, closing brace>
```

- [ ] **Step 2: Create `zone3/_svloi/utils.py`**

Read lines 129-206 of `zone3/sv_loi.py` (utilities section + `load_entities`). Create `zone3/_svloi/utils.py`:

```python
"""SV-LOI utilities: LLM/Neo4j factories, sanitization, JSON parsing, entity loading."""
from __future__ import annotations

import json
import re
from typing import Union

from langchain_core.messages import HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config
from zone3.graph_cache import load_cached_entities

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# <paste lines 133-206 of zone3/sv_loi.py exactly — the 7 functions
# get_llm, get_neo4j_graph, _sanitize_label, _sanitize_rel_type,
# _parse_json_safely, _invoke_llm, load_entities>
```

- [ ] **Step 3: Update `zone3/sv_loi.py` — delete extracted blocks, add imports**

At the top of `zone3/sv_loi.py` (right after the existing `from zone3.graph_cache import ...` block, around line 70), add:

```python
from zone3._svloi.constants import (
    BATCH_SIZE, MAX_MEMBERS_IN_PROMPT, MIN_CLASS_SIZE,
    DEVIATION_THRESHOLD, MAX_ARBITRATION_BATCH, MAX_CLASS_FRACTION,
    _STRUCTURED_PREFIXES, TARGET_CLASSES_MIN, TARGET_CLASSES_MAX,
    PROTECTED_CLASS_NAMES, FORBIDDEN_CLASS_NAMES, ZONE2_TYPE_NORMALIZATION,
)
from zone3._svloi.utils import (
    get_llm, get_neo4j_graph,
    _sanitize_label, _sanitize_rel_type,
    _parse_json_safely, _invoke_llm,
    load_entities,
)
```

Then delete lines 71-206 (the Constants + Utilities sections and `load_entities`). Leave the file-level docstring and other imports untouched.

- [ ] **Step 4: Import smoke test**

```bash
python3 -c "from zone3.sv_loi import get_llm, get_neo4j_graph, _invoke_llm, _sanitize_label, _parse_json_safely, type_value_entities, propagate_to_records, derive_interclass_edges, write_ontology, run_sv_loi; print('ok')"
```

Expected: prints `ok`. If it fails with `ImportError`, inspect which symbol is missing and check that it was re-exported in Step 3.

- [ ] **Step 5: End-to-end rerun**

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_step1.log
```

Expected: pipeline completes. Note stdout/stderr differences vs baseline.

- [ ] **Step 6: Diff against baseline**

Dump classes from Neo4j same way as Task 1 Step 3, diff against baseline:

```bash
python3 - <<'PY' > /tmp/classes_step1.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_step1.txt
```

Acceptable: ≤ 1 class-count diff, per-class member drift ≤ 5. Anything larger → `git revert HEAD` and investigate before proceeding.

- [ ] **Step 7: Commit**

```bash
git add zone3/_svloi/constants.py zone3/_svloi/utils.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/constants.py and _svloi/utils.py"
```

---

## Task 4: Extract `sohd.py` (Step 2 of spec)

**Files:**
- Create: `zone3/_svloi/sohd.py`
- Modify: `zone3/sv_loi.py` (remove extracted block, add import)

- [ ] **Step 1: Create `zone3/_svloi/sohd.py`**

Read lines 2391-2845 of current `zone3/sv_loi.py` (the SOHD block: section header + 7 functions `_build_class_relation_profiles`, `_cosine_similarity_matrix`, `_js_divergence`, `_top_distinguishing_relations`, `_name_subclass_llm`, `_auto_name_subclass`, `detect_and_split_heterogeneous_classes`).

Create `zone3/_svloi/sohd.py`:

```python
"""Structural Ontological Heterogeneity Detection — splits semantically impure classes."""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from typing import Any

from langchain_ollama import ChatOllama

from zone3._svloi.utils import _invoke_llm, _parse_json_safely, _sanitize_label

# ---------------------------------------------------------------------------
# SOHD: Structural Ontological Heterogeneity Detection
# ---------------------------------------------------------------------------
# <paste lines 2391-2845 of zone3/sv_loi.py exactly — comment block + 7 functions>
```

Before saving, grep the pasted block for any references to names not yet imported. If `_build_class_relation_profiles` references e.g. `BATCH_SIZE` or a helper from elsewhere, add the appropriate import from `zone3._svloi.constants` or `zone3._svloi.utils`. Confirmed dependencies from a read of lines 2391-2845: only uses `_invoke_llm`, `_parse_json_safely`, `_sanitize_label`, and numpy/stdlib. No constants needed.

- [ ] **Step 2: Update `zone3/sv_loi.py` — delete block, add import**

Near the top-of-file imports (after the existing `_svloi` imports added in Task 3), append:

```python
from zone3._svloi.sohd import detect_and_split_heterogeneous_classes
```

Then delete lines 2391-2845 of `zone3/sv_loi.py` (the entire SOHD section).

Note: `_build_class_relation_profiles`, `_cosine_similarity_matrix`, `_js_divergence`, `_top_distinguishing_relations`, `_name_subclass_llm`, `_auto_name_subclass` are all private helpers only called from within SOHD itself — they do not need to be re-exported from the facade. Confirm this with `grep -n "_build_class_relation_profiles\|_cosine_similarity_matrix\|_js_divergence\|_top_distinguishing_relations\|_name_subclass_llm\|_auto_name_subclass" zone3/ -r` — expect matches only inside `zone3/sv_loi.py` (now being removed) and `zone3/_svloi/sohd.py`.

- [ ] **Step 3: Import smoke test**

```bash
python3 -c "from zone3.sv_loi import get_llm, get_neo4j_graph, _invoke_llm, _sanitize_label, _parse_json_safely, type_value_entities, propagate_to_records, derive_interclass_edges, write_ontology, run_sv_loi; print('ok')"
python3 -c "from zone3._svloi.sohd import detect_and_split_heterogeneous_classes; print('sohd ok')"
```

Expected: both print `ok` / `sohd ok`.

- [ ] **Step 4: End-to-end rerun**

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_step2.log
```

Expected: pipeline completes without error.

- [ ] **Step 5: Diff against baseline**

```bash
python3 - <<'PY' > /tmp/classes_step2.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_step2.txt
```

Acceptable: class count ±1, per-class drift ≤ 5 entities. Larger → `git revert HEAD` and investigate.

- [ ] **Step 6: Commit**

```bash
git add zone3/_svloi/sohd.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/sohd.py (900-line heterogeneity block)"
```

---

## Task 5: Extract `writer.py` and `records.py` (Step 3 of spec)

**Files:**
- Create: `zone3/_svloi/records.py`
- Create: `zone3/_svloi/writer.py`
- Modify: `zone3/sv_loi.py`

- [ ] **Step 1: Create `zone3/_svloi/records.py`**

Read lines 2990-3162 (`decompose_records`, `write_record_decomposition`). Create:

```python
"""Record decomposition — split record entities into ontology-class components."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from zone3._svloi.utils import _invoke_llm, _parse_json_safely, _sanitize_label, _sanitize_rel_type

# <paste lines 2990-3162 verbatim — decompose_records + write_record_decomposition>
```

Dependencies confirmed by reading the source block: `_invoke_llm`, `_parse_json_safely`, `_sanitize_label`, `_sanitize_rel_type`. Add any additional imports that appear in the pasted body (check for `json`, `re`, etc. references).

- [ ] **Step 2: Create `zone3/_svloi/writer.py`**

Read lines 3163-3589 (`validate_backbone`, `write_ontology`, `_flush_print`, `_compute_intrinsic_quality`). Create:

```python
"""SV-LOI output: validate ontology backbone and write to Neo4j."""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from langchain_neo4j import Neo4jGraph

from zone3._svloi.utils import _sanitize_label, _sanitize_rel_type

# <paste lines 3163-3589 verbatim — validate_backbone, write_ontology, _flush_print, _compute_intrinsic_quality>
```

- [ ] **Step 3: Update `zone3/sv_loi.py`**

Add to the `_svloi` import block near the top:

```python
from zone3._svloi.records import decompose_records, write_record_decomposition
from zone3._svloi.writer import validate_backbone, write_ontology
```

Note: `write_ontology` is part of the public API (re-exported via facade later). `decompose_records`, `write_record_decomposition`, `validate_backbone`, `_flush_print`, `_compute_intrinsic_quality` are only called internally from `run_sv_loi`; they do not need to be re-exported.

Delete lines 2990-3589 from `zone3/sv_loi.py`.

- [ ] **Step 4: Import smoke test**

```bash
python3 -c "from zone3.sv_loi import write_ontology; print('ok')"
python3 -c "from zone3._svloi.records import decompose_records, write_record_decomposition; print('records ok')"
python3 -c "from zone3._svloi.writer import validate_backbone, write_ontology, _compute_intrinsic_quality; print('writer ok')"
```

Expected: all three lines print `ok`.

- [ ] **Step 5: End-to-end rerun**

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_step3.log
```

- [ ] **Step 6: Diff against baseline**

```bash
python3 - <<'PY' > /tmp/classes_step3.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_step3.txt
```

Acceptable: class count ±1, per-class drift ≤ 5. Larger → `git revert HEAD` and investigate.

- [ ] **Step 7: Commit**

```bash
git add zone3/_svloi/records.py zone3/_svloi/writer.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/records.py and _svloi/writer.py"
```

---

## Task 6: Extract `structural.py` and `hierarchy.py` (Step 4 of spec)

**Files:**
- Create: `zone3/_svloi/structural.py`
- Create: `zone3/_svloi/hierarchy.py`
- Modify: `zone3/sv_loi.py`

- [ ] **Step 1: Create `zone3/_svloi/structural.py`**

Read lines 1310-1535 (`build_structural_signatures`, `structural_consensus_check`, `arbitrate_disagreements`). Create:

```python
"""Structural consensus + disagreement arbitration for LLM-assigned classes."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from langchain_ollama import ChatOllama

from zone3._svloi.constants import DEVIATION_THRESHOLD, MAX_ARBITRATION_BATCH
from zone3._svloi.utils import _invoke_llm, _parse_json_safely

# <paste lines 1310-1535 verbatim — build_structural_signatures, structural_consensus_check, arbitrate_disagreements>
```

- [ ] **Step 2: Create `zone3/_svloi/hierarchy.py`**

`hierarchy.py` receives two non-contiguous ranges. Assemble in this order:

1. Lines 1537-2389 (`infer_class_relations`, `merge_small_classes`, `merge_leaf_classes`, `derive_interclass_edges`, `derive_hierarchy`, `derive_taxonomy`)
2. Lines 2846-2988 (`derive_taxonomy_llm_pairwise`)

Create:

```python
"""IS-A hierarchy derivation: class merging, inter-class edges, taxonomy building."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from langchain_ollama import ChatOllama

from zone3._svloi.constants import (
    MIN_CLASS_SIZE, MAX_CLASS_FRACTION,
    PROTECTED_CLASS_NAMES, FORBIDDEN_CLASS_NAMES,
)
from zone3._svloi.utils import (
    _invoke_llm, _parse_json_safely,
    _sanitize_label, _sanitize_rel_type,
)

# <paste lines 1537-2389 verbatim>

# <paste lines 2846-2988 verbatim — derive_taxonomy_llm_pairwise>
```

Before saving: read the pasted bodies and confirm every `CONSTANT_NAME` referenced is either in the imports above or defined in `_svloi/constants.py`. Add any missing names to the constants import.

- [ ] **Step 3: Update `zone3/sv_loi.py`**

Add to the `_svloi` import block:

```python
from zone3._svloi.structural import (
    build_structural_signatures, structural_consensus_check, arbitrate_disagreements,
)
from zone3._svloi.hierarchy import (
    infer_class_relations, merge_small_classes, merge_leaf_classes,
    derive_interclass_edges, derive_hierarchy, derive_taxonomy,
    derive_taxonomy_llm_pairwise,
)
```

Note: `derive_interclass_edges` is part of the public API (imported by `recursive_induction.py`).

Delete lines 1310-1535, 1537-2389, and 2846-2988 from `zone3/sv_loi.py`.

- [ ] **Step 4: Import smoke test**

```bash
python3 -c "from zone3.sv_loi import derive_interclass_edges; print('ok')"
python3 -c "from zone3._svloi.structural import build_structural_signatures, structural_consensus_check, arbitrate_disagreements; print('structural ok')"
python3 -c "from zone3._svloi.hierarchy import merge_small_classes, merge_leaf_classes, derive_hierarchy, derive_taxonomy, derive_taxonomy_llm_pairwise, infer_class_relations; print('hierarchy ok')"
```

Expected: all three print `ok`.

- [ ] **Step 5: End-to-end rerun**

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_step4.log
```

- [ ] **Step 6: Diff against baseline**

```bash
python3 - <<'PY' > /tmp/classes_step4.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_step4.txt
```

Acceptable: class count ±1, per-class drift ≤ 5. Larger → `git revert HEAD`.

- [ ] **Step 7: Commit**

```bash
git add zone3/_svloi/structural.py zone3/_svloi/hierarchy.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/structural.py and _svloi/hierarchy.py"
```

---

## Task 7: Extract `typing.py` (Step 5 of spec)

**Files:**
- Create: `zone3/_svloi/typing.py`
- Modify: `zone3/sv_loi.py`

This is the largest and most coupled extraction — saved for last so it's the only remaining unknown when debugged.

- [ ] **Step 1: Create `zone3/_svloi/typing.py`**

Read lines 208-1308 (section header + 7 functions: `analyze_record_evidence`, `discover_class_vocabulary`, `batch_type_entities`, `rescue_other_entities`, `type_value_entities`, `rebalance_mega_classes`, `propagate_to_records`).

⚠️ `typing` collides with Python's standard library. Import order matters: this module is `zone3._svloi.typing`, NOT `typing`. Other files in `_svloi` that want stdlib `typing` should continue to `from typing import Any` at the top — that resolves to stdlib because `zone3._svloi.typing` is only reachable via its full dotted name.

Create `zone3/_svloi/typing.py`:

```python
"""SV-LOI entity typing: evidence analysis, class vocabulary discovery, batched LLM assignment."""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from zone3._svloi.constants import (
    BATCH_SIZE, MAX_MEMBERS_IN_PROMPT, MIN_CLASS_SIZE, MAX_CLASS_FRACTION,
    _STRUCTURED_PREFIXES, TARGET_CLASSES_MIN, TARGET_CLASSES_MAX,
    FORBIDDEN_CLASS_NAMES, ZONE2_TYPE_NORMALIZATION,
)
from zone3._svloi.utils import (
    _invoke_llm, _parse_json_safely,
    _sanitize_label, _sanitize_rel_type,
)
from zone3.graph_cache import (
    get_concept_entities, get_entity_lane, is_concept_entity,
    STRUCTURED_PREFIXES,
)

# <paste lines 208-1308 verbatim — 7 functions including section comment headers>
```

After pasting, grep the module for any `NameError`-risk references — any constant used inside a function that isn't in the imports above. Add imports as needed. Re-check that no stdlib `typing` reference was broken (search for `typing.` prefix; there should be none because the code uses `from typing import ...`).

- [ ] **Step 2: Update `zone3/sv_loi.py`**

Add to the `_svloi` import block:

```python
from zone3._svloi.typing import (
    analyze_record_evidence, discover_class_vocabulary, batch_type_entities,
    rescue_other_entities, type_value_entities, rebalance_mega_classes,
    propagate_to_records,
)
```

Note: `type_value_entities` and `propagate_to_records` are part of the public API.

Delete lines 208-1308 from `zone3/sv_loi.py`.

- [ ] **Step 3: Import smoke test**

```bash
python3 -c "from zone3.sv_loi import type_value_entities, propagate_to_records; print('ok')"
python3 -c "from zone3._svloi.typing import analyze_record_evidence, discover_class_vocabulary, batch_type_entities, rescue_other_entities, rebalance_mega_classes; print('typing ok')"
```

Expected: both print `ok` / `typing ok`. If an `ImportError` about `typing` appears (stdlib shadowed), the file's internal imports need fixing — verify every `from typing import ...` resolved to stdlib.

- [ ] **Step 4: End-to-end rerun**

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_step5.log
```

This is the most critical rerun — typing is the core pipeline; any regression will manifest as class-count or membership drift.

- [ ] **Step 5: Diff against baseline**

```bash
python3 - <<'PY' > /tmp/classes_step5.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_step5.txt
```

Acceptable: class count ±1, per-class drift ≤ 5. Larger → `git revert HEAD`.

- [ ] **Step 6: Commit**

```bash
git add zone3/_svloi/typing.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/typing.py (Phase 1+2+3 typing block)"
```

---

## Task 8: Extract `pipeline.py` and trim facade (Step 6 of spec)

**Files:**
- Create: `zone3/_svloi/pipeline.py`
- Rewrite: `zone3/sv_loi.py` (shrinks to ~30-line facade)

- [ ] **Step 1: Create `zone3/_svloi/pipeline.py`**

Read lines 3590-4188 (`run_sv_loi` + `__main__` block). Create:

```python
"""SV-LOI pipeline orchestrator + CLI entry point."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from zone3._svloi.constants import MIN_CLASS_SIZE
from zone3._svloi.utils import get_llm, get_neo4j_graph, load_entities
from zone3._svloi.typing import (
    analyze_record_evidence, discover_class_vocabulary, batch_type_entities,
    rescue_other_entities, type_value_entities, rebalance_mega_classes,
    propagate_to_records,
)
from zone3._svloi.structural import (
    build_structural_signatures, structural_consensus_check, arbitrate_disagreements,
)
from zone3._svloi.hierarchy import (
    infer_class_relations, merge_small_classes, merge_leaf_classes,
    derive_interclass_edges, derive_hierarchy, derive_taxonomy,
    derive_taxonomy_llm_pairwise,
)
from zone3._svloi.sohd import detect_and_split_heterogeneous_classes
from zone3._svloi.records import decompose_records, write_record_decomposition
from zone3._svloi.writer import validate_backbone, write_ontology, _compute_intrinsic_quality

# <paste lines 3590-4188 verbatim — run_sv_loi function + if __name__ == "__main__" block>


def main() -> None:
    """Module-level entry point exposed for zone3/sv_loi.py facade."""
    # <if the existing __main__ block is a standalone sequence of statements
    #  rather than wrapped in a function, move those statements here.>
```

The existing `if __name__ == "__main__":` block must be restructured into a `main()` function so the facade can call it. Copy the body of the `if __name__` block into `def main():`, then at the bottom of `pipeline.py` add:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rewrite `zone3/sv_loi.py` as thin facade**

Replace the entire file with exactly this content:

```python
"""SV-LOI facade — preserves the public API. Implementation lives in zone3/_svloi/."""
from zone3._svloi.utils import (
    get_llm, get_neo4j_graph,
    _invoke_llm, _sanitize_label, _parse_json_safely,
)
from zone3._svloi.typing import type_value_entities, propagate_to_records
from zone3._svloi.hierarchy import derive_interclass_edges
from zone3._svloi.writer import write_ontology
from zone3._svloi.pipeline import run_sv_loi, main

__all__ = [
    "get_llm",
    "get_neo4j_graph",
    "_invoke_llm",
    "_sanitize_label",
    "_parse_json_safely",
    "type_value_entities",
    "propagate_to_records",
    "derive_interclass_edges",
    "write_ontology",
    "run_sv_loi",
]


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify no external caller references a non-exported symbol**

```bash
grep -rn "from zone3.sv_loi import\|from zone3 import sv_loi" --include="*.py" . | grep -v "zone3/sv_loi.py\|zone3/_svloi/"
```

Expected: only matches are in `zone3/recursive_induction.py`. Every imported name in those matches must appear in `__all__` of the new facade. If any external caller imports a symbol not in `__all__` (e.g. `_sanitize_rel_type`), add it to the facade's imports and `__all__`.

- [ ] **Step 4: Verify facade line count**

```bash
wc -l zone3/sv_loi.py
```

Expected: fewer than 50 lines.

- [ ] **Step 5: Import smoke test (full public surface)**

```bash
python3 -c "from zone3.sv_loi import get_llm, get_neo4j_graph, _invoke_llm, _sanitize_label, _parse_json_safely, type_value_entities, propagate_to_records, derive_interclass_edges, write_ontology, run_sv_loi; print('ok')"
python3 -c "from zone3.recursive_induction import *; print('recursive_induction still imports ok')"
```

Expected: both print `ok`.

- [ ] **Step 6: End-to-end rerun + diff**

Final validation — full rerun, classes + F1 vs baseline.

```bash
python3 zone3/sv_loi.py 2>&1 | tee /tmp/svloi_final.log
python3 - <<'PY' > /tmp/classes_final.txt
from zone3.sv_loi import get_neo4j_graph
g = get_neo4j_graph()
rows = g.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL RETURN n.ontology_class AS cls, count(*) AS n ORDER BY cls")
for r in rows: print(f"{r['cls']}\t{r['n']}")
PY
diff docs/superpowers/baselines/2026-04-21-svloi-baseline/classes.txt /tmp/classes_final.txt
```

Acceptable diff: F1 within ±0.01, class count within ±1, per-class drift ≤ 5.

- [ ] **Step 7: Commit**

```bash
git add zone3/_svloi/pipeline.py zone3/sv_loi.py
git commit -m "refactor(zone3): extract _svloi/pipeline.py, trim sv_loi.py to 30-line facade"
```

---

## Task 9: Final audit

**Files:** (read-only checks)

- [ ] **Step 1: Verify target file sizes**

```bash
wc -l zone3/sv_loi.py zone3/_svloi/*.py
```

Expected:
- `zone3/sv_loi.py` — < 50 lines
- Each `zone3/_svloi/*.py` — < 1000 lines

- [ ] **Step 2: Verify no stale references**

```bash
grep -rn "from zone3.sv_loi import" --include="*.py" .
```

Expected: only `recursive_induction.py` matches, and every imported symbol is in `sv_loi.__all__`.

- [ ] **Step 3: Verify recursive_induction.py is unchanged**

```bash
git log --oneline zone3/recursive_induction.py | head -3
```

Expected: the most recent commit touching `recursive_induction.py` predates this refactor branch.

- [ ] **Step 4: Run the full test-equivalent once more — end-to-end rerun + diff**

Same as Task 8 Step 6. This is the final gate.

- [ ] **Step 5: No commit** — this task is pure verification. If any check fails, fix in a new follow-up commit.

---

## Rollback Procedure

If any migration task's validation gate (end-to-end rerun) fails:

```bash
git revert HEAD        # reverts only the failing task's commit
```

Then investigate the regression. Each task is independently revertable because it moves one coherent block with no dependency on a later task's moves. Baseline capture (Task 1) and package skeleton (Task 2) are prerequisites for all subsequent tasks, but once they land, Tasks 3–8 are isolated commits.

---

## Summary of Task ↔ Spec Step Mapping

| Spec Step | Task(s) | Deliverable |
|-----------|---------|-------------|
| Step 0 — Baseline | Task 1 | Baseline artifacts in `docs/superpowers/baselines/` |
| Setup | Task 2 | Empty `_svloi/` package |
| Step 1 — constants + utils | Task 3 | `constants.py`, `utils.py` |
| Step 2 — SOHD | Task 4 | `sohd.py` |
| Step 3 — writer + records | Task 5 | `writer.py`, `records.py` |
| Step 4 — structural + hierarchy | Task 6 | `structural.py`, `hierarchy.py` |
| Step 5 — typing | Task 7 | `typing.py` |
| Step 6 — facade + pipeline | Task 8 | `pipeline.py`, trimmed `sv_loi.py` |
| Verification | Task 9 | Final audit checks |
