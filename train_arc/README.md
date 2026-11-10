# train_arc

Training and evaluation scripts. Run everything from `train_arc/`.

## Two-stage taxonomic pipeline (crab, star)

Stage 1: YOLO broad detector (one class). Stage 2: hierarchical (genus + species) classifier on ground-truth crops.
Crab and star run through the **same** script; only the system profile differs.

```bash
nohup ./job/gcp_yolo_job.sh crab > ./log/yolo_crab_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup ./job/gcp_yolo_job.sh star > ./log/yolo_star_$(date +%Y%m%d_%H%M%S).log 2>&1 &
./job/gcp_eval.sh crab <run_name under output/>     # re-evaluate an existing run, no retraining
FORCE_PREP=1 ./job/gcp_yolo_job.sh crab              # redo broad-label + crop prep
```

| Piece | Where |
|---|---|
| System profile (year, data label, model, epochs, broad class name, split CSV) | `job/systems/<system>.env` |
| Taxonomy (single source of truth for class ids) | `config/<LABEL>_taxonomy.json` |
| Data layout | `<LABEL><YEAR>/{annotations,annotated,empties}` in; `yolo/`, `yolo_broad/`, `crops/` built by the job |

Steps: -1 build `yolo/` from the raw layout -> 0 taxonomy gate (JSON vs `yolo/data.yaml`) -> 1 broad labels -> 2 crops -> 3 Stage 1 train -> 4 Stage 1 eval -> 5 Stage 2 train -> 6 cascade eval.
Steps -1, 1 and 2 are skipped when their outputs exist.

Crab data (`crabdata2226`, 2022-2026) is prepared on the SSD under `crab_data/` (`s/build_crab_master.py`, `s/build_crab_split.py`) and pushed here with `crab_data/s/sync_crab2226.sh`.

**Add a system:** create `job/systems/<name>.env` (copy `crab.env`) and `config/<LABEL>_taxonomy.json` (copy `crabdata_taxonomy.json`). No script edits.

**Taxonomy rules:** class ids come from the taxonomy JSON, never from data-prep scripts. Genus-only classes (e.g. `cancer_sp`) get a real `species_id` and `is_resolved_species: false`. If a top-level `species_id_to_name` / `genus_id_to_name` is present it must match `classes` (checked on every load).

## Scallop (single-stage)

`job/gcp_yolo_scallop_job.sh`, `job/gcp_eval_scallop.sh` (unchanged from `main`; reproduce paper results from tag `v1.1-revision1`).
