# nwb-eval-capsule

A Code Ocean capsule that evaluates NWB (Neurodata Without Borders) files for
quality, completeness, and best-practices compliance using an LLM.

## What it does

1. Discovers NWB files from attached data assets (or from S3 paths provided as
   parameters)
2. Extracts metadata (HDF5/Zarr attributes) and internal paths from each file
   using [lazynwb](https://github.com/AllenNeuralDynamics/lazynwb)
3. Sends the extracted data to Claude, which evaluates the files across six
   criteria and writes structured markdown reports

### Evaluation criteria

| Criterion | What it checks |
|---|---|
| **Cross-file consistency** | Structural differences, attribute mismatches, and naming divergence across files in the same dataset |
| **Naming conventions** | Mixed naming styles, non-descriptive names, redundant paths, inconsistent abbreviations |
| **Usability & self-documentation** | Whether an unfamiliar scientist could understand the data from metadata alone |
| **Missing metadata** | Subject info, session-level metadata, device info, electrode metadata, etc. |
| **NWB best practices** | Container type usage, raw/processed separation, time series alignment, DynamicTable usage |
| **Data completeness** | Empty containers, orphaned references, incomplete time series, mismatched dimensions |

Each criterion produces a report with per-finding severity ratings (PASS / LOW /
MEDIUM / HIGH / CRITICAL) and actionable recommendations.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `nwb_paths` | All `.nwb` files in attached assets | Comma-separated list of S3 paths to NWB files to evaluate |
| `llm_model` | `claude-sonnet-4-6-20250514` | Anthropic model ID to use for evaluation |
| `logging_level` | `INFO` | Python logging level |
| `sample_n_files` | *(all files)* | If set, randomly sample this many NWB files instead of evaluating all |

An `ANTHROPIC_API_KEY` must be available as a secret (via `.env` or environment
variable). If missing, metadata extraction still runs but LLM summaries are
skipped.

## Output

Results are written to `/root/capsule/results/`:

```
results/
  attrs/           # Per-file JSON: internal paths -> attributes
  internal_paths/  # Per-file JSON: list of internal paths
  summary/         # Per-criterion markdown evaluation reports
```
