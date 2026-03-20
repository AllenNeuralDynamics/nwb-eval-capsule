from __future__ import annotations

import json
import logging
import random

import anthropic
import lazynwb
import pydantic
import pydantic_settings
import upath

logger = logging.getLogger(__name__)

RESULTS_DIR = upath.UPath("/root/capsule/results")

SUMMARY_SYSTEM_PROMPT = """\
You are an expert in the Neurodata Without Borders (NWB) file format, neuroscience data management, and FAIR data principles.

You will be given metadata and internal paths extracted from one or more NWB files from the same project. Your job is to evaluate them according to a specific criterion and produce a structured report.

## Input format
- **Internal path files** (`internal_paths_*.json`): For each NWB file, a JSON object whose keys are the internal paths to data containers. These represent the data hierarchy within the file.
- **Metadata files** (`attrs_*.json`): For each NWB file, a JSON object mapping internal HDF5/Zarr paths to their attributes (key-value pairs). These represent the metadata attached to each container in the file, for example: timeseries units, scaling factors, table column descriptions.

## Output format
You MUST use exactly this structure (markdown). Do not add extra sections or change headings.

```
## Overall Assessment
[1-3 sentences summarizing the evaluation results for this criterion]

**Severity**: [one of: PASS | LOW | MEDIUM | HIGH | CRITICAL]

## Findings

### 1. [Short title]
- **Severity**: [PASS | LOW | MEDIUM | HIGH | CRITICAL]
- **Affected paths**: [list the specific internal paths or attributes involved]
- **Files affected**: [which NWB file(s), or "all"]
- **Issue**: [clear description of what is wrong or suboptimal]
- **Recommendation**: [specific, actionable fix the scientist can implement]

### 2. [Short title]
...
(repeat for each finding, or write "No issues found." if the files pass this check)

## Summary
- Total findings: [N]
- Critical: [N] | High: [N] | Medium: [N] | Low: [N]
```

## Severity definitions
- **PASS**: No issues found for this criterion.
- **LOW**: Minor cosmetic or style issue. Not blocking but would improve clarity.
- **MEDIUM**: Non-trivial issue that reduces data usability or discoverability. Should be fixed before sharing.
- **HIGH**: Significant problem that could cause misinterpretation or prevent reuse. Must be fixed.
- **CRITICAL**: Data integrity or correctness concern. May indicate data loss or corruption risk.

## Guidelines
- Be specific: always cite the exact paths, attribute names, and file names.
- Be actionable: every recommendation should tell the scientist what to change and where.
- Be concise: no filler, no praise, no disclaimers. Go straight to findings.
- If a file has no issues for a criterion, say so explicitly with a PASS severity.
- When comparing across files, use tables where they aid clarity.
"""

EVALUATION_PROMPTS: dict[str, str] = {
    "cross_file_consistency": """\
## Criterion: Cross-File Consistency

Evaluate whether the NWB files in this dataset are consistent with each other. Scientists often process files from the same experiment together, so inconsistencies between files cause downstream analysis failures.

Check for:
1. **Structural differences**: Paths that exist in some files but not others (missing containers, extra containers).
2. **Attribute inconsistencies**: The same path having different attributes across files, or the same attribute having different data types or value formats.
3. **Naming divergence**: Equivalent data stored under different path names across files (e.g., `/processing/ecephys` in one file vs `/processing/Ecephys` in another).
4. **Schema version mismatches**: Different `neurodata_type` values or NWB version attributes across files.

For each inconsistency, indicate which files diverge and what the expected uniform structure should be.
If only one NWB file is provided, note that cross-file consistency cannot be evaluated and assign PASS.
""",
    "naming_conventions": """\
## Criterion: Naming Conventions & Internal Consistency

Evaluate whether the naming of paths, containers, columns, and attributes within each NWB file follows consistent, readable conventions.

Check for:
1. **Mixed naming styles**: Paths or columns mixing camelCase, snake_case, PascalCase, or UPPER_CASE inconsistently within the same file or table.
2. **Abbreviations vs full words**: Inconsistent use of abbreviations (e.g., `stim` vs `stimulus`, `resp` vs `response`) across sibling paths.
3. **Redundant or overly nested paths**: Paths with unnecessary repetition (e.g., `/processing/ecephys/LFP/LFP`).
4. **Non-descriptive generic names**: Containers named `data_0`, `temp`, `unnamed`, or similar placeholders.
5. **Inconsistent pluralization**: Mixing singular and plural forms for equivalent container types (e.g., `electrode` vs `electrodes` in sibling paths).

For each finding, suggest a specific corrected name following NWB/HDF5 conventions (snake_case for user-defined names, PascalCase for neurodata_type names).
""",
    "usability": """\
## Criterion: Usability & Self-Documentation

Evaluate whether a scientist unfamiliar with this dataset could understand the contents of these NWB files from the metadata and structure alone.

Check for:
1. **Opaque path or column names**: Names that require domain-specific insider knowledge to understand. Acceptable NWB-standard names (like `units`, `electrodes`, `stimulus`) should NOT be flagged.
2. **Missing descriptions**: Containers or columns that lack a `description` attribute, especially for custom/non-standard data.
3. **Unexplained units**: Data paths with a `unit` or `units` attribute that is missing, empty, or ambiguous (e.g., `a.u.`).
4. **Unclear processing provenance**: Processed data containers (under `/processing/`) that don't describe what processing was applied.
5. **Missing experiment context**: Whether the file-level metadata (session description, experiment description, subject info) provides enough context to understand what the data represents.

Focus on what would trip up a new lab member or external collaborator trying to reuse this data.
""",
    "missing_metadata": """\
## Criterion: Missing Metadata

Evaluate whether the NWB files contain the metadata necessary for data reuse, reproducibility, and compliance with FAIR data principles.

Check for:
1. **Empty or missing subject metadata**: Species, age, sex, genotype, subject_id - especially critical for shared datasets.
2. **Missing session-level metadata**: session_description, experiment_description, session_start_time, institution, lab, experimenter.
3. **Missing container descriptions**: Any container (group or dataset) without a `description` attribute, particularly custom containers not defined by the NWB core schema.
4. **Missing electrode/channel metadata**: For electrophysiology data, check for electrode locations, filtering info, impedance, and group assignments.
5. **Missing stimulus metadata**: For stimulus data, check for descriptions of stimulus parameters, timing, and conditions.
6. **Missing device information**: Whether devices referenced in the file have manufacturer, description, or other identifying metadata.

Rank missing metadata by importance: metadata needed for correct interpretation first, then metadata needed for reproducibility, then nice-to-have context.
""",
    "nwb_best_practices": """\
## Criterion: NWB Best Practices Compliance

Evaluate whether the NWB files follow community best practices for NWB file organization and data storage.

Check for:
1. **Flat vs hierarchical organization**: Data that should be grouped into processing modules but is placed at unexpected locations in the hierarchy.
2. **Misuse of container types**: Data stored in generic containers when a specific NWB neurodata_type exists for it (visible from neurodata_type attributes).
3. **Raw vs processed separation**: Whether raw and processed data are properly separated (raw in `/acquisition/`, processed in `/processing/`).
4. **Time series alignment**: Whether TimeSeries containers have appropriate timestamps or starting_time + rate attributes (check for presence of these attributes, not their values).
5. **Proper use of DynamicTable**: Whether tabular data uses DynamicTable with appropriate column descriptions rather than ad-hoc structures.
6. **Extension usage**: Whether custom neurodata_types (non-core extensions) are clearly identified and whether standard types could have been used instead.

Focus on issues that affect interoperability with NWB-compatible analysis tools (e.g., NWBWidgets, pynapple, CaImAn).
""",
    "data_completeness": """\
## Criterion: Data Completeness & Structural Integrity

Evaluate whether the NWB files appear to contain complete, well-formed data based on their structure and metadata.

Check for:
1. **Empty containers**: Paths that exist in the hierarchy but have no data or no child containers, suggesting incomplete writes or failed processing.
2. **Orphaned references**: Containers that reference other paths (e.g., electrode_table_region) where the referenced target is missing or inconsistent.
3. **Incomplete time series**: TimeSeries containers missing expected companion datasets (e.g., `data` without `timestamps`, or `timestamps` without corresponding `data`).
4. **Mismatched table dimensions**: DynamicTable columns that likely should have the same length but appear structurally inconsistent.
5. **Missing index arrays**: Ragged array columns (VectorData) without corresponding VectorIndex entries, or vice versa.
6. **Dangling processing modules**: Processing modules that exist but contain no data interfaces.

Focus on issues that would cause errors when loading or analyzing the data programmatically.
""",
}


def get_nwb_paths_from_attached_assets() -> list[upath.UPath]:
    return list(upath.UPath("/root/capsule/data").rglob("*.nwb*"))


class Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    nwb_paths: list[upath.UPath] = pydantic.Field(
        default_factory=get_nwb_paths_from_attached_assets,
        description="Comma-separated list of S3 paths to NWB files to evaluate. By default, will include all NWB files found in the attached assets.",
    )
    logging_level: str = "INFO"
    llm_model: str = "claude-sonnet-4-6-20250514"
    anthropic_api_key: pydantic.SecretStr | None = None
    sample_n_files: int | None = pydantic.Field(
        default=None,
        description="If set, randomly sample this many NWB files from the discovered paths. If None or greater than the number of available files, all files are used.",
    )

def write_to_json(data: dict, path: upath.UPath, prefix: str) -> upath.UPath:
    output_dir = RESULTS_DIR / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.json"
    logger.info(f"Writing to {output_path}")
    output_path.write_text(json.dumps(data, indent=2))
    return output_path

def _build_file_context() -> str:
    """Read all previously-written JSON results and format them as LLM context."""
    sections: list[str] = []
    for subfolder, heading in [
        ("attrs", "Metadata (attributes per internal path)"),
        ("internal_paths", "Internal paths (data containers)"),
    ]:
        files = sorted((RESULTS_DIR / subfolder).glob("*.json")) if (RESULTS_DIR / subfolder).exists() else []
        if not files:
            continue
        sections.append(f"# {heading}\n")
        for f in files:
            sections.append(f"## File: {f.stem}\n```json\n{f.read_text()}\n```\n")
    return "\n".join(sections)


def write_llm_summaries(model: str, config: Config) -> None:
    """Use an LLM to evaluate NWB files and write structured summaries."""
    if config.anthropic_api_key is None:
        logger.warning("LLM credentials not found. Skipping LLM summaries.")
        return
    
    file_context = _build_file_context()
    if not file_context.strip():
        logger.warning("No JSON results found to summarize.")
        return

    client = anthropic.Anthropic(api_key=config.anthropic_api_key.get_secret_value())

    for category, prompt in EVALUATION_PROMPTS.items():
        logger.info(f"Generating LLM summary: {category}")
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n---\n\n{file_context}",
                },
            ],
        )
        text_blocks = [block.text for block in message.content if isinstance(block, anthropic.types.TextBlock)]
        if not text_blocks:
            logger.warning(f"No text in LLM response for {category}")
            continue
        summary_dir = RESULTS_DIR / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        output_path = summary_dir / f"{category}.md"
        output_path.write_text("\n\n".join(text_blocks))
        logger.info(f"Wrote summary to {output_path}")


def main():
    config = Config()
    logging.basicConfig(level=config.logging_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.info(f"Running with config: {config}")
    nwb_paths = config.nwb_paths
    if len(nwb_paths) == 0:
        logger.warning("No remote NWB paths provided and no NWBs found in attached assets. Exiting.")
        return
    if config.sample_n_files is not None and config.sample_n_files < len(nwb_paths):
        nwb_paths = random.sample(nwb_paths, config.sample_n_files)
        logger.info(f"Sampled {config.sample_n_files} of {len(config.nwb_paths)} files")
    for path in nwb_paths:
        logger.info(f"Processing NWB file at path: {path}")
        write_to_json(lazynwb.get_sub_attrs(path, exclude_private=True, exclude_empty=True), path, "attrs")
        write_to_json(list(lazynwb.get_internal_paths(path).keys()), path, "internal_paths")
    write_llm_summaries(model=config.llm_model, config=config)


if __name__ == "__main__":
    main()