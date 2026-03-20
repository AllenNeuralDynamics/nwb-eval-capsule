import os
import json
import logging

import lazynwb
import pydantic
import pydantic_settings
import upath

logger = logging.getLogger(__name__)

def get_nwb_paths_from_attached_assets() -> list[upath.UPath]:
    return list(upath.UPath("/root/capsule/data").rglob("*.nwb*"))

class Config(pydantic_settings.BaseSettings):
    nwb_paths: list[upath.UPath] = pydantic.Field(default_factory=get_nwb_paths_from_attached_assets)
    logging_level: str = "INFO"

def write_to_json(data: dict, path: upath.UPath, prefix: str) -> None:
    output_path = upath.UPath("/root/capsule/results") / f"{prefix}_{path.stem}.json"
    logger.info(f"Writing to {output_path}")
    output_path.write_text(json.dumps(data, indent=2))

def assert_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

def write_llm_summaries() -> None:
    """Use an LLM to write digestible summaries of: 
        - CROSS-FILE CONSISTENCY
        - INTERNALLY INCONSISTENT NAMING
        - USABILITY
        - MISSING METADATA
    """
    if not assert_credentials():
        logger.warning("LLM credentials not found. Skipping LLM summary.")
        return
    path_to_prompt = {
        "cross_file_consistency": "Identify any discrepancies between internal paths and attributes across the NWB files, such as attributes missing from some files, inconsistent names or data types.",
        "internally_inconsistent_naming": "Identify any internal paths or column names that are inconsistent within files and could be standardized with better naming conventions.",
        "usability": "Identify any internal path or column names that are NOT self-explanatory and could be improved with better naming conventions.",
        "missing_metadata": "Identify any important metadata that is missing from the files, such as descriptions of what certain attributes or internal paths represent.",
    }
    
def main():
    config = Config()
    logging.basicConfig(level=config.logging_level)
    logger.info(f"Running with config: {config}")
    for path in config.nwb_paths:
        logger.info(f"Processing NWB file at path: {path}")
        write_to_json(lazynwb.get_sub_attrs(path, exclude_private=True, exclude_empty=True), path, "attrs")
        write_to_json(lazynwb.get_internal_paths(path), path, "internal_paths")
    write_llm_summaries()

if __name__ == "__main__": 
    main()