"""Launch the nwb-eval capsule on Code Ocean with S3 NWB paths or a data asset."""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "codeocean>=0.14.0",
#   "pydantic-settings>=2.13.1",
# ]
# ///

from __future__ import annotations

import logging
from typing import ClassVar

import pydantic
import pydantic_settings

import codeocean
import codeocean.computation
import codeocean.data_asset

logger = logging.getLogger(__name__)


class LaunchConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,
        cli_kebab_case="all",
        cli_implicit_flags=True,
        populate_by_name=True,
    )

    codeocean_token: pydantic.SecretStr = pydantic.Field(
        alias="code_ocean_api_token",
        description="Code Ocean API access token",
    )
    codeocean_domain: str = pydantic.Field(
        alias="code_ocean_domain",
        default="https://codeocean.allenneuraldynamics.org",
        description="Code Ocean domain URL",
    )
    capsule_id: str = pydantic.Field(
        default="dae9c3fd-5893-4082-b78a-a7136a7fd78d",
        description="ID of the nwb-eval capsule to run",
    )
    nwb_s3_paths: list[str] = pydantic.Field(
        default_factory=list,
        description="S3 paths to NWB files, passed to the capsule as the NWB_PATHS env var. If not provided, a data asset ID must be provided.",
    )
    data_asset_id: str | None = pydantic.Field(
        default=None,
        description="Code Ocean data asset ID to attach (mounted at its name under /data)",
    )
    capsule_version: int | None = pydantic.Field(
        default=None,
        description="Specific capsule version to run (latest if omitted)",
    )
    timeout: float | None = pydantic.Field(
        default=None,
        description="Max seconds to wait for the computation to finish (None = no limit)",
    )
    polling_interval: float = pydantic.Field(
        default=30,
        description="Seconds between status checks while waiting",
    )

    # Capsule parameters (forwarded as named_parameters to the run)
    sample_n_files: int | None = pydantic.Field(
        default=None,
        description="Randomly sample this many NWB files (default: capsule decides)",
    )
    llm_model: str | None = pydantic.Field(
        default=None,
        description="Anthropic model ID for evaluation (default: capsule decides)",
    )
    logging_level: str | None = pydantic.Field(
        default=None,
        description="Python logging level for the capsule (default: INFO)",
    )

    CAPSULE_PARAM_FIELDS: ClassVar[tuple[str, ...]] = (
        "sample_n_files",
        "llm_model",
        "logging_level",
    )


def _get_data_asset_name(client: codeocean.CodeOcean, data_asset_id: str) -> str:
    """Fetch the data asset's name to use as its mount path."""
    asset = client.data_assets.get_data_asset(data_asset_id)
    logger.info(f"Data asset name: {asset.name!r}")
    return asset.name


def launch(config: LaunchConfig) -> codeocean.computation.Computation:
    client = codeocean.CodeOcean(
        domain=config.codeocean_domain,
        token=config.codeocean_token.get_secret_value(),
    )

    data_assets: list[codeocean.computation.DataAssetsRunParam] = []
    if config.data_asset_id is not None:
        mount = _get_data_asset_name(client, config.data_asset_id)
        data_assets.append(
            codeocean.computation.DataAssetsRunParam(id=config.data_asset_id, mount=mount)
        )

    named_parameters: list[codeocean.computation.NamedRunParam] = []
    if config.nwb_s3_paths:
        named_parameters.append(
            codeocean.computation.NamedRunParam(
                param_name="nwb_paths",
                value=",".join(config.nwb_s3_paths),
            )
        )
    for field_name in config.CAPSULE_PARAM_FIELDS:
        value = getattr(config, field_name)
        if value is not None:
            named_parameters.append(
                codeocean.computation.NamedRunParam(param_name=field_name, value=str(value))
            )

    run_params = codeocean.computation.RunParams(
        capsule_id=config.capsule_id,
        version=config.capsule_version,
        data_assets=data_assets or None,
        named_parameters=named_parameters or None,
    )

    logger.info(f"Launching capsule {config.capsule_id}")
    computation = client.computations.run_capsule(run_params)
    logger.info(f"Computation started: id={computation.id} state={computation.state}")

    computation = client.computations.wait_until_completed(
        computation,
        polling_interval=max(config.polling_interval, 5),
        timeout=config.timeout,
    )
    logger.info(f"Computation finished: state={computation.state} end_status={computation.end_status} exit_code={computation.exit_code}")
    return computation


def main() -> None:
    config = LaunchConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not config.nwb_s3_paths and config.data_asset_id is None:
        logger.error("Provide --nwb-s3-paths or --data-asset-id (or both)")
        raise SystemExit(1)

    computation = launch(config)
    if computation.end_status != codeocean.computation.ComputationEndStatus.Succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
