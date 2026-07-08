"""Runtime settings loader for the Music Gen Framework.

This module is responsible for *materializing* a :class:`config.schema.Config`
instance at runtime by layering, in increasing priority order:

1. Dataclass field defaults (see :mod:`config.schema`).
2. An optional YAML configuration file.
3. Environment variables (prefixed with ``MGF_``).
4. Explicit keyword overrides passed to :func:`load_settings`.

The resulting :class:`~config.schema.Config` is cached as a process-wide
singleton via :func:`get_settings`, so that repeated calls do not re-parse
files or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from config.schema import (
    AudioConfig,
    ColabConfig,
    Config,
    ConfigValidationError,
    LoggingConfig,
    LogLevel,
    ModelConfig,
    PathConfig,
    PrecisionMode,
    RuntimeEnvironment,
    TrainingConfig,
    detect_runtime_environment,
)

_ENV_PREFIX = "MGF_"
_settings_singleton: Optional[Config] = None


class SettingsLoadError(RuntimeError):
    """Raised when settings cannot be loaded or parsed."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a nested dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary. Returns an empty dict if the
        file is empty.

    Raises:
        SettingsLoadError: If the file cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise SettingsLoadError(f"Could not read config file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise SettingsLoadError(f"Invalid YAML in config file '{path}': {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsLoadError(f"Top-level YAML content in '{path}' must be a mapping")
    return data


def _collect_env_overrides() -> dict[str, dict[str, str]]:
    """Collect environment variable overrides into a nested override map.

    Environment variables are expected in the form::

        MGF_<SECTION>__<FIELD>=value

    For example, ``MGF_AUDIO__SAMPLE_RATE=48000`` overrides
    ``Config.audio.sample_rate``.

    Returns:
        A nested dictionary of the form ``{section: {field: raw_value}}``.
    """
    overrides: dict[str, dict[str, str]] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        remainder = key[len(_ENV_PREFIX) :]
        if "__" not in remainder:
            continue
        section, field_name = remainder.split("__", maxsplit=1)
        section = section.lower()
        field_name = field_name.lower()
        overrides.setdefault(section, {})[field_name] = value
    return overrides


def _coerce_value(raw: str, target_type: type) -> Any:
    """Coerce a raw string (from env or YAML) into the target Python type.

    Args:
        raw: The raw string value.
        target_type: The type the value should be coerced into.

    Returns:
        The coerced value.

    Raises:
        SettingsLoadError: If coercion fails.
    """
    try:
        if target_type is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if target_type is int:
            return int(raw)
        if target_type is float:
            return float(raw)
        if target_type is Path:
            return Path(raw)
        if issubclass(target_type, (PrecisionMode, RuntimeEnvironment, LogLevel)):
            return target_type(raw)
        return raw
    except (ValueError, TypeError) as exc:
        raise SettingsLoadError(
            f"Could not coerce value '{raw}' to type {target_type.__name__}: {exc}"
        ) from exc


def _build_section(
    dataclass_type: type,
    yaml_section: dict[str, Any],
    env_section: dict[str, str],
) -> Any:
    """Instantiate a configuration dataclass, layering YAML and env overrides.

    Args:
        dataclass_type: The dataclass to instantiate (e.g. ``AudioConfig``).
        yaml_section: Parsed YAML values for this section.
        env_section: Environment variable overrides for this section.

    Returns:
        An instance of ``dataclass_type``.
    """
    import dataclasses

    field_types = {f.name: f.type for f in dataclasses.fields(dataclass_type)}
    resolved: dict[str, Any] = dict(yaml_section)

    for field_name, raw_value in env_section.items():
        if field_name not in field_types:
            continue
        declared_type = field_types[field_name]
        target_type = declared_type if isinstance(declared_type, type) else str
        resolved[field_name] = _coerce_value(raw_value, target_type)

    try:
        return dataclass_type(**resolved)
    except TypeError as exc:
        raise SettingsLoadError(
            f"Invalid fields for {dataclass_type.__name__}: {exc}"
        ) from exc
    except ConfigValidationError:
        raise


def load_settings(
    config_path: Optional[str | Path] = None,
    *,
    env_file: Optional[str | Path] = None,
    overrides: Optional[dict[str, dict[str, Any]]] = None,
    force_environment: Optional[RuntimeEnvironment] = None,
) -> Config:
    """Build a fully-validated :class:`~config.schema.Config` instance.

    Args:
        config_path: Optional path to a YAML config file. If omitted, only
            defaults, environment variables, and ``overrides`` are used.
        env_file: Optional path to a ``.env`` file to load before reading
            environment variables (uses ``python-dotenv``).
        overrides: Optional nested dict of explicit overrides, of the form
            ``{"audio": {"sample_rate": 48000}}``, applied with the highest
            priority.
        force_environment: Optionally force the detected runtime environment
            instead of auto-detecting it.

    Returns:
        A validated :class:`~config.schema.Config` instance. Note that this
        does **not** update the process-wide singleton; use
        :func:`get_settings` for cached, singleton access.

    Raises:
        SettingsLoadError: If the YAML file is missing/invalid or values
            cannot be coerced to their declared types.
        ConfigValidationError: If the resulting configuration fails
            dataclass-level validation.
    """
    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=False)
    else:
        load_dotenv(override=False)

    yaml_data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise SettingsLoadError(f"Config file not found: {path}")
        yaml_data = _read_yaml(path)

    env_overrides = _collect_env_overrides()
    user_overrides = overrides or {}

    section_map: dict[str, type] = {
        "audio": AudioConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "paths": PathConfig,
        "colab": ColabConfig,
        "logging": LoggingConfig,
    }

    sections: dict[str, Any] = {}
    for section_name, dataclass_type in section_map.items():
        merged_yaml = dict(yaml_data.get(section_name, {}))
        merged_yaml.update(user_overrides.get(section_name, {}))
        sections[section_name] = _build_section(
            dataclass_type,
            merged_yaml,
            env_overrides.get(section_name, {}),
        )

    environment = force_environment or detect_runtime_environment()
    debug_flag = _coerce_value(os.environ.get(f"{_ENV_PREFIX}DEBUG", "false"), bool)

    config = Config(
        environment=environment,
        debug=debug_flag,
        **sections,
    )
    return config


def get_settings(*, reload: bool = False, **load_kwargs: Any) -> Config:
    """Return the process-wide :class:`~config.schema.Config` singleton.

    On first call (or when ``reload=True``), builds the settings via
    :func:`load_settings` and caches the result. Subsequent calls return the
    cached instance without re-reading files or environment variables.

    Args:
        reload: If True, force a rebuild of the cached settings.
        **load_kwargs: Forwarded to :func:`load_settings` on (re)build.

    Returns:
        The cached, validated :class:`~config.schema.Config` instance.
    """
    global _settings_singleton
    if _settings_singleton is None or reload:
        _settings_singleton = load_settings(**load_kwargs)
    return _settings_singleton


def reset_settings() -> None:
    """Clear the cached settings singleton.

    Primarily useful in tests, where each test may want a fresh
    configuration built from a clean environment.
    """
    global _settings_singleton
    _settings_singleton = None
