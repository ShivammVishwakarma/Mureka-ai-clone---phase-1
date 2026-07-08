"""Configuration package for the Music Gen Framework.

This package exposes the strongly-typed configuration schema
(:mod:`config.schema`) and the runtime settings loader
(:mod:`config.settings`) used throughout the framework.

Example:
    >>> from config import get_settings
    >>> settings = get_settings()
    >>> settings.audio.sample_rate
    44100
"""

from __future__ import annotations

from config.schema import (
    AudioConfig,
    ColabConfig,
    Config,
    LoggingConfig,
    ModelConfig,
    PathConfig,
    PrecisionMode,
    RuntimeEnvironment,
    TrainingConfig,
)
from config.settings import get_settings, load_settings, reset_settings

__all__ = [
    "AudioConfig",
    "ColabConfig",
    "Config",
    "LoggingConfig",
    "ModelConfig",
    "PathConfig",
    "PrecisionMode",
    "RuntimeEnvironment",
    "TrainingConfig",
    "get_settings",
    "load_settings",
    "reset_settings",
]
