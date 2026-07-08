"""Strongly-typed configuration schema for the Music Gen Framework.

All runtime configuration is expressed as immutable (frozen) ``dataclasses``.
Each dataclass validates its own fields in ``__post_init__`` so that invalid
configuration fails fast, at construction time, rather than deep inside a
training or inference loop.

This module intentionally has **zero** dependency on ``torch`` so that it can
be imported in lightweight contexts (e.g. config validation in CI) without
pulling in the full deep-learning stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ConfigValidationError(ValueError):
    """Raised when a configuration dataclass receives invalid values."""


class RuntimeEnvironment(str, Enum):
    """Detected or forced runtime environment."""

    LOCAL = "local"
    COLAB = "colab"
    KAGGLE = "kaggle"
    CI = "ci"
    UNKNOWN = "unknown"


class PrecisionMode(str, Enum):
    """Supported numerical precision modes for training/inference."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Supported logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio processing configuration.

    Attributes:
        sample_rate: Target sample rate in Hz for all audio I/O.
        num_channels: Number of audio channels (1 = mono, 2 = stereo).
        bit_depth: Bit depth used when writing PCM audio to disk.
        max_duration_seconds: Hard cap on clip duration accepted by the
            framework, used to guard against runaway memory usage.
        min_duration_seconds: Minimum clip duration considered valid.
        normalize: Whether audio should be peak-normalized on load.
        target_loudness_db: Target loudness (dBFS) when ``normalize`` is True.
        supported_formats: File extensions accepted by the audio I/O layer.
    """

    sample_rate: int = 44_100
    num_channels: int = 2
    bit_depth: int = 16
    max_duration_seconds: float = 300.0
    min_duration_seconds: float = 0.1
    normalize: bool = True
    target_loudness_db: float = -20.0
    supported_formats: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigValidationError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.sample_rate not in (8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000):
            raise ConfigValidationError(
                f"sample_rate {self.sample_rate} is unusual; expected a standard rate "
                "(8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000)."
            )
        if self.num_channels not in (1, 2):
            raise ConfigValidationError(
                f"num_channels must be 1 (mono) or 2 (stereo), got {self.num_channels}"
            )
        if self.bit_depth not in (16, 24, 32):
            raise ConfigValidationError(f"bit_depth must be 16, 24, or 32, got {self.bit_depth}")
        if self.max_duration_seconds <= 0:
            raise ConfigValidationError("max_duration_seconds must be positive")
        if self.min_duration_seconds < 0:
            raise ConfigValidationError("min_duration_seconds must be non-negative")
        if self.min_duration_seconds >= self.max_duration_seconds:
            raise ConfigValidationError(
                "min_duration_seconds must be strictly less than max_duration_seconds"
            )
        if not (-60.0 <= self.target_loudness_db <= 0.0):
            raise ConfigValidationError(
                f"target_loudness_db should be in [-60, 0] dBFS, got {self.target_loudness_db}"
            )
        if not self.supported_formats:
            raise ConfigValidationError("supported_formats must not be empty")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model architecture and precision configuration.

    Note:
        Phase 1 defines *only* the configuration surface. No model
        architecture or generation logic is implemented here.

    Attributes:
        name: Identifier for the model configuration (used for checkpoint
            naming and logging), not tied to any specific architecture.
        hidden_size: Width of the model's hidden representations.
        num_layers: Depth of the model.
        num_attention_heads: Number of attention heads, must evenly divide
            ``hidden_size``.
        dropout: Dropout probability applied within the model.
        precision: Requested numerical precision mode.
        max_sequence_length: Maximum token/frame sequence length supported.
        gradient_checkpointing: Whether to trade compute for memory.
    """

    name: str = "base"
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    dropout: float = 0.1
    precision: PrecisionMode = PrecisionMode.AUTO
    max_sequence_length: int = 4096
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ConfigValidationError("name must be a non-empty string")
        if self.hidden_size <= 0:
            raise ConfigValidationError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ConfigValidationError("num_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ConfigValidationError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigValidationError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ConfigValidationError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.max_sequence_length <= 0:
            raise ConfigValidationError("max_sequence_length must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training loop hyperparameters.

    Attributes:
        batch_size: Per-step batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay coefficient.
        num_epochs: Total number of training epochs.
        warmup_steps: Number of linear warmup steps for the LR scheduler.
        gradient_accumulation_steps: Number of micro-batches accumulated
            before an optimizer step, used to simulate larger batch sizes.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        seed: Global random seed for reproducibility.
        checkpoint_every_n_steps: Frequency of checkpoint writes.
        num_dataloader_workers: Number of subprocess workers for data loading.
    """

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1_000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_every_n_steps: int = 1_000
    num_dataloader_workers: int = 2

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ConfigValidationError("weight_decay must be non-negative")
        if self.num_epochs <= 0:
            raise ConfigValidationError("num_epochs must be positive")
        if self.warmup_steps < 0:
            raise ConfigValidationError("warmup_steps must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigValidationError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ConfigValidationError("max_grad_norm must be positive")
        if self.seed < 0:
            raise ConfigValidationError("seed must be non-negative")
        if self.checkpoint_every_n_steps <= 0:
            raise ConfigValidationError("checkpoint_every_n_steps must be positive")
        if self.num_dataloader_workers < 0:
            raise ConfigValidationError("num_dataloader_workers must be non-negative")


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Filesystem layout configuration.

    Attributes:
        root_dir: Project root directory.
        data_dir: Location of raw/processed datasets.
        checkpoint_dir: Location for model checkpoints.
        log_dir: Location for log files.
        cache_dir: Location for cached artifacts (e.g. resampled audio).
        output_dir: Location for generated artifacts / exports.
    """

    root_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    checkpoint_dir: Path = field(default_factory=lambda: Path.cwd() / "checkpoints")
    log_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / ".cache")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")

    def __post_init__(self) -> None:
        for attr_name in (
            "root_dir",
            "data_dir",
            "checkpoint_dir",
            "log_dir",
            "cache_dir",
            "output_dir",
        ):
            value = getattr(self, attr_name)
            if not isinstance(value, Path):
                object.__setattr__(self, attr_name, Path(value))

    def ensure_directories(self) -> None:
        """Create all configured directories if they do not already exist."""
        for attr_name in ("data_dir", "checkpoint_dir", "log_dir", "cache_dir", "output_dir"):
            path: Path = getattr(self, attr_name)
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class ColabConfig:
    """Google Colab-specific runtime configuration.

    Attributes:
        mount_drive: Whether Google Drive should be mounted for persistent
            storage of datasets/checkpoints.
        drive_mount_point: Local mount point for Google Drive.
        drive_project_subdir: Subdirectory within Drive used to store
            project artifacts, relative to the Drive mount point.
        use_tpu: Whether to prefer a TPU runtime over GPU if available.
        colab_high_ram: Hint that a "High-RAM" Colab runtime is expected;
            used only for informational logging, not enforcement.
    """

    mount_drive: bool = False
    drive_mount_point: Path = field(default_factory=lambda: Path("/content/drive"))
    drive_project_subdir: str = "MyDrive/music-gen-framework"
    use_tpu: bool = False
    colab_high_ram: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.drive_mount_point, Path):
            object.__setattr__(self, "drive_mount_point", Path(self.drive_mount_point))
        if not self.drive_project_subdir or not self.drive_project_subdir.strip():
            raise ConfigValidationError("drive_project_subdir must be a non-empty string")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging configuration.

    Attributes:
        level: Minimum log level emitted to handlers.
        log_to_file: Whether logs should also be written to disk.
        log_filename: Filename used when ``log_to_file`` is True.
        use_rich: Whether to use ``rich`` for colorized console output
            (falls back to standard formatting if ``rich`` is unavailable).
        json_format: Whether file logs should be emitted as JSON lines,
            useful for downstream log aggregation.
    """

    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_filename: str = "music_gen_framework.log"
    use_rich: bool = True
    json_format: bool = False

    def __post_init__(self) -> None:
        if not self.log_filename or not self.log_filename.strip():
            raise ConfigValidationError("log_filename must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Config:
    """Root configuration object aggregating all configuration sections.

    Attributes:
        environment: Detected or forced runtime environment.
        audio: Audio processing configuration.
        model: Model architecture configuration.
        training: Training loop configuration.
        paths: Filesystem layout configuration.
        colab: Google Colab-specific configuration.
        logging: Logging configuration.
        debug: Global debug flag; when True, components may enable extra
            assertions and verbose diagnostics.
    """

    environment: RuntimeEnvironment = RuntimeEnvironment.UNKNOWN
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: bool = False

    def __post_init__(self) -> None:
        for section_name in ("audio", "model", "training", "paths", "colab", "logging"):
            section = getattr(self, section_name)
            if section is None:
                raise ConfigValidationError(f"Configuration section '{section_name}' cannot be None")

    def summary(self) -> dict[str, object]:
        """Return a flat, human-readable summary of key configuration values.

        Returns:
            A dictionary suitable for logging at startup.
        """
        return {
            "environment": self.environment.value,
            "sample_rate": self.audio.sample_rate,
            "model_name": self.model.name,
            "precision": self.model.precision.value,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "seed": self.training.seed,
            "root_dir": str(self.paths.root_dir),
            "debug": self.debug,
        }


def detect_runtime_environment() -> RuntimeEnvironment:
    """Best-effort detection of the current runtime environment.

    Returns:
        The detected :class:`RuntimeEnvironment`. Falls back to
        :attr:`RuntimeEnvironment.UNKNOWN` if detection is inconclusive.
    """
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return RuntimeEnvironment.COLAB
    if Path("/kaggle/working").exists():
        return RuntimeEnvironment.KAGGLE
    if os.environ.get("CI", "").lower() == "true":
        return RuntimeEnvironment.CI
    try:
        import sys

        if "google.colab" in sys.modules:
            return RuntimeEnvironment.COLAB
    except Exception:  # pragma: no cover - defensive
        pass
    if Path("/content").exists():
        return RuntimeEnvironment.COLAB
    return RuntimeEnvironment.LOCAL if Path.cwd().exists() else RuntimeEnvironment.UNKNOWN
