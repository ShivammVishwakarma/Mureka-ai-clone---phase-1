"""Audio input/output utilities.

Provides safe, validated loading and saving of audio files backed by
``soundfile`` and ``torchaudio``. This module deals exclusively with data
plumbing (reading, writing, resampling, channel/format normalization) and
contains no model or generation logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from utils.validation import (
    ValidationError,
    validate_file_extension,
    validate_path_exists,
    validate_positive,
    validate_range,
)

_logger = logging.getLogger(__name__)

_DEFAULT_SUPPORTED_FORMATS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be loaded or is invalid."""


class AudioSaveError(RuntimeError):
    """Raised when an audio tensor cannot be written to disk."""


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Metadata describing a loaded audio clip.

    Attributes:
        path: Source file path.
        sample_rate: Sample rate of the returned waveform, in Hz (after any
            requested resampling).
        original_sample_rate: Sample rate of the file on disk, before
            resampling.
        num_channels: Number of channels in the returned waveform.
        num_frames: Number of samples per channel in the returned waveform.
        duration_seconds: Duration of the returned waveform, in seconds.
        was_resampled: Whether resampling was performed.
    """

    path: Path
    sample_rate: int
    original_sample_rate: int
    num_channels: int
    num_frames: int
    duration_seconds: float
    was_resampled: bool


def get_audio_info(path: str | Path) -> AudioMetadata:
    """Inspect an audio file's metadata without loading full sample data.

    Args:
        path: Path to the audio file.

    Returns:
        An :class:`AudioMetadata` instance reflecting the file *as stored on
        disk* (``was_resampled`` is always ``False`` here).

    Raises:
        AudioLoadError: If the file does not exist or cannot be inspected.
    """
    resolved = validate_path_exists(path, field_name="path", must_be_file=True)
    try:
        info = sf.info(str(resolved))
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise AudioLoadError(f"Could not read audio metadata for '{resolved}': {exc}") from exc

    return AudioMetadata(
        path=resolved,
        sample_rate=info.samplerate,
        original_sample_rate=info.samplerate,
        num_channels=info.channels,
        num_frames=info.frames,
        duration_seconds=round(info.frames / info.samplerate, 6) if info.samplerate else 0.0,
        was_resampled=False,
    )


def load_audio(
    path: str | Path,
    *,
    target_sample_rate: Optional[int] = None,
    mono: bool = False,
    normalize: bool = False,
    target_loudness_db: float = -20.0,
    max_duration_seconds: Optional[float] = None,
    supported_formats: tuple[str, ...] = _DEFAULT_SUPPORTED_FORMATS,
) -> tuple[torch.Tensor, AudioMetadata]:
    """Load an audio file into a normalized ``torch.Tensor`` waveform.

    Args:
        path: Path to the audio file.
        target_sample_rate: If provided, the waveform is resampled to this
            rate using ``torchaudio``'s high-quality resampler.
        mono: If True, downmix multi-channel audio to a single channel by
            averaging across channels.
        normalize: If True, peak- and loudness-adjust the waveform toward
            ``target_loudness_db`` (approximate RMS-based normalization).
        target_loudness_db: Target loudness in dBFS, used only if
            ``normalize`` is True.
        max_duration_seconds: If provided, raises :class:`AudioLoadError` for
            clips exceeding this duration (checked on the *original* file
            duration, before any resampling).
        supported_formats: File extensions this function will accept.

    Returns:
        A tuple ``(waveform, metadata)`` where ``waveform`` has shape
        ``(channels, num_samples)`` and dtype ``torch.float32`` in the
        range ``[-1, 1]``, and ``metadata`` describes the returned waveform.

    Raises:
        AudioLoadError: If the file is missing, has an unsupported format,
            exceeds ``max_duration_seconds``, or cannot be decoded.
        ValidationError: If arguments are out of valid ranges.
    """
    if target_sample_rate is not None:
        validate_positive(target_sample_rate, field_name="target_sample_rate")
    if max_duration_seconds is not None:
        validate_positive(max_duration_seconds, field_name="max_duration_seconds")
    validate_range(target_loudness_db, field_name="target_loudness_db", minimum=-60.0, maximum=0.0)

    resolved = validate_path_exists(path, field_name="path", must_be_file=True)
    validate_file_extension(resolved, field_name="path", allowed_extensions=supported_formats)

    original_info = get_audio_info(resolved)
    if max_duration_seconds is not None and original_info.duration_seconds > max_duration_seconds:
        raise AudioLoadError(
            f"Audio '{resolved}' duration {original_info.duration_seconds:.2f}s exceeds "
            f"max_duration_seconds={max_duration_seconds:.2f}s"
        )

    try:
        waveform, original_sample_rate = torchaudio.load(str(resolved))
    except Exception as exc:  # torchaudio raises varied backend-specific errors
        raise AudioLoadError(f"Failed to decode audio file '{resolved}': {exc}") from exc

    if waveform.numel() == 0:
        raise AudioLoadError(f"Audio file '{resolved}' contains no samples")

    waveform = waveform.to(dtype=torch.float32)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    was_resampled = False
    final_sample_rate = original_sample_rate
    if target_sample_rate is not None and target_sample_rate != original_sample_rate:
        waveform = resample_audio(waveform, original_sample_rate, target_sample_rate)
        final_sample_rate = target_sample_rate
        was_resampled = True

    if normalize:
        waveform = normalize_audio(waveform, target_loudness_db=target_loudness_db)

    num_channels, num_frames = waveform.shape
    metadata = AudioMetadata(
        path=resolved,
        sample_rate=final_sample_rate,
        original_sample_rate=original_sample_rate,
        num_channels=num_channels,
        num_frames=num_frames,
        duration_seconds=round(num_frames / final_sample_rate, 6) if final_sample_rate else 0.0,
        was_resampled=was_resampled,
    )

    _logger.debug(
        "Loaded audio '%s': %d channel(s), %d Hz, %.2fs (resampled=%s).",
        resolved,
        metadata.num_channels,
        metadata.sample_rate,
        metadata.duration_seconds,
        was_resampled,
    )
    return waveform, metadata


def resample_audio(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample a waveform tensor to a new sample rate.

    Args:
        waveform: Tensor of shape ``(channels, num_samples)``.
        orig_freq: Original sample rate, in Hz.
        new_freq: Target sample rate, in Hz.

    Returns:
        The resampled waveform tensor of shape ``(channels, new_num_samples)``.

    Raises:
        ValidationError: If ``orig_freq`` or ``new_freq`` are not positive.
    """
    validate_positive(orig_freq, field_name="orig_freq")
    validate_positive(new_freq, field_name="new_freq")

    if orig_freq == new_freq:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(waveform)


def normalize_audio(
    waveform: torch.Tensor, *, target_loudness_db: float = -20.0, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize a waveform toward a target RMS loudness, with peak safety.

    Args:
        waveform: Tensor of shape ``(channels, num_samples)``.
        target_loudness_db: Desired loudness in dBFS (RMS-based approximation).
        eps: Small constant to avoid division by zero on silent input.

    Returns:
        The normalized waveform, peak-clamped to ``[-1, 1]``.

    Raises:
        ValidationError: If ``target_loudness_db`` is out of the valid range.
    """
    validate_range(target_loudness_db, field_name="target_loudness_db", minimum=-60.0, maximum=0.0)

    rms = torch.sqrt(torch.mean(waveform**2) + eps)
    if rms.item() <= eps:
        _logger.debug("Waveform is near-silent; skipping loudness normalization.")
        return waveform

    target_rms = 10.0 ** (target_loudness_db / 20.0)
    gain = target_rms / rms
    normalized = waveform * gain
    peak = normalized.abs().max()
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.clamp(-1.0, 1.0)


def save_audio(
    waveform: torch.Tensor,
    path: str | Path,
    sample_rate: int,
    *,
    bit_depth: int = 16,
    overwrite: bool = False,
) -> Path:
    """Save a waveform tensor to disk as a WAV/FLAC file.

    Args:
        waveform: Tensor of shape ``(channels, num_samples)`` with values in
            ``[-1, 1]``.
        path: Destination file path. Parent directories are created if
            needed.
        sample_rate: Sample rate to embed in the output file, in Hz.
        bit_depth: PCM bit depth (16, 24, or 32).
        overwrite: If False, raises :class:`AudioSaveError` when the
            destination file already exists.

    Returns:
        The resolved output path.

    Raises:
        ValidationError: If arguments are invalid (bad shape, bit depth,
            sample rate).
        AudioSaveError: If the destination exists and ``overwrite`` is
            False, or if writing fails.
    """
    validate_positive(sample_rate, field_name="sample_rate")
    if bit_depth not in (16, 24, 32):
        raise ValidationError(f"bit_depth must be 16, 24, or 32, got {bit_depth}")
    if waveform.ndim != 2:
        raise ValidationError(
            f"waveform must have shape (channels, num_samples), got {tuple(waveform.shape)}"
        )

    resolved = Path(path)
    if resolved.exists() and not overwrite:
        raise AudioSaveError(f"Destination file already exists (overwrite=False): {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)

    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
    numpy_waveform: np.ndarray = waveform.detach().cpu().clamp(-1.0, 1.0).numpy().T

    try:
        sf.write(
            file=str(resolved),
            data=numpy_waveform,
            samplerate=sample_rate,
            subtype=subtype_map[bit_depth],
        )
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise AudioSaveError(f"Failed to write audio file '{resolved}': {exc}") from exc

    _logger.debug(
        "Saved audio to '%s' (%d ch, %d Hz, %d-bit).",
        resolved,
        numpy_waveform.shape[1] if numpy_waveform.ndim > 1 else 1,
        sample_rate,
        bit_depth,
    )
    return resolved
