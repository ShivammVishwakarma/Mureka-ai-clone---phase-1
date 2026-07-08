"""Device and precision management utilities.

Centralizes all hardware-detection logic (CUDA availability, GPU
capabilities, mixed-precision support) behind a single :class:`DeviceManager`
so that the rest of the framework never has to call ``torch.cuda`` directly.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch

from config.schema import PrecisionMode

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Snapshot of the detected compute device and its capabilities.

    Attributes:
        device: The resolved ``torch.device`` to use for tensors/models.
        device_type: Either ``"cuda"``, ``"mps"``, or ``"cpu"``.
        device_name: Human-readable device name (e.g. GPU model string).
        cuda_available: Whether CUDA is available in this environment.
        mps_available: Whether Apple Metal Performance Shaders is available.
        num_gpus: Number of visible CUDA devices (0 if none).
        total_memory_gb: Total device memory in gigabytes, if determinable.
        compute_capability: CUDA compute capability as ``(major, minor)``,
            or ``None`` if not applicable.
        supports_bf16: Whether the device supports native bfloat16 compute.
        supports_fp16: Whether the device supports native float16 compute.
        recommended_precision: Best available precision mode for this device.
        is_colab_gpu: Whether the device appears to be a Colab-provisioned GPU.
    """

    device: torch.device
    device_type: str
    device_name: str
    cuda_available: bool
    mps_available: bool
    num_gpus: int
    total_memory_gb: Optional[float]
    compute_capability: Optional[tuple[int, int]]
    supports_bf16: bool
    supports_fp16: bool
    recommended_precision: PrecisionMode
    is_colab_gpu: bool

    def summary(self) -> dict[str, object]:
        """Return a flat, loggable summary of this device snapshot."""
        return {
            "device": str(self.device),
            "device_type": self.device_type,
            "device_name": self.device_name,
            "num_gpus": self.num_gpus,
            "total_memory_gb": self.total_memory_gb,
            "compute_capability": self.compute_capability,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "recommended_precision": self.recommended_precision.value,
            "is_colab_gpu": self.is_colab_gpu,
        }


class DeviceManager:
    """Resolves and caches compute device and precision capabilities.

    Example:
        >>> manager = DeviceManager()
        >>> info = manager.detect()
        >>> info.device_type in {"cuda", "mps", "cpu"}
        True
    """

    def __init__(self, *, force_cpu: bool = False, gpu_index: int = 0) -> None:
        """Initialize the device manager.

        Args:
            force_cpu: If True, always resolve to CPU regardless of
                available accelerators. Useful for debugging and for
                deterministic CI environments.
            gpu_index: Index of the CUDA device to prefer when multiple
                GPUs are visible.
        """
        self._force_cpu = force_cpu
        self._gpu_index = gpu_index
        self._cached_info: Optional[DeviceInfo] = None

    def detect(self, *, refresh: bool = False) -> DeviceInfo:
        """Detect the best available compute device and its capabilities.

        Args:
            refresh: If True, bypass the cache and re-run detection.

        Returns:
            A :class:`DeviceInfo` describing the resolved device.
        """
        if self._cached_info is not None and not refresh:
            return self._cached_info

        if self._force_cpu:
            info = self._build_cpu_info()
            self._cached_info = info
            return info

        try:
            if torch.cuda.is_available():
                info = self._build_cuda_info()
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                info = self._build_mps_info()
            else:
                info = self._build_cpu_info()
        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.warning(
                "Device detection raised an unexpected error (%s); falling back to CPU.", exc
            )
            info = self._build_cpu_info()

        self._cached_info = info
        return info

    def _build_cuda_info(self) -> DeviceInfo:
        num_gpus = torch.cuda.device_count()
        index = self._gpu_index if self._gpu_index < num_gpus else 0
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(index)
        compute_capability = (props.major, props.minor)
        total_memory_gb = round(props.total_memory / (1024**3), 2)

        supports_bf16 = self._safe_bf16_check()
        supports_fp16 = compute_capability[0] >= 6  # Pascal (6.x) and newer support fp16 well
        recommended = (
            PrecisionMode.BF16
            if supports_bf16
            else (PrecisionMode.FP16 if supports_fp16 else PrecisionMode.FP32)
        )
        is_colab_gpu = self._detect_colab_gpu()

        return DeviceInfo(
            device=device,
            device_type="cuda",
            device_name=props.name,
            cuda_available=True,
            mps_available=False,
            num_gpus=num_gpus,
            total_memory_gb=total_memory_gb,
            compute_capability=compute_capability,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            recommended_precision=recommended,
            is_colab_gpu=is_colab_gpu,
        )

    def _build_mps_info(self) -> DeviceInfo:
        return DeviceInfo(
            device=torch.device("mps"),
            device_type="mps",
            device_name="Apple MPS",
            cuda_available=False,
            mps_available=True,
            num_gpus=0,
            total_memory_gb=None,
            compute_capability=None,
            supports_bf16=False,
            supports_fp16=True,
            recommended_precision=PrecisionMode.FP16,
            is_colab_gpu=False,
        )

    def _build_cpu_info(self) -> DeviceInfo:
        return DeviceInfo(
            device=torch.device("cpu"),
            device_type="cpu",
            device_name="CPU",
            cuda_available=False,
            mps_available=False,
            num_gpus=0,
            total_memory_gb=None,
            compute_capability=None,
            supports_bf16=False,
            supports_fp16=False,
            recommended_precision=PrecisionMode.FP32,
            is_colab_gpu=False,
        )

    @staticmethod
    def _safe_bf16_check() -> bool:
        """Safely check bfloat16 support, tolerating older torch builds."""
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:  # pragma: no cover - defensive
            return False

    @staticmethod
    def _detect_colab_gpu() -> bool:
        """Heuristically detect whether the current GPU is Colab-provisioned."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return False
            name = result.stdout.strip().lower()
            return any(tag in name for tag in ("t4", "a100", "v100", "l4", "p100"))
        except (OSError, subprocess.SubprocessError):
            return False

    def resolve_precision(
        self, requested: PrecisionMode, info: Optional[DeviceInfo] = None
    ) -> PrecisionMode:
        """Resolve a requested precision mode against actual hardware support.

        Args:
            requested: The precision mode requested by configuration.
            info: Optional pre-computed :class:`DeviceInfo`; detected fresh
                if omitted.

        Returns:
            The precision mode that should actually be used. ``AUTO``
            resolves to the device's recommended precision; unsupported
            explicit modes are downgraded with a warning.
        """
        device_info = info or self.detect()

        if requested == PrecisionMode.AUTO:
            return device_info.recommended_precision
        if requested == PrecisionMode.BF16 and not device_info.supports_bf16:
            _logger.warning(
                "bf16 requested but not supported on %s; falling back to %s.",
                device_info.device_name,
                device_info.recommended_precision.value,
            )
            return (
                PrecisionMode.FP16 if device_info.supports_fp16 else PrecisionMode.FP32
            )
        if requested == PrecisionMode.FP16 and not device_info.supports_fp16:
            _logger.warning(
                "fp16 requested but not supported on %s; falling back to fp32.",
                device_info.device_name,
            )
            return PrecisionMode.FP32
        return requested

    def autocast_dtype(self, precision: PrecisionMode) -> Optional[torch.dtype]:
        """Map a resolved :class:`PrecisionMode` to a ``torch.dtype``.

        Args:
            precision: A concrete (already-resolved) precision mode. Passing
                ``AUTO`` here is a programming error since it must first be
                resolved via :meth:`resolve_precision`.

        Returns:
            The corresponding ``torch.dtype`` for use with
            ``torch.autocast``, or ``None`` for FP32 (i.e. no autocast).

        Raises:
            ValueError: If ``precision`` is :attr:`PrecisionMode.AUTO`.
        """
        if precision == PrecisionMode.AUTO:
            raise ValueError("autocast_dtype requires a resolved precision, not AUTO.")
        mapping: dict[PrecisionMode, Optional[torch.dtype]] = {
            PrecisionMode.FP32: None,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
        }
        return mapping[precision]
