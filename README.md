# Music Gen Framework — Phase 1: Core Infrastructure

An enterprise-grade, open-source scaffolding layer for AI music generation,
optimized for Google Colab. **Phase 1** delivers only the foundational
infrastructure — configuration, device management, logging, reproducibility,
validation, and audio I/O. No model architecture or generation logic is
included at this stage.

## Project structure

```
music-gen-framework/
├── config/
│   ├── __init__.py        # Public API re-exports
│   ├── schema.py           # Frozen dataclasses: AudioConfig, ModelConfig,
│   │                        # TrainingConfig, PathConfig, ColabConfig,
│   │                        # LoggingConfig, Config (root)
│   ├── settings.py          # load_settings() / get_settings(): layers
│   │                        # defaults -> YAML -> env vars -> overrides
│   └── example.yaml         # Example configuration file
├── utils/
│   ├── __init__.py          # Public API re-exports
│   ├── device.py            # DeviceManager: CUDA/MPS/CPU detection,
│   │                        # mixed-precision (bf16/fp16) capability checks
│   ├── logging.py           # setup_logging()/get_logger(): rich console +
│   │                        # rotating file handlers, optional JSON logs
│   ├── seed.py               # set_global_seed(), seed_worker() for
│   │                        # reproducible Python/NumPy/PyTorch RNGs
│   ├── validation.py         # Reusable validation helpers + ValidationError
│   └── audio_io.py           # load_audio()/save_audio()/resample_audio()/
│                             # normalize_audio() — I/O only, no modeling
├── data/                    # Datasets (gitignored, .gitkeep only)
├── checkpoints/             # Model checkpoints (gitignored, .gitkeep only)
├── logs/                    # Log files (gitignored, .gitkeep only)
├── outputs/                 # Generated artifacts (gitignored, .gitkeep only)
├── tests/                   # Test suite (empty in Phase 1)
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Design principles

- **Configuration as code**: every setting is a validated, frozen
  `dataclass`. Invalid values raise `ConfigValidationError` at construction
  time, not mid-training.
- **Layered settings resolution**: `config.settings.load_settings()` merges,
  in increasing priority, dataclass defaults → an optional YAML file →
  `MGF_<SECTION>__<FIELD>` environment variables → explicit keyword
  overrides.
- **Hardware-aware, not hardware-dependent**: `utils.device.DeviceManager`
  detects CUDA / Apple MPS / CPU, GPU compute capability, and native
  bf16/fp16 support, then recommends (and can resolve) a precision mode —
  with automatic, logged fallback to CPU/fp32 when accelerators or
  precision modes are unavailable.
- **Reproducibility by default**: `utils.seed.set_global_seed()` seeds
  Python, NumPy, and PyTorch (CPU + all visible CUDA devices) in one call,
  plus a `worker_init_fn` for deterministic `DataLoader` workers.
- **Fail fast, fail clearly**: `utils.validation` centralizes range/type/path
  checks with a single `ValidationError` type carrying the offending field
  name and value.

## Quick start (local)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, for an editable install with dev tools:
pip install -e ".[dev]"
```

```python
from config import get_settings
from utils import DeviceManager, set_global_seed, setup_logging

settings = get_settings(config_path="config/example.yaml")
setup_logging(
    level=settings.logging.level.value,
    log_dir=settings.paths.log_dir,
    log_to_file=settings.logging.log_to_file,
)
set_global_seed(settings.training.seed)

device_info = DeviceManager().detect()
print(device_info.summary())
```

## Running in Google Colab

1. **Open a new Colab notebook** and select a GPU runtime:
   `Runtime → Change runtime type → Hardware accelerator → GPU (T4/A100/etc.)`.

2. **Upload or clone the project** into the Colab filesystem:

   ```python
   # Option A: clone from your Git remote
   !git clone https://github.com/your-org/music-gen-framework.git
   %cd music-gen-framework

   # Option B: upload the provided zip and unpack it
   from google.colab import files
   uploaded = files.upload()  # select music-gen-framework.zip
   !unzip -q music-gen-framework.zip
   %cd music-gen-framework
   ```

3. **Install dependencies** (Colab already ships compatible `torch`/CUDA
   drivers, so this installs the remaining packages quickly):

   ```python
   !pip install -q -r requirements.txt
   ```

4. **(Optional) Mount Google Drive** for persistent checkpoints/datasets
   across sessions:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

   Then point `PathConfig` at Drive-backed directories, e.g. via
   `config/example.yaml`:

   ```yaml
   paths:
     checkpoint_dir: "/content/drive/MyDrive/music-gen-framework/checkpoints"
     data_dir: "/content/drive/MyDrive/music-gen-framework/data"
   colab:
     mount_drive: true
   ```

5. **Initialize the framework** in a notebook cell:

   ```python
   from config import get_settings
   from utils import DeviceManager, set_global_seed, setup_logging

   settings = get_settings(config_path="config/example.yaml")
   settings.paths.ensure_directories()

   setup_logging(
       level=settings.logging.level.value,
       log_dir=settings.paths.log_dir,
       log_to_file=settings.logging.log_to_file,
       use_rich=True,
   )

   set_global_seed(settings.training.seed)

   device_manager = DeviceManager()
   device_info = device_manager.detect()
   resolved_precision = device_manager.resolve_precision(settings.model.precision, device_info)

   print("Detected device:", device_info.summary())
   print("Resolved precision:", resolved_precision.value)
   ```

   On a T4 Colab runtime this will typically detect CUDA, report
   `supports_fp16=True` (and `supports_bf16=False` on T4; `True` on A100/L4),
   and resolve `PrecisionMode.AUTO` to the best available mode automatically.
   If no GPU is attached, it logs a clear fallback message and proceeds
   correctly on CPU with `fp32`.

6. **Verify audio I/O** with any sample WAV file uploaded to the runtime:

   ```python
   from utils import load_audio, save_audio

   waveform, metadata = load_audio(
       "sample.wav",
       target_sample_rate=settings.audio.sample_rate,
       normalize=settings.audio.normalize,
   )
   print(metadata.summary() if hasattr(metadata, "summary") else metadata)

   save_audio(waveform, "outputs/roundtrip.wav", sample_rate=metadata.sample_rate, overwrite=True)
   ```

## What's next (out of scope for Phase 1)

Model architectures, training loops, dataset pipelines, and generation/
inference code are intentionally **not** part of this phase and will be
introduced in subsequent phases on top of this infrastructure layer.
