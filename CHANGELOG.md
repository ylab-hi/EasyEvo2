# Changelog

## [0.3.0] - 2026-03-18

### Added
- **`vep` command**: Zero-shot variant effect prediction from VCF files using Evo2
  log-likelihoods, following the methodology from the Evo2 Nature paper (Brixi et al., 2026).
  Supports SNVs, insertions, deletions, and multi-allelic records.
- **`generate` command**: DNA sequence generation with support for literal prompts,
  FASTA file prompts, and species-tagged phylogenetic prompting via GBIF.
- **`slurm` command**: CLI wrapper for generating SLURM job scripts for HPC submission.
- New model support: `evo2_20b`, `evo2_7b_262k`, `evo2_7b_microviridae` from
  upstream evo2 v0.5.0.
- Docker multi-target builds: `full` (NGC-based, all models) and `light` (7B only,
  any CUDA GPU).
- `docker-compose.yml` for GPU orchestration with HuggingFace cache volumes.
- `.dockerignore` for efficient builds.
- Comprehensive test suite covering model, IO, utils, dataloader, SLURM, VEP,
  generation, and CLI integration.
- `CITATION.cff` for machine-readable citation metadata.
- `CONTRIBUTING.md` with development guidelines.
- PyPI classifiers, keywords, and author metadata for discoverability.

### Changed
- `ModelType` migrated from `str, Enum` to `StrEnum` (Python 3.11+).
- `load_model()` simplified — removed unreachable else branch.
- All docstrings standardized to NumPy style (replaced `Args:` format).
- Added full type annotations to all public functions.
- Replaced all `print()` calls with proper `log.info()`/`log.error()`/`typer.echo()`.
- `load_tensor()` now defaults to CPU device (safer default, prevents unintended
  CUDA initialization).
- Dockerfile rewritten to use NVIDIA NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.04-py3`).
- `rich` dependency unpinned from `==13.9.2` to `>=13.9.2`.
- `pandas-stubs` moved from runtime to dev dependency.
- `pysam` added as runtime dependency for VCF parsing.

### Fixed
- SLURM gres format: changed comma separator to colon (`gpu:a100:1` instead of `gpu:a100,1`).
- `test_evo2.py`: fixed `.items()` call on generator object.
- `test_embed.py`: fixed CUDA initialization failure on CPU-only test runs.
- `test_probs.py`: added actual assertions (was only printing).
- `exon_classifier.py`: replaced debug `print()` with `log.debug()`.
- `dataloader.py`: fixed unused loop variable lint error (B007).
- `dataloader.py`: corrected docstring (said "Yields" but function returns dict).
- `embed.py`: moved `import gc` to module top level.
- `__init__.py`: added missing `exon_classifier` and `transform` modules to exports.

## [0.1.8] - 2024-03-21

### Added

- New `score` command (renamed from `calculate_probs`) with improved functionality
- Added pandas support for more efficient data processing
- Added metadata tracking for probability calculations including:
  - Model type
  - Window size
  - Step size
  - Device used
  - Timestamp
- Added support for custom output file paths

### Changed

- Improved error handling with proper exception chaining
- Enhanced file I/O operations using pathlib
- More efficient data processing using pandas DataFrame
- Better output format with separate metadata JSON file

### Fixed

- Improved file path handling using pathlib consistently
- Better error messages with proper exception context
