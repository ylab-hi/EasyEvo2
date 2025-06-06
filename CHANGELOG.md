# Changelog

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
