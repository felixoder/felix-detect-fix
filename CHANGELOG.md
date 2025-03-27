# Changelog

All notable changes to this project felix-detect-fix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- v1.0.1 The initial Release
- v1.0.2 Made the extension available for testing purpose
- v1.0.3 Added Installation scripts for better and easy installation
- v1.0.4 Made fixes and also made this cross platform


### Changed

- Initial pre-release
- Added some fixes for tensorflow and cuda. It should check the user that cuda/GPU is there or not if he have it will use CUDA otherwise it will use the CPU
- Installed the huggingface model from bash script for installation
- Found problems after installing in different OS. So we will install the model after the first run. After the first run the model will be installed with ease.

### Removed
- Manual installation v1.0.4

## [1.0.0] - 2025-02-26

### Added

- Initialized the project. Added the model for testing

## [1.0.1] - 2025-02-27

### Added

- Designed the README.md.
- Added some deployment scripts.

### Fixed

- Manual Installation.

## [1.0.2] - 2017-03-17

### Added

- Added installation script for both linux and macos.
- Generalized the fine tune model to optimize the accuracy and other matrics.
- Added more labeled data to the dataset by scraping w3schools.

### Changed

- Start using "changelog" over "change log" since it's the common usage.
- Fix some minor problems.

### Removed

- Section about "changelog" vs "CHANGELOG".

## [1.0.3] - 2025-03-18

### Added

- Made it cross platform.

## [1.0.4] - 2025-03-27

### Changed

- Made the accuracy score good, F1, other matrices are nice for the model.
- Fixed the run_model.py and setup bash file
 
### Added

- Formatting of the code after rewriting on it


Made by Debayan Ghosh (@felixoder)
