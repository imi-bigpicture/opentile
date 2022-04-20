# wsidicom changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...

## [0.3.0] - 2021-04-20
### Added
- Scripts and github actions for downloading test data.
- Properties photometric_interpretation, subsampling, and samples_per_pixel for OpenTilePage.
- __enter__ and __exit__ for Tiler.
- __version__ added.

### Changed
- Dropped support for python 3.7.

### Fixed
- Generation of blank tiles for philips tiff.
- Border appearing when cropping one-frammed ndpi pages.
- Even more descriptive error when jpeg crop fails.

## [0.2.0] - 2021-02-14
### Added
- Support for svs overview and label images.

### Changed
- More descriptive error when jpeg crop fails.

### Fixed
- Jpeg tables are not duplicated.
- Return j2k and not jp2 when encoding jpeg2000.

## [0.1.1] - 2021-12-02
### Changed
- Fix calculations of pixel spacing and mpp.

## [0.1.0] - 2021-11-30
### Added
- Initial release of opentile.

[Unreleased]: https://github.com/imi-bigpicture/opentile/compare/v0.3.0..HEAD
[0.3.0]: https://github.com/imi-bigpicture/opentile/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/imi-bigpicture/opentile/compare/v0.1.1..v0.2.0
[0.1.1]: https://github.com/imi-bigpicture/opentile/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/imi-bigpicture/opentile/tree/v0.1.0
