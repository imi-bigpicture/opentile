# wsidicom changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] -

## [0.8.0] - 2023-03-21

### Added

- Added optional image_offset property in metadata class.

## [0.7.1] - 2023-03-15

### Fixed

- Use MCU read from frame instead of from subsampling tag as the latter can be incorrect.

## [0.7.0] - 2023-02-13

### Changed

- Crop out label and overview page from macro page of ndpi-files.

## [0.6.0] - 2023-01-24

### Changed

- Added Python 3.11 as supported version.

## [0.5.0] - 2022-12-13

### Changed

- get_tiles() changed from list comprehension of get_tile to sequentially read all tiles and then process them. This allows all tiles to be read with the same file lock improving threading performance.

## [0.4.2] - 2022-11-24

### Fixed

- Use PHOTOMETRIC-enum from tifffile for photometric interpretation.
- Update imagecodecs to 2022.9.26.

## [0.4.1] - 2022-11-24

### Fixed

- Added missing tag in Philips tiff tiler.

## [0.4.0] - 2022-11-24

### Added

- Support for 3DHISTECH tiff files.
- Tiler-property `icc_profile` returning icc profile if found in file.

### Changed

- tifffile minimun version set to 2022.5.4.
- Replaced Dict property `properties` of tilers with property `metadata` of Metadata-class.

### Fixed

- Use `COMPRESSION`-enum from TiffFile for compression type checks.
- Decompress jpeg data to RGB instead of BGR when using turbojpeg.

## [0.3.0] - 2022-04-20

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

## [0.2.0] - 2022-02-14

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

[Unreleased]: https://github.com/imi-bigpicture/opentile/compare/v0.8.0..HEAD
[0.8.0]: https://github.com/imi-bigpicture/opentile/compare/v0.7.1..v0.8.0
[0.7.1]: https://github.com/imi-bigpicture/opentile/compare/v0.7.0..v0.7.1
[0.7.0]: https://github.com/imi-bigpicture/opentile/compare/v0.6.0..v0.7.0
[0.6.0]: https://github.com/imi-bigpicture/opentile/compare/v0.5.0..v0.6.0
[0.5.0]: https://github.com/imi-bigpicture/opentile/compare/v0.4.2..v0.5.0
[0.4.2]: https://github.com/imi-bigpicture/opentile/compare/v0.4.1..v0.4.2
[0.4.1]: https://github.com/imi-bigpicture/opentile/compare/v0.4.0..v0.4.1
[0.4.0]: https://github.com/imi-bigpicture/opentile/compare/v0.3.0..v0.4.0
[0.3.0]: https://github.com/imi-bigpicture/opentile/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/imi-bigpicture/opentile/compare/v0.1.1..v0.2.0
[0.1.1]: https://github.com/imi-bigpicture/opentile/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/imi-bigpicture/opentile/tree/v0.1.0
