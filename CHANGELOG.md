# opentile changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] -

## [0.13.1] - 2024-07-01

### Fixed

- Fix for not closing file handle when `OpenTile.open()` is not used as a context manager.

## [0.13.0] - 2024-07-01

### Changed

- More efficient `OpenTile.open()` by reusing the `TiffFile` instance.

### Removed

- `OpenTile.get_tiler()` method removed. Use `OpenTile.detect_format()` to get the format of a file or `OpenTile.open()` to get an opened instance of a tiler instead.

### Fixed

- Missing to close file handle when using `OpenTile.open()` or `OpenTile.detect_format()`.

## [0.12.0] - 2024-02-20

### Added

- Support for opening files using fsspec.

## [0.11.2] - 2024-02-20

### Fixed

- Updated `ome-types` to 0.5.0.

## [0.11.1] - 2023-12-06

### Fixed

- Missing jpeg headers when using TurboJpeg 3.
- Check for either `libturbojpeg.dll` or `turbojpeg.dll`.

## [0.11.0] - 2023-12-06

### Changed

- Return iterators of tiles instead of lists.
- Encode Jpeg and Jpeg 2000 using imagecodecs.

## [0.10.4] - 2023-11-30

### Fixed

- Order of pixel spacing for philips tiff files.

## [0.10.3] - 2023-09-01

### Fixed

- Change version requirement for tifffile and imagecodecs to allow newer versions.

## [0.10.2] - 2023-08-29

### Fixed

- Bumped version of ome-types to support pydantic 2.0.

## [0.10.1] - 2023-07-07

### Fixed

- Bumped version of ome-types to pin pydantic to 1.x.

## [0.10.0] - 2023-06-26

### Changed

- Relaxed python requirement to >= 3.8.

## [0.9.0] - 2023-04-03

### Added

- Basic support for OME tiff files.

### Changed

- Refactored code, renamed OpenTilePage to TiffImage (and reflecting the change to subclasses). Removed abstract method get_image() and added abstract methods get_level(), get_label(), and get_overview().

## [0.8.1] - 2023-03-31

### Added

- Added conda installation instructions to readme.

### Changed

- Implementation and use of Region updated to that of wsidicom.

## [0.8.0] - 2023-03-21

### Added

- Added optional image_offset property in metadata class.

### Fixed

- Fixed error in readme.

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

- tifffile minimum version set to 2022.5.4.
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

[Unreleased]: https://github.com/imi-bigpicture/opentile/compare/v0.13.0..HEAD
[0.13.0]: https://github.com/imi-bigpicture/opentile/compare/v0.12.0..v0.13.0
[0.12.0]: https://github.com/imi-bigpicture/opentile/compare/v0.11.2..v0.12.0
[0.11.2]: https://github.com/imi-bigpicture/opentile/compare/v0.11.1..v0.11.2
[0.11.1]: https://github.com/imi-bigpicture/opentile/compare/v0.11.0..v0.11.1
[0.11.0]: https://github.com/imi-bigpicture/opentile/compare/v0.10.4..v0.11.0
[0.10.4]: https://github.com/imi-bigpicture/opentile/compare/v0.10.3..v0.10.4
[0.10.3]: https://github.com/imi-bigpicture/opentile/compare/v0.10.2..v0.10.3
[0.10.2]: https://github.com/imi-bigpicture/opentile/compare/v0.10.1..v0.10.2
[0.10.1]: https://github.com/imi-bigpicture/opentile/compare/v0.10.0..v0.10.1
[0.10.0]: https://github.com/imi-bigpicture/opentile/compare/v0.9.0..v0.10.0
[0.9.0]: https://github.com/imi-bigpicture/opentile/compare/v0.8.1..v0.9.0
[0.8.1]: https://github.com/imi-bigpicture/opentile/compare/v0.8.0..v0.8.1
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
