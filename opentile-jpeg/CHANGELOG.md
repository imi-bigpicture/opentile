# opentile-jpeg changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial version: lossless JPEG operations on the DCT coefficients, built on
  libjpeg-turbo.
  - `crop` and `crop_multiple`, optionally extending past the image edges with a
    `background` color.
  - `compose`, to re-tile sources placed on a grid into output tiles.
  - `optimize`, rebuilding with optimal Huffman tables, optionally progressive.
  - `fill`, a solid-color tile matching a template's size and encoding.
  - `header`, reading dimensions and sampling without decoding.
  - `decode_coefficients` and `encode_coefficients`, DCT coefficients as numpy
    arrays.
  - Metadata (ICC, EXIF, XMP, comments) kept by default, removable with `strip`.
  - A per-call `max_pixels` limit, by default `DEFAULT_MAX_PIXELS` (Pillow's),
    refusing larger images with `JpegTooLargeError` before allocating for them.
- Wheels with libjpeg-turbo included, for Linux (x86_64 and aarch64), Windows
  (x64 and ARM64) and macOS (Apple Silicon and Intel), Python 3.11 or later.
