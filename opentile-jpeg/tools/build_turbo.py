#    Copyright 2026 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Fetch and build the pinned, sha256-verified libjpeg-turbo as a static lib.

The version and tarball hash are pinned in pyproject.toml
(``[tool.opentile-jpeg.libjpeg-turbo]``). Invoked by CMakeLists.txt at configure
time when the cached prefix is missing (the result is cached under
``.turbo/<version>-<build hash>`` so repeated builds don't refetch). Set
``OPENTILE_JPEG_TURBO_PREFIX`` to an existing install prefix to skip the fetch
entirely — the escape hatch for offline/hermetic builds and distro packaging.
Needs a C compiler; on x86, nasm on PATH enables SIMD (ARM SIMD needs no nasm).
Without it libjpeg-turbo quietly builds without SIMD, unless
``OPENTILE_JPEG_REQUIRE_SIMD=1`` makes that an error (as the wheel builds do).

Build logs go to stderr: stdout carries only the install prefix, for CMake.
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

with open(_REPO_ROOT / "pyproject.toml", "rb") as pyproject:
    _PIN = tomllib.load(pyproject)["tool"]["opentile-jpeg"]["libjpeg-turbo"]
TURBO_VERSION: str = _PIN["version"]
TURBO_SHA256: str = _PIN["sha256"]
TURBO_URL = (
    "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/"
    f"{TURBO_VERSION}.tar.gz"
)

_REQUIRE_SIMD = os.environ.get("OPENTILE_JPEG_REQUIRE_SIMD") == "1"

TURBO_OPTIONS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DENABLE_SHARED=0",
    "-DENABLE_STATIC=1",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=1",
    # MSVC: link the same C run-time DLL as the extension, so buffers malloc'd
    # inside libjpeg-turbo can be freed by opentile_jpeg.
    "-DWITH_CRT_DLL=1",
    # Only the static libjpeg is linked: skip TurboJPEG, the tools and the tests.
    "-DWITH_TURBOJPEG=0",
    "-DWITH_TOOLS=0",
    "-DWITH_TESTS=0",
    *(["-DREQUIRE_SIMD=1"] if _REQUIRE_SIMD else []),
)
# A cached build is reused only by builds for the same options and target: the
# project folder (with .turbo) is copied into cibuildwheel's Linux containers,
# whose glibc or musl differs from the host's, and a Mac can build for either
# architecture.
_BUILD_IDENTITY = (
    *TURBO_OPTIONS,
    sys.platform,
    platform.machine(),
    *platform.libc_ver(),
    os.environ.get("MACOSX_DEPLOYMENT_TARGET", ""),
    os.environ.get("CC", ""),
)
_CACHE_KEY = hashlib.sha256(" ".join(_BUILD_IDENTITY).encode()).hexdigest()[:8]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _run(*args: str | Path, cwd: Path | None = None) -> None:
    _log("+ " + " ".join(str(a) for a in args))
    subprocess.run(  # noqa: S603
        [str(a) for a in args], cwd=cwd, check=True, stdout=sys.stderr
    )


def _build_turbo(install: Path, cmake: str) -> None:
    work = install.parent / f"src-{TURBO_VERSION}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    _log(f"fetching {TURBO_URL}")
    with urllib.request.urlopen(TURBO_URL) as r:  # noqa: S310 (pinned github release)
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != TURBO_SHA256:
        raise RuntimeError(
            f"libjpeg-turbo {TURBO_VERSION} sha256 mismatch: "
            f"expected {TURBO_SHA256}, got {digest}"
        )
    tarball = work / "src.tar.gz"
    tarball.write_bytes(blob)
    with tarfile.open(tarball) as tf:
        tf.extractall(work)  # noqa: S202 (trusted, hash-verified release)
    src = next(work.glob("libjpeg-turbo-*"))

    build = work / "build"
    build.mkdir()
    _run(
        cmake,
        "-S",
        src,
        "-B",
        build,
        *TURBO_OPTIONS,
        f"-DCMAKE_INSTALL_PREFIX={install}",
    )
    _run(cmake, "--build", build, "--config", "Release", "--target", "install")
    _log(f"built libjpeg-turbo {TURBO_VERSION} -> {install}")


def turbo_prefix(cmake: str) -> Path:
    override = os.environ.get("OPENTILE_JPEG_TURBO_PREFIX")
    if override:
        return Path(override)
    install = _REPO_ROOT / ".turbo" / f"{TURBO_VERSION}-{_CACHE_KEY}"
    if not (install / "include" / "jpeglib.h").exists():
        _build_turbo(install, cmake)
    return install


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake", help="cmake executable to use")
    args = parser.parse_args()
    print(turbo_prefix(args.cmake))
