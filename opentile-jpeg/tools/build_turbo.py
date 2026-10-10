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
``.turbo/<version>-<options hash>`` so repeated builds don't refetch). Set
``OPENTILE_JPEG_TURBO_PREFIX`` to an existing install prefix to skip the fetch
entirely — the escape hatch for offline/hermetic builds and distro packaging.
Needs a C compiler; on x86, nasm on PATH enables SIMD (ARM SIMD needs no nasm).
"""

import argparse
import hashlib
import os
import shutil
import subprocess
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

TURBO_OPTIONS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DENABLE_SHARED=0",
    "-DENABLE_STATIC=1",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=1",
    # MSVC: link the same C run-time DLL as the extension, so buffers malloc'd
    # inside libjpeg-turbo can be freed by opentile_jpeg.
    "-DWITH_CRT_DLL=1",
    # Without nasm, libjpeg-turbo builds without SIMD on x86 instead of failing.
    "-DWITH_SIMD=1",
)
# Changing the options must not reuse a cached build made with other options.
_CACHE_KEY = hashlib.sha256(" ".join(TURBO_OPTIONS).encode()).hexdigest()[:8]


def _run(*args: str | Path, cwd: Path | None = None) -> None:
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)  # noqa: S603


def _build_turbo(install: Path, cmake: str) -> None:
    work = install.parent / f"src-{TURBO_VERSION}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print(f"fetching {TURBO_URL}", flush=True)
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
    print(f"built libjpeg-turbo {TURBO_VERSION} -> {install}")


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
