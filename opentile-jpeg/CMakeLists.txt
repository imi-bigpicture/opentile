# nanobind extension build: statically links the pinned libjpeg-turbo (built on
# demand by tools/build_turbo.py, cached under .turbo/<version>-<build hash>).
cmake_minimum_required(VERSION 3.18)
project(opentile_jpeg LANGUAGES C CXX)

# Development.SABIModule is required for STABLE_ABI to actually take effect —
# without it nanobind silently falls back to a per-version (full API) build.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module
                                    Development.SABIModule REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Ensure the pinned libjpeg-turbo static lib exists (fetch+build if missing;
# OPENTILE_JPEG_TURBO_PREFIX overrides — see tools/build_turbo.py).
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_turbo.py"
          --cmake "${CMAKE_COMMAND}"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TURBO_PREFIX
  RESULT_VARIABLE _turbo_rc)
if(_turbo_rc)
  message(FATAL_ERROR "tools/build_turbo.py failed (rc=${_turbo_rc})")
endif()
# The script logs to stderr and prints the prefix last; keep only that line.
string(REGEX REPLACE ".*\n" "" TURBO_PREFIX "${TURBO_PREFIX}")
find_library(TURBO_STATIC NAMES jpeg-static jpeg
  PATHS "${TURBO_PREFIX}/lib" "${TURBO_PREFIX}/lib64" NO_DEFAULT_PATH)
if(NOT TURBO_STATIC)
  message(FATAL_ERROR "no libjpeg-turbo static lib under ${TURBO_PREFIX}")
endif()

# STABLE_ABI: abi3 wheel when building on Python >= 3.12, per-version below.
nanobind_add_module(_binding STABLE_ABI
  src/binding.cpp
  src/engine.c)
target_include_directories(_binding PRIVATE "${TURBO_PREFIX}/include")
target_link_libraries(_binding PRIVATE "${TURBO_STATIC}")
# ELF exports a static library's symbols and binds calls to them at load time,
# so a libjpeg already loaded globally (the host program's, an RTLD_GLOBAL or
# LD_PRELOAD one) would take over our jpeg_* calls. Keep them local instead.
# (Mach-O binds to the defining library and Windows exports nothing unasked.)
if(UNIX AND NOT APPLE)
  target_link_options(_binding PRIVATE "LINKER:--exclude-libs,ALL")
endif()

nanobind_add_stub(_binding_stub
  MODULE _binding
  OUTPUT _binding.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_binding>
  DEPENDS _binding)

# Mirror the stub into the source tree so type checkers resolve
# opentile_jpeg._binding without a build product on the import path (the editable
# install keeps the extension out of src/). Committed, build-refreshed.
add_custom_target(_binding_stub_src ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/_binding.pyi"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/opentile_jpeg/_binding.pyi"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/_binding.pyi")

# The stub reaches wheels via its src/ mirror (wheel.packages); installing it
# here too would collide with that copy.
install(TARGETS _binding LIBRARY DESTINATION opentile_jpeg)
