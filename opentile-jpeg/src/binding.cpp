/*
 *  Copyright 2026 SECTRA AB
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "engine.h"

namespace nb = nanobind;
using namespace nb::literals;

using Pair = std::pair<int, int>;
using Rgb = std::tuple<int, int, int>;

static jc_color to_color(const Rgb& rgb, bool color_transform) {
    return {std::get<0>(rgb), std::get<1>(rgb), std::get<2>(rgb), color_transform};
}

/* An optional background color for crops; nullopt keeps crops inside. */
static std::optional<jc_color> to_background(const std::optional<Rgb>& rgb,
                                             bool color_transform) {
    if (!rgb) return std::nullopt;
    return to_color(*rgb, color_transform);
}

static const jc_color* pointer(const std::optional<jc_color>& color) {
    return color ? &*color : nullptr;
}

using Coefficients =
    nb::ndarray<nb::numpy, int16_t, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
using CoefficientsIn =
    nb::ndarray<const int16_t, nb::ndim<4>, nb::c_contig, nb::device::cpu>;

struct JpegError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/* Pointer and length of a bytes object, taken while holding the GIL. */
struct BytesView {
    const unsigned char* data;
    unsigned long size;
    explicit BytesView(const nb::bytes& b)
        : data(reinterpret_cast<const unsigned char*>(b.c_str())),
          size(static_cast<unsigned long>(b.size())) {
        /* libjpeg sizes are unsigned long: 32 bits on Windows. */
        if (b.size() > std::numeric_limits<unsigned long>::max())
            throw std::invalid_argument("JPEG data is too large for libjpeg-turbo");
    }
};

static nb::object errors_class(const char* name) {
    return nb::module_::import_("opentile_jpeg.errors").attr(name);
}

/* Re-issue libjpeg-turbo's warnings as one JpegWarning, attributed to
   the code that called into opentile_jpeg. */
static void warn(const jc_messages& messages) {
    if (messages.warning_count == 0) return;
    std::string text = messages.first_warning;
    if (messages.warning_count > 1)
        text += " (and " + std::to_string(messages.warning_count - 1) +
                " more warnings)";
    if (PyErr_WarnEx(errors_class("JpegWarning").ptr(), text.c_str(), 2) < 0)
        throw nb::python_error(); /* warnings are configured as errors */
}

/* Run `engine_call(messages)` with the GIL released, then raise its error or
   issue its warnings. */
template <typename F>
static void call(F&& engine_call) {
    jc_messages messages = {};
    int status;
    {
        nb::gil_scoped_release release;
        status = engine_call(&messages);
    }
    switch (status) {
    case JC_OK: return warn(messages);
    case JC_ERROR_ARGUMENT: throw std::invalid_argument(messages.error);
    case JC_ERROR_MEMORY: throw std::bad_alloc();
    default: throw JpegError(messages.error);
    }
}

/* Run an engine call that encodes a JPEG, returning the JPEG as bytes. */
template <typename F>
static nb::bytes encode(F&& engine_call) {
    unsigned char* data = nullptr;
    unsigned long size = 0;
    call([&](auto* messages) { return engine_call(&data, &size, messages); });
    std::unique_ptr<unsigned char, decltype(&jc_free)> owner(data, jc_free);
    return nb::bytes(reinterpret_cast<const char*>(data), size);
}

static jc_layout read_layout(const BytesView& jpeg) {
    jc_layout layout;
    call([&](auto* messages) {
        return jc_read_layout(jpeg.data, jpeg.size, &layout, messages);
    });
    return layout;
}

static nb::tuple to_python(const jc_layout& layout) {
    nb::list components;
    for (int i = 0; i < layout.num_components; i++) {
        const jc_component_layout& c = layout.components[i];
        components.append(
            nb::make_tuple(nb::make_tuple(c.h_samp_factor, c.v_samp_factor),
                           nb::make_tuple(c.height_in_blocks, c.width_in_blocks)));
    }
    return nb::make_tuple(nb::make_tuple(layout.width, layout.height), components);
}

static std::string shape_string(size_t a, size_t b, size_t c, size_t d) {
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " +
           std::to_string(c) + ", " + std::to_string(d) + ")";
}

static nb::bytes encode_coefficients(const nb::bytes& template_jpeg,
                                     const std::vector<CoefficientsIn>& arrays,
                                     bool optimize, int strip) {
    BytesView tmpl(template_jpeg);
    jc_layout layout = read_layout(tmpl);
    if (arrays.size() != static_cast<size_t>(layout.num_components))
        throw std::invalid_argument(
            "expected " + std::to_string(layout.num_components) +
            " coefficient arrays, got " + std::to_string(arrays.size()));

    std::vector<const short*> components;
    for (size_t i = 0; i < arrays.size(); i++) {
        const CoefficientsIn& array = arrays[i];
        size_t rows = layout.components[i].height_in_blocks;
        size_t cols = layout.components[i].width_in_blocks;
        if (array.shape(0) != rows || array.shape(1) != cols ||
            array.shape(2) != 8 || array.shape(3) != 8)
            throw std::invalid_argument(
                "component " + std::to_string(i) + " coefficient array has shape " +
                shape_string(array.shape(0), array.shape(1), array.shape(2),
                             array.shape(3)) +
                ", expected " + shape_string(rows, cols, 8, 8));
        components.push_back(reinterpret_cast<const short*>(array.data()));
    }
    return encode([&](auto* out, auto* size, auto* messages) {
        return jc_encode_coefficients(tmpl.data, tmpl.size, components.data(),
                                      optimize, strip, out, size, messages);
    });
}

class Source {
  public:
    explicit Source(const nb::bytes& jpeg) {
        BytesView view(jpeg);
        call([&](auto* messages) {
            return jc_source_open(view.data, view.size, &handle_, messages);
        });
    }
    ~Source() { jc_source_close(handle_); }
    Source(const Source&) = delete;
    Source& operator=(const Source&) = delete;

    jc_source* handle() const { return handle_; }

    jc_layout layout() const {
        jc_layout layout;
        jc_source_layout(handle_, &layout);
        return layout;
    }

    nb::bytes crop(int x, int y, int width, int height,
                   const std::optional<Rgb>& background, bool color_transform,
                   int strip) {
        std::optional<jc_color> color = to_background(background, color_transform);
        return encode([&](auto* out, auto* size, auto* messages) {
            return jc_source_crop(handle_, x, y, width, height, pointer(color), strip,
                                  out, size, messages);
        });
    }

    Coefficients read_component(int component) {
        jc_layout layout = this->layout();
        if (component < 0 || component >= layout.num_components)
            throw std::out_of_range("component index out of range");
        size_t rows = layout.components[component].height_in_blocks;
        size_t cols = layout.components[component].width_in_blocks;
        auto* data = new int16_t[rows * cols * 64];
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<int16_t*>(p);
        });
        call([&](auto* messages) {
            return jc_source_read_component(handle_, component, data, messages);
        });
        return Coefficients(data, {rows, cols, 8, 8}, owner);
    }

  private:
    jc_source* handle_ = nullptr;
};

class Cropper {
  public:
    Cropper(const Source& source, int width, int height,
            const std::optional<Rgb>& background, bool color_transform, int strip) {
        std::optional<jc_color> color = to_background(background, color_transform);
        call([&](auto* messages) {
            return jc_cropper_open(source.handle(), width, height, pointer(color),
                                   strip, &handle_, messages);
        });
    }
    ~Cropper() { jc_cropper_close(handle_); }
    Cropper(const Cropper&) = delete;
    Cropper& operator=(const Cropper&) = delete;

    nb::bytes crop(int x, int y) {
        return encode([&](auto* out, auto* size, auto* messages) {
            return jc_cropper_crop(handle_, x, y, out, size, messages);
        });
    }

  private:
    jc_cropper* handle_ = nullptr;
};

class Composer {
  public:
    Composer(const std::vector<Source*>& sources, const std::vector<Pair>& cells,
             Pair tile_size, Pair content_size, Pair source_size,
             const std::optional<Rgb>& background, bool color_transform,
             int strip) {
        if (sources.size() != cells.size())
            throw std::invalid_argument("expected one grid cell per source");
        std::vector<jc_source*> handles;
        std::vector<jc_point> points;
        for (size_t i = 0; i < sources.size(); i++) {
            if (!sources[i]) throw std::invalid_argument("sources must not contain None");
            handles.push_back(sources[i]->handle());
            points.push_back({cells[i].first, cells[i].second});
        }
        std::optional<jc_color> color = to_background(background, color_transform);
        call([&](auto* messages) {
            return jc_composer_open(
                handles.data(), points.data(), static_cast<int>(handles.size()),
                {tile_size.first, tile_size.second},
                {content_size.first, content_size.second},
                {source_size.first, source_size.second}, pointer(color), strip,
                &handle_, messages);
        });
    }
    ~Composer() { jc_composer_close(handle_); }
    Composer(const Composer&) = delete;
    Composer& operator=(const Composer&) = delete;

    nb::bytes tile(int x, int y) {
        return encode([&](auto* out, auto* size, auto* messages) {
            return jc_composer_tile(handle_, x, y, out, size, messages);
        });
    }

  private:
    jc_composer* handle_ = nullptr;
};

NB_MODULE(_binding, m) {
    /* Raise the exception class defined in opentile_jpeg.errors. */
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const JpegError& e) {
            PyErr_SetString(errors_class("JpegError").ptr(), e.what());
        }
    });

    m.attr("LIBJPEG_TURBO_VERSION") = jc_libjpeg_turbo_version();
    m.attr("STRIP_ICC") = int(JC_STRIP_ICC);
    m.attr("STRIP_EXIF") = int(JC_STRIP_EXIF);
    m.attr("STRIP_XMP") = int(JC_STRIP_XMP);
    m.attr("STRIP_COMMENTS") = int(JC_STRIP_COMMENTS);
    m.attr("STRIP_OTHER") = int(JC_STRIP_OTHER);

    m.def(
        "read_layout",
        [](const nb::bytes& jpeg) { return to_python(read_layout(BytesView(jpeg))); },
        "jpeg"_a,
        "``((width, height), [((h_samp, v_samp), (height_in_blocks, "
        "width_in_blocks)), ...])`` from the frame header.");

    m.def(
        "optimize",
        [](const nb::bytes& jpeg, bool progressive, int strip) {
            BytesView view(jpeg);
            return encode([&](auto* out, auto* size, auto* messages) {
                return jc_optimize(view.data, view.size, progressive, strip, out, size,
                                   messages);
            });
        },
        "jpeg"_a, "progressive"_a = false, "strip"_a = 0, "See :func:`opentile_jpeg.optimize`.");

    m.def(
        "fill",
        [](const nb::bytes& template_jpeg, const Rgb& color, bool color_transform,
           int strip) {
            BytesView view(template_jpeg);
            return encode([&](auto* out, auto* size, auto* messages) {
                return jc_fill(view.data, view.size, to_color(color, color_transform),
                               strip, out, size, messages);
            });
        },
        "template"_a, "color"_a, "color_transform"_a = true, "strip"_a = 0,
        "See :func:`opentile_jpeg.fill`.");

    m.def("encode_coefficients", &encode_coefficients, "template"_a,
          "coefficients"_a, "optimize"_a = false, "strip"_a = 0,
          "See :func:`opentile_jpeg.encode_coefficients`.");

    nb::class_<Source>(m, "Source",
                       "A JPEG with its DCT coefficients read once, shared by "
                       "crops, Croppers and Composers.")
        .def(nb::init<const nb::bytes&>(), "jpeg"_a)
        .def("layout",
             [](const Source& source) { return to_python(source.layout()); },
             "Like :func:`read_layout`.")
        .def("crop", &Source::crop, "x"_a, "y"_a, "width"_a, "height"_a,
             "background"_a = nb::none(), "color_transform"_a = true, "strip"_a = 0,
             "Losslessly crop one MCU-aligned region as a new JPEG; see "
             ":func:`opentile_jpeg.crop`.")
        .def("read_component", &Source::read_component, "component"_a,
             "The component's coefficients as a new int16 array of shape "
             "``(height_in_blocks, width_in_blocks, 8, 8)``.");

    nb::class_<Cropper>(m, "Cropper",
                        "Repeated same-size crops from one Source, reusing a "
                        "single compressor.")
        .def(nb::init<const Source&, int, int, const std::optional<Rgb>&, bool, int>(),
             "source"_a, "width"_a, "height"_a, "background"_a = nb::none(),
             "color_transform"_a = true, "strip"_a = 0, nb::keep_alive<1, 2>())
        .def("crop", &Cropper::crop, "x"_a, "y"_a,
             "Losslessly crop the MCU-aligned region at ``(x, y)`` as a new JPEG.");

    nb::class_<Composer>(m, "Composer",
                         "Re-tiles Sources placed on a grid into output tiles; "
                         "see :func:`opentile_jpeg.compose`.")
        .def(nb::init<const std::vector<Source*>&, const std::vector<Pair>&, Pair,
                      Pair, Pair, const std::optional<Rgb>&, bool, int>(),
             "sources"_a, "cells"_a, "tile_size"_a, "content_size"_a,
             "source_size"_a, "background"_a = nb::none(),
             "color_transform"_a = true, "strip"_a = 0, nb::keep_alive<1, 2>())
        .def("tile", &Composer::tile, "x"_a, "y"_a,
             "The output tile whose top-left is at plane pixel ``(x, y)``.");
}
