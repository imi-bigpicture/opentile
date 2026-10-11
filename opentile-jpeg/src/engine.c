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

#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h> /* jpeglib.h uses FILE without including stdio.h */
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

#include "engine.h"

#if JC_MAX_COMPONENTS != MAX_COMPONENTS || JC_MESSAGE_LEN < JMSG_LENGTH_MAX
#error "engine.h limits disagree with jpeglib.h"
#endif

/* ---- errors and warnings ---------------------------------------------- */

/* libjpeg reports errors through error_exit, which must not return, so it
   jumps back to the setjmp of the engine call in progress. Every call points
   all libjpeg objects it touches at its own jc_error, including those held by
   long-lived sources, croppers and composers. */
typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
    jc_messages* messages;
} jc_error;

static void jump_to_caller(j_common_ptr cinfo) {
    jc_error* error = (jc_error*)cinfo->err;
    (*cinfo->err->format_message)(cinfo, error->messages->error);
    longjmp(error->jump, 1);
}

/* Collect warnings (negative levels) for the caller instead of printing them;
   non-negative levels are trace output, which is ignored. */
static void collect_warning(j_common_ptr cinfo, int msg_level) {
    jc_error* error = (jc_error*)cinfo->err;
    if (msg_level >= 0) return;
    cinfo->err->num_warnings++;
    if (error->messages->warning_count++ == 0)
        (*cinfo->err->format_message)(cinfo, error->messages->first_warning);
}

static struct jpeg_error_mgr* init_error(jc_error* error, jc_messages* messages) {
    jpeg_std_error(&error->pub);
    error->pub.error_exit = jump_to_caller;
    error->pub.emit_message = collect_warning;
    error->messages = messages;
    return &error->pub;
}

static int fail(int status, jc_messages* messages, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(messages->error, JC_MESSAGE_LEN, format, args);
    va_end(args);
    return status;
}

/* ---- helpers ---------------------------------------------------------- */

static int min_int(int a, int b) { return a < b ? a : b; }
static int max_int(int a, int b) { return a > b ? a : b; }
/* a / b for b > 0, rounded down, up, or to nearest with ties away from zero. */
static long long floor_div(long long a, long long b) { return a / b - (a % b < 0); }
static long long ceil_div(long long a, long long b) { return a / b + (a % b > 0); }
static long long round_div(long long a, long long b) {
    return a < 0 ? -floor_div(b - 2 * a, 2 * b) : floor_div(2 * a + b, 2 * b);
}
static int round_up(int a, int b) { return (int)ceil_div(a, b) * b; }

static JBLOCKROW block_row(j_common_ptr owner, jvirt_barray_ptr array, int row,
                           boolean writable) {
    return (*owner->mem->access_virt_barray)(owner, array, (JDIMENSION)row, 1,
                                             writable)[0];
}

/* Also keeps every APPn and COM marker, so outputs can carry the metadata. */
static void open_decompressor(j_decompress_ptr src, const unsigned char* jpeg,
                              unsigned long size) {
    jpeg_create_decompress(src);
    jpeg_mem_src(src, jpeg, size);
    jpeg_save_markers(src, JPEG_COM, 0xFFFF);
    for (int app = 0; app < 16; app++) jpeg_save_markers(src, JPEG_APP0 + app, 0xFFFF);
    jpeg_read_header(src, TRUE);
}

/* A compressor with the source's tables and sampling, for a new image size. */
static void start_compressor(j_compress_ptr dst, j_decompress_ptr src,
                             int width, int height) {
    jpeg_create_compress(dst);
    jpeg_copy_critical_parameters(src, dst);
    dst->image_width = (JDIMENSION)width;
    dst->image_height = (JDIMENSION)height;
}

static int has_prefix(jpeg_saved_marker_ptr marker, const char* prefix,
                      unsigned int length) {
    return marker->data_length >= length && memcmp(marker->data, prefix, length) == 0;
}

/* The JC_STRIP_* kind of a saved marker, or 0 for JFIF and Adobe markers, which
   libjpeg writes itself to match the output's color space. */
static int marker_kind(jpeg_saved_marker_ptr marker) {
    switch (marker->marker) {
    case JPEG_COM:
        return JC_STRIP_COMMENTS;
    case JPEG_APP0:
        return has_prefix(marker, "JFIF\0", 5) ? 0 : JC_STRIP_OTHER;
    case JPEG_APP0 + 1:
        if (has_prefix(marker, "Exif\0\0", 6)) return JC_STRIP_EXIF;
        if (has_prefix(marker, "http://ns.adobe.com/xap/1.0/\0", 29) ||
            has_prefix(marker, "http://ns.adobe.com/xmp/extension/\0", 35))
            return JC_STRIP_XMP;
        return JC_STRIP_OTHER;
    case JPEG_APP0 + 2:
        return has_prefix(marker, "ICC_PROFILE\0", 12) ? JC_STRIP_ICC : JC_STRIP_OTHER;
    case JPEG_APP0 + 14:
        return has_prefix(marker, "Adobe", 5) ? 0 : JC_STRIP_OTHER;
    default:
        return JC_STRIP_OTHER;
    }
}

/* Encode `coefficients`, carrying the markers of `src` that `strip` keeps. */
static void write_jpeg(j_compress_ptr dst, jvirt_barray_ptr* coefficients,
                       j_decompress_ptr src, int strip,
                       unsigned char** out, unsigned long* out_size) {
    jpeg_mem_dest(dst, out, out_size);
    jpeg_write_coefficients(dst, coefficients);
    for (jpeg_saved_marker_ptr marker = src->marker_list; marker; marker = marker->next) {
        int kind = marker_kind(marker);
        if (kind && !(kind & strip))
            jpeg_write_marker(dst, marker->marker, marker->data, marker->data_length);
    }
    jpeg_finish_compress(dst);
}

static void read_layout(j_decompress_ptr src, jc_layout* layout) {
    layout->width = (int)src->image_width;
    layout->height = (int)src->image_height;
    layout->num_components = src->num_components;
    for (int ci = 0; ci < src->num_components; ci++) {
        const jpeg_component_info* comp = &src->comp_info[ci];
        jc_component_layout* out = &layout->components[ci];
        out->h_samp_factor = comp->h_samp_factor;
        out->v_samp_factor = comp->v_samp_factor;
        out->width_in_blocks = (int)comp->width_in_blocks;
        out->height_in_blocks = (int)comp->height_in_blocks;
    }
}

void jc_free(unsigned char* buffer) { free(buffer); }

/* LIBJPEG_TURBO_VERSION is a bare token (3.2.0), so expand, then stringify. */
#define JC_STRINGIFY(token) #token
#define JC_EXPAND_AND_STRINGIFY(macro) JC_STRINGIFY(macro)

const char* jc_libjpeg_turbo_version(void) {
    return JC_EXPAND_AND_STRINGIFY(LIBJPEG_TURBO_VERSION);
}

int jc_read_layout(const unsigned char* jpeg, unsigned long size,
                   jc_layout* layout, jc_messages* messages) {
    struct jpeg_decompress_struct src = {0};
    jc_error error;

    src.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jpeg_destroy_decompress(&src);
        return JC_ERROR_JPEG;
    }
    open_decompressor(&src, jpeg, size);
    read_layout(&src, layout);
    jpeg_destroy_decompress(&src);
    return JC_OK;
}

/* ---- whole-image transcoding ------------------------------------------ */

typedef void (*coefficient_editor)(j_decompress_ptr src,
                                   jvirt_barray_ptr* coefficients,
                                   const void* context);

/* Re-encode `jpeg` with its own tables and sampling, after `edit` (if any)
   has rewritten its coefficients. */
static int transcode(const unsigned char* jpeg, unsigned long size,
                     coefficient_editor edit, const void* context,
                     int optimize, int progressive, int strip,
                     unsigned char** out, unsigned long* out_size,
                     jc_messages* messages) {
    struct jpeg_decompress_struct src = {0};
    struct jpeg_compress_struct dst = {0};
    jvirt_barray_ptr* coefficients;
    jc_error error;

    src.err = dst.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jpeg_destroy_compress(&dst);
        jpeg_destroy_decompress(&src);
        return JC_ERROR_JPEG;
    }
    open_decompressor(&src, jpeg, size);
    coefficients = jpeg_read_coefficients(&src);
    if (edit) edit(&src, coefficients, context);

    start_compressor(&dst, &src, (int)src.image_width, (int)src.image_height);
    /* Only ever switch optimization on: high-precision JPEGs require it. */
    if (optimize) dst.optimize_coding = TRUE;
    if (progressive) jpeg_simple_progression(&dst);
    write_jpeg(&dst, coefficients, &src, strip, out, out_size);

    jpeg_destroy_compress(&dst);
    jpeg_destroy_decompress(&src);
    return JC_OK;
}

int jc_optimize(const unsigned char* jpeg, unsigned long size, int progressive,
                int strip, unsigned char** out, unsigned long* out_size,
                jc_messages* messages) {
    return transcode(jpeg, size, NULL, NULL, TRUE, progressive, strip, out,
                     out_size, messages);
}

static int is_sample(int value) { return value >= 0 && value <= 255; }

static int check_color(const jc_color* color, jc_messages* messages) {
    if (!is_sample(color->r) || !is_sample(color->g) || !is_sample(color->b))
        return fail(JC_ERROR_ARGUMENT, messages,
                    "color (%d, %d, %d) must have components in 0..255",
                    color->r, color->g, color->b);
    return JC_OK;
}

/* The sample value a flat block of `component` must decode to. */
/* Component levels are kept in millionths, as integers: the YCbCr coefficients
   have six decimals, so the levels, and the DCs derived from them, are exact.
   With floating point, white could come out as 254.99999999999997 on some
   platforms (e.g. with fused multiply-add) and get another DC. */
#define LEVEL_SCALE 1000000LL

static long long fill_level(const jc_color* color, int component, int num_components) {
    long long r = color->r, g = color->g, b = color->b;
    /* JFIF full-range YCbCr, as in libjpeg's rgb_ycc_convert. */
    long long ycc[3] = {
        299000 * r + 587000 * g + 114000 * b,
        -168736 * r - 331264 * g + 500000 * b + 128 * LEVEL_SCALE,
        500000 * r - 418688 * g - 81312 * b + 128 * LEVEL_SCALE,
    };
    if (num_components == 1) return ycc[0];
    if (component >= 3) return 128 * LEVEL_SCALE;
    if (!color->color_transform)
        return (component == 0 ? r : component == 1 ? g : b) * LEVEL_SCALE;
    return ycc[component];
}

/* The DC coefficient of a flat block of component `ci` showing `color`. A
   block holding only a DC term decodes to 128 + DC * quant / 8. At the 0 and
   255 extremes the DC is rounded outwards: decoding clips the overshoot, while
   the nearest DC can fall short (e.g. 254 for white) with a coarse quant. */
static JCOEF fill_dc(j_decompress_ptr src, const jc_color* color, int ci) {
    const jpeg_component_info* comp = &src->comp_info[ci];
    int dc_quant = src->quant_tbl_ptrs[comp->quant_tbl_no]->quantval[0];
    long long level = fill_level(color, ci, src->num_components);
    long long dc_scaled = DCTSIZE * (level - 128 * LEVEL_SCALE);
    long long divisor = dc_quant * LEVEL_SCALE;
    if (level >= 255 * LEVEL_SCALE) return (JCOEF)ceil_div(dc_scaled, divisor);
    if (level <= 0) return (JCOEF)floor_div(dc_scaled, divisor);
    return (JCOEF)round_div(dc_scaled, divisor);
}

static void set_flat_blocks(JBLOCKROW blocks, int count, JCOEF dc) {
    memset(blocks, 0, count * sizeof(JBLOCK));
    for (int col = 0; col < count; col++) blocks[col][0] = dc;
}

static void fill_blocks(j_decompress_ptr src, jvirt_barray_ptr* coefficients,
                        const void* context) {
    const jc_color* color = (const jc_color*)context;
    for (int ci = 0; ci < src->num_components; ci++) {
        const jpeg_component_info* comp = &src->comp_info[ci];
        JCOEF dc = fill_dc(src, color, ci);
        for (int row = 0; row < (int)comp->height_in_blocks; row++)
            set_flat_blocks(block_row((j_common_ptr)src, coefficients[ci], row, TRUE),
                            (int)comp->width_in_blocks, dc);
    }
}

int jc_fill(const unsigned char* template_jpeg, unsigned long size,
            jc_color color, int strip,
            unsigned char** out, unsigned long* out_size,
            jc_messages* messages) {
    int status = check_color(&color, messages);
    if (status != JC_OK) return status;
    return transcode(template_jpeg, size, fill_blocks, &color, FALSE, FALSE, strip,
                     out, out_size, messages);
}

static void replace_blocks(j_decompress_ptr src, jvirt_barray_ptr* coefficients,
                           const void* context) {
    const short* const* components = (const short* const*)context;
    for (int ci = 0; ci < src->num_components; ci++) {
        const jpeg_component_info* comp = &src->comp_info[ci];
        const short* in = components[ci];
        for (int row = 0; row < (int)comp->height_in_blocks; row++) {
            memcpy(block_row((j_common_ptr)src, coefficients[ci], row, TRUE), in,
                   comp->width_in_blocks * sizeof(JBLOCK));
            in += comp->width_in_blocks * DCTSIZE2;
        }
    }
}

int jc_encode_coefficients(const unsigned char* template_jpeg, unsigned long size,
                           const short* const* components, int optimize,
                           int strip, unsigned char** out, unsigned long* out_size,
                           jc_messages* messages) {
    return transcode(template_jpeg, size, replace_blocks, components, optimize,
                     FALSE, strip, out, out_size, messages);
}

/* ---- sources and regions ---------------------------------------------- */

/* jpeg_read_coefficients consumes the whole input, so a source keeps no
   reference to the bytes it was opened from. */
struct jc_source {
    struct jpeg_decompress_struct decompressor;
    jvirt_barray_ptr* coefficients;
};

int jc_source_open(const unsigned char* jpeg, unsigned long size,
                   jc_source** out, jc_messages* messages) {
    jc_source* source = (jc_source*)calloc(1, sizeof *source);
    jc_error error;

    if (!source) return fail(JC_ERROR_MEMORY, messages, "out of memory");
    source->decompressor.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jc_source_close(source);
        return JC_ERROR_JPEG;
    }
    open_decompressor(&source->decompressor, jpeg, size);
    source->coefficients = jpeg_read_coefficients(&source->decompressor);
    *out = source;
    return JC_OK;
}

void jc_source_close(jc_source* source) {
    if (!source) return;
    jpeg_destroy_decompress(&source->decompressor);
    free(source);
}

void jc_source_layout(const jc_source* source, jc_layout* layout) {
    read_layout((j_decompress_ptr)&source->decompressor, layout);
}

int jc_source_read_component(jc_source* source, int component, short* out,
                             jc_messages* messages) {
    j_decompress_ptr src = &source->decompressor;
    jc_error error;

    if (component < 0 || component >= src->num_components)
        return fail(JC_ERROR_ARGUMENT, messages, "component %d out of range", component);
    src->err = init_error(&error, messages);
    if (setjmp(error.jump)) return JC_ERROR_JPEG;

    const jpeg_component_info* comp = &src->comp_info[component];
    jvirt_barray_ptr array = source->coefficients[component];
    for (int row = 0; row < (int)comp->height_in_blocks; row++) {
        memcpy(out, block_row((j_common_ptr)src, array, row, FALSE),
               comp->width_in_blocks * sizeof(JBLOCK));
        out += comp->width_in_blocks * DCTSIZE2;
    }
    return JC_OK;
}

typedef struct {
    int pixels_per_block_x, pixels_per_block_y;
    int width_in_blocks, height_in_blocks; /* whole MCUs covering the region */
    int v_samp_factor;
} component_geometry;

/* How a width x height region maps onto the blocks of each component. */
typedef struct {
    int width, height;
    int mcu_width, mcu_height;
    int num_components;
    component_geometry components[MAX_COMPONENTS];
} region_geometry;

static int compute_geometry(j_decompress_ptr src, int width, int height,
                            region_geometry* geometry, jc_messages* messages) {
    if (width <= 0 || height <= 0 || width > JPEG_MAX_DIMENSION ||
        height > JPEG_MAX_DIMENSION)
        return fail(JC_ERROR_ARGUMENT, messages,
                    "size %dx%d must be between 1 and %d pixels per side",
                    width, height, (int)JPEG_MAX_DIMENSION);
    geometry->width = width;
    geometry->height = height;
    geometry->mcu_width = src->max_h_samp_factor * DCTSIZE;
    geometry->mcu_height = src->max_v_samp_factor * DCTSIZE;
    geometry->num_components = src->num_components;
    for (int ci = 0; ci < src->num_components; ci++) {
        const jpeg_component_info* comp = &src->comp_info[ci];
        component_geometry* out = &geometry->components[ci];
        if (src->max_h_samp_factor % comp->h_samp_factor ||
            src->max_v_samp_factor % comp->v_samp_factor)
            return fail(JC_ERROR_JPEG, messages,
                        "component %d has unsupported sampling factors %dx%d",
                        ci, comp->h_samp_factor, comp->v_samp_factor);
        out->pixels_per_block_x = geometry->mcu_width / comp->h_samp_factor;
        out->pixels_per_block_y = geometry->mcu_height / comp->v_samp_factor;
        out->width_in_blocks =
            (JDIMENSION)ceil_div(width, geometry->mcu_width) * comp->h_samp_factor;
        out->height_in_blocks =
            (JDIMENSION)ceil_div(height, geometry->mcu_height) * comp->v_samp_factor;
        out->v_samp_factor = comp->v_samp_factor;
    }
    return JC_OK;
}

/* Per component, the DC of the flat blocks that fill a region outside the
   image, if a background is set. */
typedef struct {
    int enabled;
    JCOEF dc[MAX_COMPONENTS];
} region_background;

static int prepare_background(j_decompress_ptr src, const jc_color* color,
                              region_background* background, jc_messages* messages) {
    memset(background, 0, sizeof *background);
    background->enabled = color != NULL;
    if (!color) return JC_OK;
    int status = check_color(color, messages);
    if (status != JC_OK) return status;
    for (int ci = 0; ci < src->num_components; ci++)
        background->dc[ci] = fill_dc(src, color, ci);
    return JC_OK;
}

/* A region with a background may run past the right and bottom edges. */
static int check_region(j_decompress_ptr src, const region_geometry* geometry,
                        const region_background* background, int x, int y,
                        jc_messages* messages) {
    int image_width = (int)src->image_width, image_height = (int)src->image_height;
    if (x < 0 || y < 0 ||
        (!background->enabled && (geometry->width > image_width - x ||
                                  geometry->height > image_height - y)))
        return fail(JC_ERROR_ARGUMENT, messages,
                    "region (%d, %d, %d, %d) exceeds the %dx%d image",
                    x, y, geometry->width, geometry->height, image_width, image_height);
    if (x % geometry->mcu_width || y % geometry->mcu_height)
        return fail(JC_ERROR_ARGUMENT, messages,
                    "region origin (%d, %d) is not aligned to the %dx%d MCU grid",
                    x, y, geometry->mcu_width, geometry->mcu_height);
    return JC_OK;
}

static void request_arrays(j_common_ptr owner, const region_geometry* geometry,
                           jvirt_barray_ptr* arrays) {
    for (int ci = 0; ci < geometry->num_components; ci++) {
        const component_geometry* comp = &geometry->components[ci];
        arrays[ci] = (*owner->mem->request_virt_barray)(
            owner, JPOOL_IMAGE, FALSE, (JDIMENSION)comp->width_in_blocks,
            (JDIMENSION)comp->height_in_blocks, (JDIMENSION)comp->v_samp_factor);
    }
    (*owner->mem->realize_virt_arrays)(owner);
}

/* A region at the origin and inside the image needs no copy: as in jpegtran,
   the source's blocks are encoded directly. The encoder reads only the blocks
   within the region's size and pads partial MCUs itself, so the output is the
   same as from a copy. */
static int is_leading_region(j_decompress_ptr src, const region_geometry* geometry,
                             int x, int y) {
    return x == 0 && y == 0 && geometry->width <= (int)src->image_width &&
           geometry->height <= (int)src->image_height;
}

/* Copy the region's blocks from the source. With a background, blocks wholly
   outside the image (including the source's MCU padding) are filled instead. */
static void copy_region(jc_source* source, j_common_ptr owner,
                        jvirt_barray_ptr* arrays, const region_geometry* geometry,
                        const region_background* background, int x, int y) {
    j_common_ptr src = (j_common_ptr)&source->decompressor;
    for (int ci = 0; ci < geometry->num_components; ci++) {
        const component_geometry* comp = &geometry->components[ci];
        const jpeg_component_info* image = &source->decompressor.comp_info[ci];
        int first_col = x / comp->pixels_per_block_x;
        int first_row = y / comp->pixels_per_block_y;
        for (int row = 0; row < comp->height_in_blocks; row++) {
            JBLOCKROW out = block_row(owner, arrays[ci], row, TRUE);
            int source_row = first_row + row;
            int copy_cols = comp->width_in_blocks;
            if (background->enabled)
                copy_cols = source_row < (int)image->height_in_blocks
                    ? min_int(max_int((int)image->width_in_blocks - first_col, 0),
                              comp->width_in_blocks)
                    : 0;
            if (copy_cols > 0)
                memcpy(out,
                       block_row(src, source->coefficients[ci], source_row, FALSE) +
                           first_col,
                       copy_cols * sizeof(JBLOCK));
            if (copy_cols < comp->width_in_blocks)
                set_flat_blocks(out + copy_cols, comp->width_in_blocks - copy_cols,
                                background->dc[ci]);
        }
    }
}

int jc_source_crop(jc_source* source, int x, int y, int width, int height,
                   const jc_color* background_color, int strip,
                   unsigned char** out, unsigned long* out_size,
                   jc_messages* messages) {
    j_decompress_ptr src = &source->decompressor;
    struct jpeg_compress_struct dst = {0};
    jvirt_barray_ptr arrays[MAX_COMPONENTS];
    region_geometry geometry;
    region_background background;
    jc_error error;
    int status = compute_geometry(src, width, height, &geometry, messages);

    if (status == JC_OK)
        status = prepare_background(src, background_color, &background, messages);
    if (status == JC_OK)
        status = check_region(src, &geometry, &background, x, y, messages);
    if (status != JC_OK) return status;
    src->err = dst.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jpeg_destroy_compress(&dst);
        return JC_ERROR_JPEG;
    }
    start_compressor(&dst, src, width, height);
    if (is_leading_region(src, &geometry, x, y)) {
        write_jpeg(&dst, source->coefficients, src, strip, out, out_size);
    } else {
        request_arrays((j_common_ptr)&dst, &geometry, arrays);
        copy_region(source, (j_common_ptr)&dst, arrays, &geometry, &background, x, y);
        write_jpeg(&dst, arrays, src, strip, out, out_size);
    }
    jpeg_destroy_compress(&dst);
    return JC_OK;
}

/* ---- cropper ---------------------------------------------------------- */

/* The output arrays of croppers and composers are owned by a source's
   decompressor: arrays in the compressor's image pool would be freed by every
   jpeg_finish_compress. */
struct jc_cropper {
    jc_source* source;
    struct jpeg_compress_struct compressor;
    region_geometry geometry;
    region_background background;
    int strip;
    jvirt_barray_ptr arrays[MAX_COMPONENTS];
};

int jc_cropper_open(jc_source* source, int width, int height,
                    const jc_color* background_color, int strip,
                    jc_cropper** out, jc_messages* messages) {
    j_decompress_ptr src = &source->decompressor;
    region_geometry geometry;
    region_background background;
    jc_cropper* cropper;
    jc_error error;
    int status = compute_geometry(src, width, height, &geometry, messages);

    if (status == JC_OK)
        status = prepare_background(src, background_color, &background, messages);
    if (status == JC_OK)
        status = check_region(src, &geometry, &background, 0, 0, messages);
    if (status != JC_OK) return status;
    cropper = (jc_cropper*)calloc(1, sizeof *cropper);
    if (!cropper) return fail(JC_ERROR_MEMORY, messages, "out of memory");
    cropper->source = source;
    cropper->geometry = geometry;
    cropper->background = background;
    cropper->strip = strip;
    src->err = cropper->compressor.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jc_cropper_close(cropper);
        return JC_ERROR_JPEG;
    }
    start_compressor(&cropper->compressor, src, width, height);
    request_arrays((j_common_ptr)src, &geometry, cropper->arrays);
    *out = cropper;
    return JC_OK;
}

void jc_cropper_close(jc_cropper* cropper) {
    if (!cropper) return;
    jpeg_destroy_compress(&cropper->compressor);
    free(cropper);
}

int jc_cropper_crop(jc_cropper* cropper, int x, int y,
                    unsigned char** out, unsigned long* out_size,
                    jc_messages* messages) {
    j_decompress_ptr src = &cropper->source->decompressor;
    jc_error error;
    int status = check_region(src, &cropper->geometry, &cropper->background, x, y,
                              messages);

    if (status != JC_OK) return status;
    src->err = cropper->compressor.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jpeg_abort_compress(&cropper->compressor); /* stay usable */
        return JC_ERROR_JPEG;
    }
    if (is_leading_region(src, &cropper->geometry, x, y)) {
        write_jpeg(&cropper->compressor, cropper->source->coefficients, src,
                   cropper->strip, out, out_size);
        return JC_OK;
    }
    copy_region(cropper->source, (j_common_ptr)src, cropper->arrays,
                &cropper->geometry, &cropper->background, x, y);
    write_jpeg(&cropper->compressor, cropper->arrays, src, cropper->strip, out,
               out_size);
    return JC_OK;
}

/* ---- composer --------------------------------------------------------- */

/* Per component, in blocks: the stride between sources on the plane, and how
   much of a source the last one in a row/column keeps. */
typedef struct {
    int content_cols, content_rows;
    int source_cols, source_rows;
} component_grid;

struct jc_composer {
    jc_source** sources;
    jc_point* cells;
    int count;
    jc_point last_cell;
    struct jpeg_compress_struct compressor;
    region_geometry geometry;
    region_background background; /* without one, uncovered blocks are gray */
    int strip;                    /* markers come from sources[0] */
    component_grid grids[MAX_COMPONENTS];
    jvirt_barray_ptr arrays[MAX_COMPONENTS]; /* owned by sources[0] */
};

static int same_sampling(j_decompress_ptr a, j_decompress_ptr b) {
    if (a->num_components != b->num_components) return 0;
    for (int ci = 0; ci < a->num_components; ci++)
        if (a->comp_info[ci].h_samp_factor != b->comp_info[ci].h_samp_factor ||
            a->comp_info[ci].v_samp_factor != b->comp_info[ci].v_samp_factor)
            return 0;
    return 1;
}

static int check_composition(jc_source* const* sources, const jc_point* cells,
                             int count, jc_size content,
                             const region_geometry* geometry, jc_messages* messages) {
    if (content.width <= 0 || content.height <= 0 ||
        content.width % geometry->mcu_width || content.height % geometry->mcu_height)
        return fail(JC_ERROR_ARGUMENT, messages,
                    "content size %dx%d is not a positive multiple of the %dx%d MCU",
                    content.width, content.height,
                    geometry->mcu_width, geometry->mcu_height);
    for (int i = 0; i < count; i++) {
        if (cells[i].x < 0 || cells[i].y < 0)
            return fail(JC_ERROR_ARGUMENT, messages,
                        "source %d has negative grid cell (%d, %d)",
                        i, cells[i].x, cells[i].y);
        if (!same_sampling(&sources[0]->decompressor, &sources[i]->decompressor))
            return fail(JC_ERROR_ARGUMENT, messages,
                        "source %d's components or sampling differ from source 0's", i);
    }
    return JC_OK;
}

int jc_composer_open(jc_source* const* sources, const jc_point* cells, int count,
                     jc_size tile, jc_size content, jc_size source_size,
                     const jc_color* background_color, int strip,
                     jc_composer** out, jc_messages* messages) {
    j_decompress_ptr first;
    region_geometry geometry;
    region_background background;
    jc_composer* composer;
    jc_error error;
    int status;

    if (count < 1)
        return fail(JC_ERROR_ARGUMENT, messages, "compose needs at least one source");
    first = &sources[0]->decompressor;
    status = compute_geometry(first, tile.width, tile.height, &geometry, messages);
    if (status == JC_OK)
        status = check_composition(sources, cells, count, content, &geometry, messages);
    if (status == JC_OK)
        status = prepare_background(first, background_color, &background, messages);
    if (status != JC_OK) return status;

    composer = (jc_composer*)calloc(1, sizeof *composer);
    if (composer) {
        composer->sources = (jc_source**)malloc(count * sizeof *sources);
        composer->cells = (jc_point*)malloc(count * sizeof *cells);
    }
    if (!composer || !composer->sources || !composer->cells) {
        jc_composer_close(composer);
        return fail(JC_ERROR_MEMORY, messages, "out of memory");
    }
    memcpy(composer->sources, sources, count * sizeof *sources);
    memcpy(composer->cells, cells, count * sizeof *cells);
    composer->count = count;
    for (int i = 0; i < count; i++) {
        composer->last_cell.x = max_int(composer->last_cell.x, cells[i].x);
        composer->last_cell.y = max_int(composer->last_cell.y, cells[i].y);
    }
    composer->geometry = geometry;
    composer->background = background;
    composer->strip = strip;
    for (int ci = 0; ci < geometry.num_components; ci++) {
        const component_geometry* comp = &geometry.components[ci];
        component_grid* grid = &composer->grids[ci];
        grid->content_cols = content.width / comp->pixels_per_block_x;
        grid->content_rows = content.height / comp->pixels_per_block_y;
        grid->source_cols = source_size.width / comp->pixels_per_block_x;
        grid->source_rows = source_size.height / comp->pixels_per_block_y;
    }

    first->err = composer->compressor.err = init_error(&error, messages);
    if (setjmp(error.jump)) {
        jc_composer_close(composer);
        return JC_ERROR_JPEG;
    }
    start_compressor(&composer->compressor, first, tile.width, tile.height);
    request_arrays((j_common_ptr)first, &geometry, composer->arrays);
    *out = composer;
    return JC_OK;
}

void jc_composer_close(jc_composer* composer) {
    if (!composer) return;
    jpeg_destroy_compress(&composer->compressor);
    free(composer->sources);
    free(composer->cells);
    free(composer);
}

static jc_source* source_at(const jc_composer* composer, int cell_x, int cell_y) {
    for (int i = 0; i < composer->count; i++)
        if (composer->cells[i].x == cell_x && composer->cells[i].y == cell_y)
            return composer->sources[i];
    return NULL;
}

/* Row `row` of component `ci` in the source at the given cell, or NULL when
   nothing of it is kept; *kept_cols receives how many of its blocks are. */
static JBLOCKROW kept_block_row(const jc_composer* composer, int ci,
                                int cell_x, int cell_y, int row, int* kept_cols) {
    const component_grid* grid = &composer->grids[ci];
    jc_source* source = source_at(composer, cell_x, cell_y);
    const jpeg_component_info* comp;
    int rows, cols;

    if (!source) return NULL;
    comp = &source->decompressor.comp_info[ci];
    rows = (int)comp->height_in_blocks;
    cols = (int)comp->width_in_blocks;
    /* Without a background, keep the arrays' padding to whole MCUs too, as
       there is nothing else to show there; with one, it shows the background
       as in a crop. */
    if (!composer->background.enabled) {
        rows = round_up(rows, comp->v_samp_factor);
        cols = round_up(cols, comp->h_samp_factor);
    }
    if (row >= min_int(grid->source_rows, rows)) return NULL;
    *kept_cols = min_int(grid->source_cols, cols);
    return block_row((j_common_ptr)&source->decompressor, source->coefficients[ci],
                     row, FALSE);
}

/* Write every block of component `ci` for the tile at plane (x, y), copying
   from the covering source or filling with the background (or neutral gray)
   where there is none. */
static void compose_component(const jc_composer* composer, int ci, int x, int y) {
    const component_geometry* comp = &composer->geometry.components[ci];
    const component_grid* grid = &composer->grids[ci];
    j_common_ptr owner = (j_common_ptr)&composer->sources[0]->decompressor;
    int first_col = x / comp->pixels_per_block_x;
    int first_row = y / comp->pixels_per_block_y;

    for (int row = 0; row < comp->height_in_blocks; row++) {
        JBLOCKROW out = block_row(owner, composer->arrays[ci], row, TRUE);
        int plane_row = first_row + row;
        /* Past the last cell, the last source keeps covering (its tail). */
        int cell_y = min_int(plane_row / grid->content_rows, composer->last_cell.y);
        int source_row = plane_row - cell_y * grid->content_rows;
        int current_cell_x = -1, kept_cols = 0;
        JBLOCKROW in = NULL;

        for (int col = 0; col < comp->width_in_blocks; col++) {
            int plane_col = first_col + col;
            int cell_x = min_int(plane_col / grid->content_cols, composer->last_cell.x);
            int source_col = plane_col - cell_x * grid->content_cols;
            if (cell_x != current_cell_x) {
                in = kept_block_row(composer, ci, cell_x, cell_y, source_row, &kept_cols);
                current_cell_x = cell_x;
            }
            if (in && source_col < kept_cols)
                memcpy(out[col], in[source_col], sizeof(JBLOCK));
            else
                set_flat_blocks(out + col, 1, composer->background.dc[ci]);
        }
    }
}

int jc_composer_tile(jc_composer* composer, int x, int y,
                     unsigned char** out, unsigned long* out_size,
                     jc_messages* messages) {
    const region_geometry* geometry = &composer->geometry;
    jc_error error;

    if (x < 0 || y < 0 || x % geometry->mcu_width || y % geometry->mcu_height)
        return fail(JC_ERROR_ARGUMENT, messages,
                    "tile position (%d, %d) must be non-negative and on the "
                    "%dx%d MCU grid",
                    x, y, geometry->mcu_width, geometry->mcu_height);
    composer->compressor.err = init_error(&error, messages);
    for (int i = 0; i < composer->count; i++)
        composer->sources[i]->decompressor.err = &error.pub;
    if (setjmp(error.jump)) {
        jpeg_abort_compress(&composer->compressor); /* stay usable */
        return JC_ERROR_JPEG;
    }
    for (int ci = 0; ci < geometry->num_components; ci++)
        compose_component(composer, ci, x, y);
    write_jpeg(&composer->compressor, composer->arrays,
               &composer->sources[0]->decompressor, composer->strip, out, out_size);
    return JC_OK;
}
