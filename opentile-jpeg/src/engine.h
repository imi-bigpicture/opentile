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

/* Lossless DCT-coefficient operations on libjpeg-turbo's classic API.

   Every function returns a jc_status and reports into a zero-initialized
   jc_messages: the error on failure, and libjpeg-turbo's warnings (such as
   corrupt data it recovered from) either way. Encoded JPEGs are returned as
   malloc'd buffers that the caller releases with jc_free. Positions and sizes
   are in pixels and must follow the source's MCU grid. */

#ifndef OPENTILE_JPEG_ENGINE_H
#define OPENTILE_JPEG_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#define JC_MESSAGE_LEN 200
#define JC_MAX_COMPONENTS 10

typedef enum {
    JC_OK = 0,
    JC_ERROR_JPEG,     /* libjpeg-turbo rejected the data */
    JC_ERROR_ARGUMENT, /* invalid geometry or index */
    JC_ERROR_MEMORY,
    JC_ERROR_TOO_LARGE /* the image exceeds max_pixels */
} jc_status;

typedef struct {
    char error[JC_MESSAGE_LEN];
    char first_warning[JC_MESSAGE_LEN];
    int warning_count;
} jc_messages;

typedef struct { int x, y; } jc_point;
typedef struct { int width, height; } jc_size;

/* A solid 0..255 color; color_transform converts it from RGB to the frame's
   YCbCr (see jc_fill). */
typedef struct {
    int r, g, b;
    int color_transform;
} jc_color;

/* Metadata an output can strip, as a bitmask in the `strip` arguments. By
   default every APPn and COM marker of the source is copied, except JFIF and
   Adobe markers, which libjpeg writes itself to match the output. */
enum {
    JC_STRIP_ICC = 1 << 0,      /* APP2 ICC_PROFILE */
    JC_STRIP_EXIF = 1 << 1,     /* APP1 Exif */
    JC_STRIP_XMP = 1 << 2,      /* APP1 XMP, standard and extended */
    JC_STRIP_COMMENTS = 1 << 3, /* COM */
    JC_STRIP_OTHER = 1 << 4     /* any other APPn, e.g. IPTC in APP13 */
};

typedef struct {
    int h_samp_factor, v_samp_factor;
    int width_in_blocks, height_in_blocks;
} jc_component_layout;

typedef struct {
    int width, height;
    int num_components;
    jc_component_layout components[JC_MAX_COMPONENTS];
} jc_layout;

typedef struct jc_source jc_source;
typedef struct jc_cropper jc_cropper;
typedef struct jc_composer jc_composer;

void jc_free(unsigned char* buffer);
/* The libjpeg-turbo version compiled in, e.g. "3.2.0". */
const char* jc_libjpeg_turbo_version(void);

/* Reading a JPEG's coefficients allocates for its whole image up front, so the
   functions taking `max_pixels` first refuse an image of more pixels than that
   with JC_ERROR_TOO_LARGE; 0 means no limit. */

int jc_read_layout(const unsigned char* jpeg, unsigned long size,
                   jc_layout* layout, jc_messages* messages);
int jc_optimize(const unsigned char* jpeg, unsigned long size, int progressive,
                int strip, long long max_pixels, unsigned char** out,
                unsigned long* out_size, jc_messages* messages);
int jc_fill(const unsigned char* template_jpeg, unsigned long size,
            jc_color color, int strip, long long max_pixels,
            unsigned char** out, unsigned long* out_size, jc_messages* messages);
/* components[i] holds component i's blocks row-major, 64 coefficients each. */
int jc_encode_coefficients(const unsigned char* template_jpeg, unsigned long size,
                           const short* const* components, int optimize,
                           int strip, unsigned char** out, unsigned long* out_size,
                           jc_messages* messages);

/* A JPEG whose DCT coefficients are read once and then shared by crops,
   croppers and composers. */
int jc_source_open(const unsigned char* jpeg, unsigned long size,
                   long long max_pixels, jc_source** source, jc_messages* messages);
void jc_source_close(jc_source* source);
void jc_source_layout(const jc_source* source, jc_layout* layout);
int jc_source_read_component(jc_source* source, int component, short* out,
                             jc_messages* messages);
/* With a NULL background the region must lie inside the image. With a
   background it may run past the right and bottom edges, and every block
   wholly outside the image is filled with that color. */
int jc_source_crop(jc_source* source, int x, int y, int width, int height,
                   const jc_color* background, int strip,
                   unsigned char** out, unsigned long* out_size,
                   jc_messages* messages);

/* Repeated same-size crops from one source, reusing a single compressor.
   Borrows the source, which must outlive the cropper. `background` is as for
   jc_source_crop. */
int jc_cropper_open(jc_source* source, int width, int height,
                    const jc_color* background, int strip,
                    jc_cropper** cropper, jc_messages* messages);
void jc_cropper_close(jc_cropper* cropper);
int jc_cropper_crop(jc_cropper* cropper, int x, int y,
                    unsigned char** out, unsigned long* out_size,
                    jc_messages* messages);

/* Re-tiles sources placed on a grid into output tiles. Source i covers the
   plane from cells[i] * content, and the last source in each row/column keeps
   up to `source_size` of its content. Borrows the sources, which must share
   their sampling and outlive the composer. Blocks no source covers are filled
   with `background`, or neutral gray when it is NULL. The output carries the
   metadata of sources[0]. */
int jc_composer_open(jc_source* const* sources, const jc_point* cells, int count,
                     jc_size tile, jc_size content, jc_size source_size,
                     const jc_color* background, int strip,
                     jc_composer** composer, jc_messages* messages);
void jc_composer_close(jc_composer* composer);
int jc_composer_tile(jc_composer* composer, int x, int y,
                     unsigned char** out, unsigned long* out_size,
                     jc_messages* messages);

#ifdef __cplusplus
}
#endif

#endif
