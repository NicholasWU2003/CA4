/*
 * Skeleton code for use with Computer Architecture 2023 assignment 4,
 * LIACS, Leiden University.
 */

#include "image.h"

#include <png.h>

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#if ! defined(PNG_SIMPLIFIED_READ_SUPPORTED)
#  error "Need libpng with simplified read support."
#endif

/* We use ARGB, 8-bits per component in this program. */
#define IMAGE_FORMAT PNG_FORMAT_ABGR


/* Allocate a new image with width, height and given rowstride. */
image_t *
image_new(const int width, const int height, const int rowstride)
{
  image_t *image = malloc(sizeof(image_t));
  if (!image)
    return NULL;

  image->width = width;
  image->height = height;
  if (rowstride == 0)
    image->rowstride = width * sizeof(uint32_t);
  else
    image->rowstride = rowstride;

  image->data = malloc(image->rowstride * height);
  if (!image->data)
    {
      free(image);
      return NULL;
    }

  return image;
}

/* Allocate a new image with the same attributes as the given image,
 * but a new data area.
 */
image_t *
image_new_from_image(const image_t *image)
{
  return image_new(image->width, image->height, image->rowstride);
}

/* Allocate a new image with the same attributes as the given image,
 * but transposed dimensions and new data area.
 */
image_t *
image_new_from_image_T(const image_t *image)
{
  return image_new(image->height, image->width, 0);
}


/* Release resources used by "image". */
void
image_free(image_t *image)
{
  if (image)
    {
      free(image->data);
      free(image);
    }
}

/* Verify the attributes of the two given images are equal. */
bool
image_equal_dimensions(const image_t *a, const image_t *b)
{
  return a->width == b->width &&
      a->height == b->height &&
      a->rowstride == b->rowstride;
}

/* Load an image from the given filename. Must be in PNG format. */
image_t *
image_new_from_pngfile(const char *filename)
{
  png_image pimage = { .opaque = NULL, .version = PNG_IMAGE_VERSION };

  png_image_begin_read_from_file(&pimage, filename);
  if (PNG_IMAGE_FAILED(pimage))
    {
      fprintf(stderr, "Failed to load image: %s\n", pimage.message);
      return NULL;
    }

  pimage.format = IMAGE_FORMAT;

  image_t *image = image_new(pimage.width, pimage.height, 0);
  if (!image)
    {
      fprintf(stderr, "Failed to allocate memory for image.\n");
      png_image_free(&pimage);
      return NULL;
    }

  png_image_finish_read(&pimage, NULL, image->data, image->rowstride, NULL);
  if (PNG_IMAGE_FAILED(pimage))
    {
      fprintf(stderr, "Failed to load image: %s\n", pimage.message);
      png_image_free(&pimage);
      image_free(image);
      return NULL;
    }

  png_image_free(&pimage);
  return image;
}

/* Save the given image to filename in PNG format. */
bool
image_save_as_pngfile(const image_t *image, const char *filename)
{
  png_image dst_image = { 0, };

  dst_image.version = PNG_IMAGE_VERSION;
  dst_image.width = image->width;
  dst_image.height = image->height;
  dst_image.format = IMAGE_FORMAT;

  png_image_write_to_file(&dst_image, filename, 0, image->data,
                          image->rowstride, NULL);
  if (PNG_IMAGE_FAILED(dst_image))
    {
      fprintf(stderr, "Failed to save image: %s\n", dst_image.message);
      png_image_free(&dst_image);
      return false;
    }

  png_image_free(&dst_image);

  return true;
}



/*
 * Routine to compare a given image to a reference image. Checks for
 * every pixel whether all channels are "close enough" to each other
 * (defined by EPS below). The values do not have to be equal, as
 * rounding errors may occur.
 *
 * Includes contributions by Miguel Blom and Koen Putman.
 */
#define EPS 0.075

int
image_compare(const image_t *computed, const image_t *reference, const int max_error)
{
  int width, height;
  int errors = 0;

  assert(computed->width == reference->width && computed->height == reference->height);

  width = computed->width;
  height = computed->height;

  for (int x = 0; x < width; ++x)
    {
      for (int y = 0; y < height; ++y)
        {
          rgba_t pa, pb;
          RGBA_unpack(pa, *image_get_pixel(computed, x, y));
          RGBA_unpack(pb, *image_get_pixel(reference, x, y));

          /* The values of the different channels of the pixels don't have
           * to be equal, but must be "close enough" to each other.
           */
          if (fabs(pa.x - pb.x) > EPS ||
              fabs(pa.y - pb.y) > EPS ||
              fabs(pa.z - pb.z) > EPS ||
              fabs(pa.w - pb.w) > EPS)
            {
              fprintf(stderr, "Pixel (%d, %d): %.4f %.4f %.4f %.4f (should be %.4f %.4f %.4f %.4f)\n",
                      x, y,
                      pa.x, pa.y, pa.z, pa.w,
                      pb.x, pb.y, pb.z, pb.w);

              errors += 1;

              if (max_error != -1 && errors >= max_error)
                return errors;
            }
        }
    }

  return errors;
}
