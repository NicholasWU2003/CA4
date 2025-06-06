/*
 * Skeleton code for use with Computer Architecture 2023 assignment 4,
 * LIACS, Leiden University.
 *
 * Task 1: grayscale kernel
 *
 */

#include "image.h"
#include "experiment.hpp"

#include <iostream>

#include <cstring>
#include <cerrno>

#include <getopt.h>

#include <sys/types.h>
#include <dirent.h>

/* Some simple assert macro and inline function to handle CUDA-errors
 * more easily.
 */
#define CUDA_ASSERT(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void
cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
              file, line);

      if (abort)
        exit(code);
    }
}


/*
 * CUDA kernels
 *
 * "op_" functions below to be converted to CUDA kernels. Remember that
 * every kernel should be declared as `__global__ void`. Every function
 * you want to call from a kernel, should begin with `__device__`.
 *
 * op_cuda_copy is a CUDA kernel provided as example.
 */

 __device__ float compute_intensity_CUDA(float4 color) {
    return color.w * (0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z);
}


/* Example kernel for an image copy operation. */
__global__ void
op_cuda_copy(uint32_t *dst, const uint32_t *src, const int rowstride,
             const int x, const int y,
             const int width, const int height)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

  if (xx < x || xx >= width || yy < y || y >= height)
    return;

  /* Get the pixel in src and store in dst. */
  uint32_t pixel = *image_get_pixel_data(src, rowstride, xx, yy);
  *image_get_pixel_data(dst, rowstride, xx, yy) = pixel;
}

__global__ void
op_cuda_RQ1(uint32_t *dst, const uint32_t *src, const int rowstride,
             const int x, const int y,
             const int width, const int height)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

  if (xx < x || xx >= width || yy < y || yy >= height)
    return;

  // Get the pixel in src
  uint32_t pixel = *image_get_pixel_data(src, rowstride, xx, yy);

  float4 color;
  RGBA_unpack(color, pixel);
  float intensity = compute_intensity_CUDA(color);
  float4 gray;
  gray.x = gray.y = gray.z = intensity;
  gray.w = color.w;

  uint32_t gray_pixel;
  RGBA_pack(gray_pixel, gray);
  *image_get_pixel_data(dst, rowstride, xx, yy) = gray_pixel;
}

__global__ void
op_cuda_RQ2(uint32_t *dst, const uint32_t *src, const int rowstride,
             const int x, const int y,
             const int width, const int height)
{
  const int base_xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int base_yy = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Process 4 pixels per thread in a 2x2 block
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      const int xx = base_xx * 2 + dx;
      const int yy = base_yy * 2 + dy;
      
      if (xx < x || xx >= width || yy < y || yy >= height)
        continue;

      // Get the pixel in src
      uint32_t pixel = *image_get_pixel_data(src, rowstride, xx, yy);

      float4 color;
      RGBA_unpack(color, pixel);
      float intensity = compute_intensity_CUDA(color);
      float4 gray;
      gray.x = gray.y = gray.z = intensity;
      gray.w = color.w;

      uint32_t gray_pixel;
      RGBA_pack(gray_pixel, gray);
      *image_get_pixel_data(dst, rowstride, xx, yy) = gray_pixel;
    }
  }
}


/*
 * CPU kernels
 *
 * Leave these functions in place, they are required for the "test
 * mode" to work. Make a copy in order to convert to a CUDA kernel.
 *
 */


/* Computes the intensity of @color. To do so, we use CIE 1931 weights
 * multiplied by alpha: Y = A( 0.2126R + 0.7152G + 0.0722B ).
 */
static inline float
compute_intensity(rgba_t color)
{
  return color.w * (0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z);
}

/* Computes the grayscale value for each pixel in @src and stores this in @dst.
 * @dst is expected to have been created already with the correct dimensions.
 * Safe to use a in-place operation.
 */
/* Do not remove this function, it is required for the "test mode" to work. */
void
op_grayscale(image_t *dst, const image_t *src)
{
  for (int x = 0; x < dst->width; x++)
    {
      for (int y = 0; y < dst->height; y++)
        {
          rgba_t color, gray;
          RGBA_unpack(color, *image_get_pixel(src, x, y));
          float intensity = compute_intensity(color);
          RGBA(gray, intensity, intensity, intensity, 1.f);
          RGBA_pack(*image_get_pixel(dst, x, y), gray);
        }
    }
}





/* Returns elapsed time in msec */
static float
run_cuda_kernel(image_t *background)
{
  /* TODO: allocate buffers to contain background image. */
  uint32_t *d_img = nullptr;
  size_t img_bytes = background->rowstride * background->height;
  cudaMalloc(&d_img, img_bytes);


   /* TODO: copy the input image to the background buffer allocated
   * on the GPU.
   */
  cudaMemcpy(d_img, background->data, img_bytes, cudaMemcpyHostToDevice);

  /* TODO: calculate the block size and number of thread blocks. */

  // for RQ1:
  // dim3 block(8, 8); 
  // dim3 grid((background->width + block.x - 1) / block.x,
  //           (background->height + block.y - 1) / block.y);


  // for RQ2:
  dim3 block(8, 8); 
  int PPTx = 2;
  int PPTy = 2;
  dim3 grid((background->width + (block.x * PPTx - 1)) / (block.x * PPTx),
            (background->height + (block.y * PPTy - 1)) / (block.y * PPTy));

  /* "computetime" will only include the actual time taken by the GPU
   * to perform the image operation. So, this excludes image loading,
   * saving and the memory transfers to/from the GPU.
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Start the timer */
  CUDA_ASSERT(cudaEventRecord(start));


  /* TODO: replace with CUDA kernel call. If you have multiple variants
   * of the kernel, you can choose which one to run here. Or make copies
   * of this run_cuda_kernel() function.
   */

  //   #if 0
  //   op_grayscale(background, background); /* in-place */
  // #endif
  // op_cuda_RQ1<<<grid, block>>>(d_img, d_img, background->rowstride, 0, 0, background->width, background->height);
  op_cuda_RQ2<<<grid, block>>>(d_img, d_img, background->rowstride, 0, 0, background->width, background->height);
  CUDA_ASSERT(cudaGetLastError());




  /* Stop timer */
  CUDA_ASSERT(cudaEventRecord(stop));
  CUDA_ASSERT(cudaEventSynchronize(stop));

  float msec = 0;
  CUDA_ASSERT(cudaEventElapsedTime(&msec, start, stop));

  cudaMemcpy(background->data, d_img, img_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_img);

  return msec;
}


static void
run_test(const std::string &infilename,
         const std::string &outfilename)
{
  std::cout << "Testing with " << infilename << " ...\n";

  image_t *background;
  background = image_new_from_pngfile(infilename.c_str());
  if (!background)
    return;

  /* Create a copy to be manipulated on CPU */
  image_t *original = image_new_from_image(background);
  std::memcpy(original->data, background->data,
              background->rowstride * background->height);

  /* Run CPU kernels */
  op_grayscale(original, original); /* in-place */

  /* Run GPU kernels */
  run_cuda_kernel(background);

  /* Compare the results */
  const int max_error = 64;
  int errors = image_compare(background, original, max_error);
  if (errors > 0)
    std::cerr << "Images do not match, " << errors
              << " errors detected (max=" << max_error << ").\n";
  else
    std::cerr << "Images match.\n";

  /* Write GPU result to PNG if requested */
  if (not outfilename.empty())
    {
      image_save_as_pngfile(background, outfilename.c_str());
      std::cerr << "Wrote GPU result to " << outfilename << "\n";
    }

  image_free(original);
  image_free(background);
}


/* Process a single image. Warning: does not detect errors.
 */
static void
process_images(size_t i, ExperimentTimer &timer,
               const std::string &infilename,
               const std::string &outfilename)
{
  /* Load image */
  image_t *background;

  auto startTime = ExperimentTimer::now();
  background = image_new_from_pngfile(infilename.c_str());
  if (!background)
    return;
  auto endTime = ExperimentTimer::now();

  timer.setLoadTime(i, endTime, startTime);

  float msec = run_cuda_kernel(background);
  timer.setComputeTime(i, msec / 1000.);

  /* Save results if desired and if applicable */
  if (not outfilename.empty())
    image_save_as_pngfile(background, outfilename.c_str());

  image_free(background);
}


/* Code to run a single experiment, depending on batchSize parameter. */
static void
run_experiment(Experiment &exp,
               const std::string &infile, const std::string &outfile,
               bool silentMode)
{
  ExperimentTimer timer(exp.addMeasurement());

  timer.start();

  if (not silentMode)
    std::cout << "Processing " << infile << " ...\n" << std::flush;

  process_images(0, timer, infile, outfile);

  /* Note that the full timing of the experiment will include image
   * loading & saving time and memory transfers to and from the GPU.
   * The memory transfers are not counted in the runtime of the
   * individual images.
   */
  timer.end();

  /* Print statistics */
  if (not silentMode)
    {
      std::cout << "====\n";
      exp.printCPUUtilization(exp.back(), std::cout);
      exp.printStatistics(exp.back(), std::cout);
      std::cout << "====\n";
    }
}


/*
 * Main function
 */

static void
show_help(const char *progName)
{
  std::cerr << "usage: " << progName << " [-t] [-s] [-c] [-r REPEAT] <infile> [outfile]\n"
            << "\n  where <infile>, and [outfile] are PNG files."
            << "\n  [outfile] is an optional parameter.\n"
            << "\n  -t  test mode: compares the GPU result to the CPU result. Optionally outputs "
            << "\n      the GPU results if outfile is specified."
            << "\n  -s  disables output of experiment summaries (silent mode)."
            << "\n  -c  outputs an overview of all experiment results in CSV format."
            << "\n  -r  configures the number of times the experiment is repeated.\n";
}


int
main(int argc, char **argv)
{
  char c;
  long int n_repeat = 1;
  bool csvOutput = false;
  bool silentMode = false;
  bool testMode = false;

  /* Command line parsing */
  const char *progName = argv[0];

  while ((c = getopt(argc, argv, "r:b:1cst")) != -1)
    {
      switch (c)
        {
          case 'r':
            n_repeat = std::strtol(optarg, NULL, 10);
            if (errno > 0 || n_repeat == 0)
              {
                std::cerr << "Could not convert n_repeat argument to integer.\n";
                exit(EXIT_FAILURE);
              }
            break;

          case 'c':
            csvOutput = true;
            break;

          case 's':
            silentMode = true;
            break;

          case 't':
            testMode = true;
            break;

          case 'h':
          default:
            show_help(progName);
            return EXIT_FAILURE;
            break;
        }
    }

  argc -= optind;
  argv += optind;

  if (argc < 1)
    {
      show_help(progName);
      return EXIT_FAILURE;
    }

  const char *infile = argv[0];
  const char *outfile = argv[1];

  if (testMode)
    {
      /* Test mode: process the first image found in "indir" and
       * compare the GPU result to the CPU result.
       */
      std::string infilename(infile);
      std::string outfilename = outfile ? std::string(outfile) : std::string();

      run_test(infilename, outfilename);
    }
  else
    {
      /* Run experiment the requested number of times. */
      std::vector<std::string> files {infile};
      Experiment exp(files, 1, true);
      for (int i = 0; i < n_repeat; ++i)
        run_experiment(exp, std::string(infile),
                       outfile ? std::string(outfile) : std::string(),
                       silentMode);

      if (csvOutput)
        exp.outputCSVData(std::cout);
    }

  return EXIT_SUCCESS;
}
