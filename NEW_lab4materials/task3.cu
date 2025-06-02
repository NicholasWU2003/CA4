/*
 * Skeleton code for use with Computer Architecture 2023 assignment 4,
 * LIACS, Leiden University.
 *
 * Task 3: Batched Processing. 
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


/*
 * CPU kernels
 *
 * Leave these functions in place, they are required for the "test
 * mode" to work. Make a copy in order to convert to a CUDA kernel.
 *
 */


/* Tiles the @tile image on @background using alpha blending. For the tile
 * an alpha value of @tile_alpha is used.
 */
/* Do not remove this function, it is required for the "test mode" to work. */
void
op_tile_composite(image_t *background,
                  const image_t *tile, const float tile_alpha)
{
  for (int y = 0; y < background->height; y++)
    {
      for (int x = 0; x < background->width; x++)
        {
          int tx = x % tile->width;
          int ty = y % tile->height;

          rgba_t dst, src;
          RGBA_unpack(dst, *image_get_pixel(background, x, y));
          RGBA_unpack(src, *image_get_pixel(tile, tx, ty));
          RGBA_mults(src, src, tile_alpha);
          RGBA_mults(dst, dst, 1.f - tile_alpha);
          RGBA_add(dst, dst, src);
          RGBA_pack(*image_get_pixel(background, x, y), dst);
        }
    }
}



// Batched global memory kernel
__global__ void tile_global_batched(uint32_t **dst_batch, int bg_width, int bg_height, int bg_rowbytes,
                                    const uint32_t *tile, int tile_width, int tile_height, int tile_rowbytes,
                                    const float alpha, int batch_size)
{
    // Compute the (x, y) coordinates of the pixel this thread will process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Exit if the thread is outside the bounds of the background image
    if (x >= bg_width || y >= bg_height) return;

    // Calculate the number of uint32_t elements per row for background and tile
    const int bg_pitch   = bg_rowbytes   >> 2;      // rowstride in uint32_t words
    const int tile_pitch = tile_rowbytes >> 2;

    // Compute the corresponding tile coordinates (tx, ty) using modulo for tiling
    int tx = x % tile_width;
    int ty = y % tile_height;
    // Fetch the tile pixel at (tx, ty)
    uint32_t tile_pix = tile[ty * tile_pitch + tx];

    // Unpack tile pixel once
    rgba_t t;
    RGBA_unpack(t, tile_pix);
    RGBA_mults(t, t, alpha);  // Scale tile by alpha once

    // Process all images in the batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        uint32_t *dst = dst_batch[batch_idx];
        
        // Fetch the background pixel at (x, y)
        uint32_t bg_pix = dst[y * bg_pitch + x];

        // Unpack background pixel
        rgba_t b;
        RGBA_unpack(b, bg_pix);

        // Perform alpha blending
        rgba_t out;
        RGBA_mults(b, b, 1.f - alpha);   // Scale background by (1 - alpha)
        RGBA_add(out, b, t);             // Add the two results

        // Pack the blended result back into uint32_t and store it
        RGBA_pack(bg_pix, out);
        dst[y * bg_pitch + x] = bg_pix;
    }
}

// Batched shared memory kernel
__global__ void tile_shared_batched(uint32_t **dst_batch, int bg_width, int bg_height, int bg_rowbytes,
                                    const uint32_t *tile, int tile_width, int tile_height, int tile_rowbytes,
                                    const float alpha, int batch_size)
{
    // Shared memory for the tile (allocated dynamically)
    extern __shared__ uint32_t tile_shmem[];

    // Calculate the number of uint32_t elements per row for tile
    const int tile_pitch = tile_rowbytes >> 2;

    // Each thread loads one or more tile pixels into shared memory
    int tile_size = tile_width * tile_height;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;
    for (int i = thread_id; i < tile_size; i += block_threads) {
        int tx = i % tile_width;
        int ty = i / tile_width;
        tile_shmem[i] = tile[ty * tile_pitch + tx];
    }
    __syncthreads();

    // Compute the (x, y) coordinates of the pixel this thread will process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bg_width || y >= bg_height) return;

    // Calculate the number of uint32_t elements per row for background
    const int bg_pitch = bg_rowbytes >> 2;

    // Compute the corresponding tile coordinates (tx, ty) using modulo for tiling
    int tx = x % tile_width;
    int ty = y % tile_height;
    // Fetch the tile pixel from shared memory
    uint32_t tile_pix = tile_shmem[ty * tile_width + tx];

    // Unpack tile pixel once
    rgba_t t;
    RGBA_unpack(t, tile_pix);
    RGBA_mults(t, t, alpha);  // Scale tile by alpha once

    // Process all images in the batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        uint32_t *dst = dst_batch[batch_idx];
        
        // Fetch the background pixel at (x, y)
        uint32_t bg_pix = dst[y * bg_pitch + x];

        // Unpack background pixel
        rgba_t b;
        RGBA_unpack(b, bg_pix);

        // Perform alpha blending
        rgba_t out;
        RGBA_mults(b, b, 1.f - alpha);   // Scale background by (1 - alpha)
        RGBA_add(out, b, t);             // Add the two results

        // Pack the blended result back into uint32_t and store it
        RGBA_pack(bg_pix, out);
        dst[y * bg_pitch + x] = bg_pix;
    }
}

/* Returns elapsed time in msec */
static float
run_cuda_kernels(image_t *background[], const size_t nImages,
                 const image_t *tile)
{
  /* Allocate device memory for background images and tile */
  uint32_t *d_tile = nullptr;
  size_t bg_bytes = background[0]->rowstride * background[0]->height;
  size_t tile_bytes = tile->rowstride * tile->height;

  // Allocate memory for tile
  cudaMalloc(&d_tile, tile_bytes);
  cudaMemcpy(d_tile, tile->data, tile_bytes, cudaMemcpyHostToDevice);

  // For batched processing: allocate array of device pointers
  uint32_t **d_bg_array = nullptr;
  uint32_t **h_bg_pointers = new uint32_t*[nImages];
  
  // Allocate device memory for each background image
  for (size_t i = 0; i < nImages; i++) {
    cudaMalloc(&h_bg_pointers[i], bg_bytes);
    cudaMemcpy(h_bg_pointers[i], background[i]->data, bg_bytes, cudaMemcpyHostToDevice);
  }
  
  // Allocate device memory for the array of pointers
  cudaMalloc(&d_bg_array, nImages * sizeof(uint32_t*));
  cudaMemcpy(d_bg_array, h_bg_pointers, nImages * sizeof(uint32_t*), cudaMemcpyHostToDevice);

  /* Calculate block size and grid dimensions */
  dim3 block(16, 16); 
  int PPTx = 1;
  int PPTy = 1;
  size_t sharedMem = tile->width * tile->height * sizeof(uint32_t);

  dim3 grid((background[0]->width + (block.x * PPTx - 1)) / (block.x * PPTx),
            (background[0]->height + (block.y * PPTy - 1)) / (block.y * PPTy));

  /* Timing setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Start the timer */
  CUDA_ASSERT(cudaEventRecord(start));

  
    // For batch processing, use batched kernels
    // tile_global_batched<<<grid, block>>>(d_bg_array, background[0]->width, background[0]->height, background[0]->rowstride,
    //                                       d_tile, tile->width, tile->height, tile->rowstride, 0.2f, nImages);
    
    // batched shared memory kernel
    tile_shared_batched<<<grid, block, sharedMem>>>(d_bg_array, background[0]->width, background[0]->height, background[0]->rowstride,
                                                     d_tile, tile->width, tile->height, tile->rowstride, 0.2f, nImages);
  

  CUDA_ASSERT( cudaGetLastError() );

  /* Stop timer */
  CUDA_ASSERT(cudaEventRecord(stop));
  CUDA_ASSERT(cudaEventSynchronize(stop));

  float msec = 0;
  CUDA_ASSERT(cudaEventElapsedTime(&msec, start, stop));

  /* Copy results back */
  for (size_t i = 0; i < nImages; i++) {
    cudaMemcpy(background[i]->data, h_bg_pointers[i], bg_bytes, cudaMemcpyDeviceToHost);
    cudaFree(h_bg_pointers[i]);
  }

  /* Cleanup */
  cudaFree(d_bg_array);
  cudaFree(d_tile);
  delete[] h_bg_pointers;

  return msec;
}

static void
run_test(const std::string &infilename, image_t *tile,
         const std::string &outfilename)
{
  std::cout << "Testing with " << infilename << " ...\n";

  image_t *background[1];
  background[0] = image_new_from_pngfile(infilename.c_str());
  if (!background[0])
    return;

  /* Create a copy to be manipulated on CPU */
  image_t *original = image_new_from_image(background[0]);
  std::memcpy(original->data, background[0]->data,
              background[0]->rowstride * background[0]->height);

  /* Run CPU kernels */
  op_tile_composite(original, tile, 0.2f);

  /* Run GPU kernels */
  run_cuda_kernels(background, 1, tile);

  /* Compare the results */
  const int max_error = 64;
  int errors = image_compare(background[0], original, max_error);
  if (errors > 0)
    std::cerr << "Images do not match, " << errors
              << " errors detected (max=" << max_error << ").\n";
  else
    std::cerr << "Images match.\n";

  /* Write GPU result to PNG if requested */
  if (not outfilename.empty())
    {
      image_save_as_pngfile(background[0], outfilename.c_str());
      std::cerr << "Wrote GPU result to " << outfilename << "\n";
    }

  image_free(original);
  image_free(background[0]);
}

/* Process a single image, or set/batch of images.. Warning: does not detect
 * errors.
 */
static void
process_images(size_t i, ExperimentTimer &timer,
               const std::vector<std::string> &infilenames,
               const image_t *tile,
               const std::vector<std::string> &outfilenames)
{
  /* Load image */
  image_t *background[infilenames.size()];

  auto startTime = ExperimentTimer::now();
  for (size_t j = 0; j < infilenames.size(); ++j)
    {
      background[j] = image_new_from_pngfile(infilenames[j].c_str());
      if (!background[j])
        return;
    }
  auto endTime = ExperimentTimer::now();

  timer.setLoadTime(i, endTime, startTime);

  float msec = run_cuda_kernels(background, infilenames.size(), tile);
  timer.setComputeTime(i, msec / 1000.);

  /* Save results if desired and if applicable */
  if (not outfilenames.empty())
    {
      for (size_t j = 0; j < outfilenames.size(); ++j)
        image_save_as_pngfile(background[j], outfilenames[j].c_str());
    }

  for (size_t j = 0; j < infilenames.size(); ++j)
    image_free(background[j]);
}

/* Code to run a single experiment, depending on batchSize parameter. */
static void
run_experiment(Experiment &exp,
               const std::string &indir, const std::string &outdir,
               image_t *tile, bool silentMode)
{
  ExperimentTimer timer(exp.addMeasurement());

  timer.start();

  for (size_t i = 0; i < exp.getNBatches(); ++i)
    {
      size_t count = exp.getBatchSize(i);

      std::vector<std::string> infilenames;
      std::vector<std::string> outfilenames;

      for (size_t j = 0; j < count; ++j)
        {
          infilenames.emplace_back(indir + std::string("/") + exp.getFrameFile(i, j));
          if (not outdir.empty())
            outfilenames.emplace_back(outdir + std::string("/") + exp.getFrameFile(i, j));
        }

      // if (not silentMode)
      //   {
      //     if (count == 1)
      //       std::cout << "Processing " << infilenames[0] << " ...\n" << std::flush;
      //     else
      //       std::cout << "Processing " << infilenames[0] << " - "
      //                 << infilenames[count - 1] << "...\n" << std::flush;
      //   }

      process_images(i, timer, infilenames, tile, outfilenames);
    }

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

static bool
has_png_extension(const std::string &filename)
{
  if (filename.size() < 4)
    return false;

  return filename.substr(filename.size() - 4, 4) == ".png";
}

static void
show_help(const char *progName)
{
  std::cerr << "usage: " << progName << " [-t] [-s] [-c] [-r REPEAT] [-b BATCH_SIZE] [-1] <indir> <tilefile> [outdir]\n"
            << "\n  where <indir>, and [outdir] are directories containing PNG files."
            << "\n  <tilefile> is a PNG file."
            << "\n  [outdir] is an optional parameter.\n"
            << "\n  -t  test mode: only processes the first image found in <indir> and compares"
            << "\n      the GPU result to the CPU result. Optionally outputs the GPU result if"
            << "\n      outdir is specified."
            << "\n  -s  disables output of experiment summaries (silent mode)."
            << "\n  -c  outputs an overview of all experiment results in CSV format."
            << "\n  -r  configures the number of times the experiment is repeated."
            << "\n  -b  configures the batch size (defaults to 1)."
            << "\n  -1  stops the experiment after processing one batch (defaults to false).\n";
}

int
main(int argc, char **argv)
{
  char c;
  long int n_repeat = 1;
  long int batchSize = 1;
  bool singleBatch = false;
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

          case 'b':
            batchSize = std::strtol(optarg, NULL, 10);
            if (errno > 0 || batchSize == 0)
              {
                std::cerr << "Could not convert batchSize argument to integer.\n";
                exit(EXIT_FAILURE);
              }
            break;

          case '1':
            singleBatch = true;
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

  if (argc < 2)
    {
      show_help(progName);
      return EXIT_FAILURE;
    }

  const char *indir = argv[0];
  const char *tilefile = argv[1];
  const char *outdir = argv[2];

  /* Check and open directories */
  DIR *indirp = opendir(indir);
  if (indirp == NULL)
    {
      const char *err = strerror(errno);
      std::cerr << "error: could not open directory '" << indir
                << "': " << err << "\n";
      return EXIT_FAILURE;
    }

  DIR *outdirp = NULL;
  if (outdir)
    {
      outdirp = opendir(outdir);
      if (outdirp == NULL)
        {
          const char *err = strerror(errno);
          std::cerr << "error: could not open directory '" << outdir
              << "': " << err << "\n";
          closedir(indirp);
          return EXIT_FAILURE;
        }

      closedir(outdirp);
    }

  /* Createa a list pf PNG files in given directory. */
  std::vector<std::string> framefiles;
  for (struct dirent *ent = readdir(indirp);
       ent != NULL; ent = readdir(indirp))
    {
      std::string filename(ent->d_name);
      if (has_png_extension(filename))
        framefiles.emplace_back(std::move(filename));
    }

  closedir(indirp);

  /* Load tile image */
  image_t *tile = image_new_from_pngfile(tilefile);
  if (!tile)
    return EXIT_FAILURE;

  if (testMode)
    {
      /* Test mode: process the first image found in "indir" and
       * compare the GPU result to the CPU result.
       */
      std::string infilename = std::string(indir) + std::string("/") + framefiles[0];
      std::string outfilename = outdir ? std::string(outdir) + std::string("/") + framefiles[0] : std::string();

      run_test(infilename, tile, outfilename);
    }
  else
    {
      /* Run experiment the requested number of times. */
      Experiment exp(framefiles, batchSize, singleBatch);
      for (int i = 0; i < n_repeat; ++i)
        run_experiment(exp, std::string(indir),
                       outdir ? std::string(outdir) : std::string(),
                       tile, silentMode);

      if (csvOutput)
        exp.outputCSVData(std::cout);
    }

  /* Clean up */
  image_free(tile);

  return EXIT_SUCCESS;
} 