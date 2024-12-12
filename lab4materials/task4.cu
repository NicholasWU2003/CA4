/*
 * Skeleton code for use with Computer Architecture 2023 assignment 4,
 * LIACS, Leiden University.
 *
 * Task 2: tile composite. After solving task 2, your modified program
 * will be the template for tasks 3 and 4 (make copies!)
 *
 */

#include "image.h"
#include "experiment.hpp"

#include <iostream>
#include <vector>
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

  if (xx < x || xx >= width || yy < y || yy >= height)
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

__global__ void
op_tile_composite_noShared_batch(uint32_t *background, const uint32_t *tile,
                                 int bg_width, int bg_height,
                                 int tile_width, int tile_height,
                                 float tile_alpha, int nImages)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int total_height = bg_height * nImages;

    if (global_x >= bg_width || global_y >= total_height)
        return;

    int image_idx = global_y / bg_height;
    int local_y = global_y % bg_height;

    // Compute index into background array
    int pixel_idx = image_idx * (bg_width * bg_height) + local_y * bg_width + global_x;

    // Tile coordinates
    int tx = global_x % tile_width;
    int ty = (local_y) % tile_height;

    uint32_t bg_pix = background[pixel_idx];
    uint32_t tile_pix = tile[ty * tile_width + tx];

    rgba_t dst, src;
    RGBA_unpack(dst, bg_pix);
    RGBA_unpack(src, tile_pix);
    RGBA_mults(src, src, tile_alpha);
    RGBA_mults(dst, dst, 1.0f - tile_alpha);
    RGBA_add(dst, dst, src);
    RGBA_pack(background[pixel_idx], dst);
}


/* GPU kernel using shared memory to store the tile.
 * Here we assume tile_width = tile_height = 64.
 * Make sure your block size is (32,32) for optimal performance.
 */
__global__ void
op_tile_composite_shared_batch(uint32_t *background, const uint32_t *tile,
                               int bg_width, int bg_height,
                               int tile_width, int tile_height,
                               float tile_alpha, int nImages)
{
    __shared__ uint32_t shared_tile[64][64];

    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int total_height = bg_height * nImages;

    // Load tile into shared memory
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int total_pixels = tile_width * tile_height; // e.g., 64*64=4096
    int stride = blockDim.x * blockDim.y;        // 32*32=1024 if block size is (32,32)

    for (int i = local_idx; i < total_pixels; i += stride) {
        int ty = i / tile_width;
        int tx = i % tile_width;
        shared_tile[ty][tx] = tile[i];
    }

    __syncthreads();

    if (global_x >= bg_width || global_y >= total_height)
        return;

    int image_idx = global_y / bg_height;
    int local_y = global_y % bg_height;

    // Compute index into background array
    int pixel_idx = image_idx * (bg_width * bg_height) + local_y * bg_width + global_x;

    int tx = global_x % tile_width;
    int ty = local_y % tile_height;

    uint32_t background_px = background[pixel_idx];
    uint32_t tile_px = shared_tile[ty][tx];

    rgba_t dst, src;
    RGBA_unpack(dst, background_px);
    RGBA_unpack(src, tile_px);

    RGBA_mults(src, src, tile_alpha);
    RGBA_mults(dst, dst, 1.0f - tile_alpha);
    RGBA_add(dst, dst, src);

    RGBA_pack(background[pixel_idx], dst);
}



/* Returns elapsed time in msec with overlapping data transfers and compute */
static float
run_cuda_kernels(image_t *background[], const size_t nImages,
                 const image_t *tile,
                 uint32_t *d_background_buffer, uint32_t *d_tile_buffer,
                 cudaStream_t stream)
{
  // Get dimensions
  size_t background_width = background[0]->width;
  size_t background_height = background[0]->height;

  // Calculate number of pixels in one image and total pixels in the batch
  size_t image_pixels = background_width * background_height;
  size_t total_pixels = image_pixels * nImages;

  // Byte sizes
  size_t image_bytes = image_pixels * sizeof(uint32_t);
  size_t tile_width = tile->width;
  size_t tile_height = tile->height;
  size_t tile_pixels = tile_width * tile_height;
  size_t tile_size = tile_pixels * sizeof(uint32_t);

  // Asynchronously copy all images into the background buffer on the device
  for (size_t i = 0; i < nImages; i++) {
    CUDA_ASSERT(cudaMemcpyAsync(d_background_buffer + i * image_pixels,
                                background[i]->data,
                                image_bytes,
                                cudaMemcpyHostToDevice,
                                stream));
  }

  // Asynchronously copy the tile to the device buffer
  CUDA_ASSERT(cudaMemcpyAsync(d_tile_buffer, tile->data, tile_size,
                              cudaMemcpyHostToDevice, stream));

  // Define block and grid sizes
  int total_height = static_cast<int>(nImages) * static_cast<int>(background_height);
  dim3 block_size(32, 32); // Optimal block size for (32,32) threads
  dim3 grid_size((background_width + block_size.x - 1) / block_size.x,
                 (total_height + block_size.y - 1) / block_size.y);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_ASSERT(cudaEventCreate(&start));
  CUDA_ASSERT(cudaEventCreate(&stop));

  // Record the start event
  CUDA_ASSERT(cudaEventRecord(start, stream));

  // Launch batched kernel on the provided stream
  op_tile_composite_shared_batch<<<grid_size, block_size, 0, stream>>>(
    d_background_buffer,
    d_tile_buffer,
    static_cast<int>(background_width),
    static_cast<int>(background_height),
    static_cast<int>(tile_width),
    static_cast<int>(tile_height),
    0.2f,
    static_cast<int>(nImages)
  );

  // Check for kernel launch errors
  CUDA_ASSERT(cudaGetLastError());

  // Record the stop event
  CUDA_ASSERT(cudaEventRecord(stop, stream));

  // Wait for the kernel to finish
  CUDA_ASSERT(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float msec = 0.0f;
  CUDA_ASSERT(cudaEventElapsedTime(&msec, start, stop));

  // Asynchronously copy results back to host
  for (size_t i = 0; i < nImages; i++) {
    CUDA_ASSERT(cudaMemcpyAsync(background[i]->data,
                                d_background_buffer + i * image_pixels,
                                image_bytes,
                                cudaMemcpyDeviceToHost,
                                stream));
  }

  // Synchronize the stream to ensure all copies are done
  CUDA_ASSERT(cudaStreamSynchronize(stream));

  // Destroy CUDA events
  CUDA_ASSERT(cudaEventDestroy(start));
  CUDA_ASSERT(cudaEventDestroy(stop));

  return msec;
}



/* Runs a test by processing a single image and comparing the GPU result with the CPU result.
 * Allocates necessary device memory and CUDA streams, invokes run_cuda_kernels with the correct arguments,
 * and cleans up allocated resources.
 */
static void
run_test(const std::string &infilename, image_t *tile,
         const std::string &outfilename)
{
  std::cout << "Testing with " << infilename << " ...\n";

  // Load the input image
  image_t *background[1];
  background[0] = image_new_from_pngfile(infilename.c_str());
  if (!background[0]) {
    std::cerr << "Failed to load input image: " << infilename << "\n";
    return;
  }

  // Create a copy of the background image for CPU processing
  image_t *original = image_new_from_image(background[0]);
  if (!original) {
    std::cerr << "Failed to create a copy of the input image.\n";
    image_free(background[0]);
    return;
  }
  std::memcpy(original->data, background[0]->data,
              background[0]->rowstride * background[0]->height);

  // Run the CPU reference kernel
  op_tile_composite(original, tile, 0.2f);

  // Prepare for GPU processing
  size_t background_width = background[0]->width;
  size_t background_height = background[0]->height;
  size_t image_pixels = background_width * background_height;
  size_t image_bytes = image_pixels * sizeof(uint32_t);

  size_t tile_width = tile->width;
  size_t tile_height = tile->height;
  size_t tile_pixels = tile_width * tile_height;
  size_t tile_size = tile_pixels * sizeof(uint32_t);

  uint32_t *d_background_buffer = nullptr;
  uint32_t *d_tile_buffer = nullptr;

  // Allocate device memory for background and tile
  CUDA_ASSERT(cudaMalloc(&d_background_buffer, image_pixels * sizeof(uint32_t)));
  CUDA_ASSERT(cudaMalloc(&d_tile_buffer, tile_size));

  // Create a CUDA stream for asynchronous operations
  cudaStream_t stream;
  CUDA_ASSERT(cudaStreamCreate(&stream));

  // Run GPU kernels with asynchronous data transfers and compute
  float msec = run_cuda_kernels(background, 1, tile,
                                d_background_buffer, d_tile_buffer,
                                stream);

  // Compare the GPU result with the CPU result
  const int max_error = 64;
  int errors = image_compare(background[0], original, max_error);
  if (errors > 0)
    std::cerr << "Images do not match, " << errors
              << " errors detected (max=" << max_error << ").\n";
  else
    std::cerr << "Images match.\n";

  // Optionally save the GPU-processed image
  if (!outfilename.empty()) {
    image_save_as_pngfile(background[0], outfilename.c_str());
    std::cerr << "Wrote GPU result to " << outfilename << "\n";
  }

  // Clean up device memory and CUDA stream
  CUDA_ASSERT(cudaFree(d_background_buffer));
  CUDA_ASSERT(cudaFree(d_tile_buffer));
  CUDA_ASSERT(cudaStreamDestroy(stream));

  // Free host memory
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
               const std::vector<std::string> &outfilenames,
               uint32_t *d_background_buffer, uint32_t *d_tile_buffer,
               cudaStream_t stream)
{
  /* Load images */
  image_t *background[infilenames.size()];

  auto startTime = ExperimentTimer::now();
  for (size_t j = 0; j < infilenames.size(); ++j)
    {
      background[j] = image_new_from_pngfile(infilenames[j].c_str());
      if (!background[j])
        {
          std::cerr << "Failed to load image: " << infilenames[j] << "\n";
          // Free previously loaded images
          for (size_t k = 0; k < j; ++k)
            image_free(background[k]);
          return;
        }
    }
  auto endTime = ExperimentTimer::now();

  timer.setLoadTime(i, endTime, startTime);

  /* Run CUDA kernels with overlapping data transfer and compute */
  float msec = run_cuda_kernels(background, infilenames.size(), tile,
                                d_background_buffer, d_tile_buffer,
                                stream);
  timer.setComputeTime(i, msec / 1000.0f); // Convert ms to seconds

  /* Save results if desired and if applicable */
  if (!outfilenames.empty())
    {
      for (size_t j = 0; j < outfilenames.size(); ++j)
        image_save_as_pngfile(background[j], outfilenames[j].c_str());
    }

  /* Free host memory */
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

  // Determine the maximum batch size across all batches
  size_t maxBatchSize = 0;
  for (size_t i = 0; i < exp.getNBatches(); ++i) {
    size_t currentBatchSize = exp.getBatchSize(i);
    if (currentBatchSize > maxBatchSize) {
      maxBatchSize = currentBatchSize;
    }
  }

  // Preallocate device buffers for two streams (double-buffering)
  size_t background_width = 0;
  size_t background_height = 0;

  if (maxBatchSize > 0) {
    // Assume all images have the same dimensions; load the first image to get dimensions
    std::string first_image = indir + "/" + exp.getFrameFile(0, 0);
    image_t *temp = image_new_from_pngfile(first_image.c_str());
    if (!temp) {
      std::cerr << "Failed to load image for dimension info.\n";
      return;
    }
    background_width = temp->width;
    background_height = temp->height;
    image_free(temp);
  }

  size_t image_pixels = background_width * background_height;
  size_t total_pixels = image_pixels * maxBatchSize;
  size_t image_bytes = image_pixels * sizeof(uint32_t);
  size_t tile_width = tile->width;
  size_t tile_height = tile->height;
  size_t tile_pixels = tile_width * tile_height;
  size_t tile_size = tile_pixels * sizeof(uint32_t);

  // Allocate two sets of device buffers for double-buffering
  uint32_t *d_background_buffer[2];
  uint32_t *d_tile_buffer[2];
  CUDA_ASSERT(cudaMalloc(&d_background_buffer[0], total_pixels * sizeof(uint32_t)));
  CUDA_ASSERT(cudaMalloc(&d_background_buffer[1], total_pixels * sizeof(uint32_t)));
  CUDA_ASSERT(cudaMalloc(&d_tile_buffer[0], tile_size));
  CUDA_ASSERT(cudaMalloc(&d_tile_buffer[1], tile_size));

  // Create two CUDA streams for double-buffering
  cudaStream_t streams[2];
  CUDA_ASSERT(cudaStreamCreate(&streams[0]));
  CUDA_ASSERT(cudaStreamCreate(&streams[1]));

  // Initialize timing variables
  size_t currentStream = 0;

  for (size_t i = 0; i < exp.getNBatches(); i++)
    {
      size_t count = exp.getBatchSize(i);

      std::vector<std::string> infilenames;
      std::vector<std::string> outfilenames;

      for (size_t j = 0; j < count; j++)
        {
          infilenames.emplace_back(indir + "/" + exp.getFrameFile(i, j));
          if (!outdir.empty())
            outfilenames.emplace_back(outdir + "/" + exp.getFrameFile(i, j));
        }

      if (!silentMode)
        {
          if (count == 1)
            std::cout << "Processing " << infilenames[0] << " ...\n" << std::flush;
          else
            std::cout << "Processing " << infilenames[0] << " - "
                      << infilenames[count - 1] << "...\n" << std::flush;
        }

      // Assign current stream (alternating between 0 and 1)
      size_t streamIdx = currentStream % 2;
      cudaStream_t currentCudaStream = streams[streamIdx];

      // Prepare device buffers for the current stream
      uint32_t *current_d_background_buffer = d_background_buffer[streamIdx];
      uint32_t *current_d_tile_buffer = d_tile_buffer[streamIdx];

      // Run CUDA kernels asynchronously on the current stream
      process_images(i, timer, infilenames, tile, outfilenames,
                    current_d_background_buffer, current_d_tile_buffer,
                    currentCudaStream);

      // Move to the next stream
      currentStream++;
    }

  /* Note that the full timing of the experiment will include image
   * loading & saving time and memory transfers to and from the GPU.
   * The memory transfers are now overlapped with computation.
   */
  timer.end();

  /* Print statistics */
  if (!silentMode)
    {
      std::cout << "====\n";
      exp.printCPUUtilization(exp.back(), std::cout);
      exp.printStatistics(exp.back(), std::cout);
      std::cout << "====\n";
    }

  /* Clean up device memory and streams */
  CUDA_ASSERT(cudaFree(d_background_buffer[0]));
  CUDA_ASSERT(cudaFree(d_background_buffer[1]));
  CUDA_ASSERT(cudaFree(d_tile_buffer[0]));
  CUDA_ASSERT(cudaFree(d_tile_buffer[1]));
  CUDA_ASSERT(cudaStreamDestroy(streams[0]));
  CUDA_ASSERT(cudaStreamDestroy(streams[1]));
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
  const char *outdir = (argc > 2) ? argv[2] : "";

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
  if (strlen(outdir) > 0)
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

  /* Create a list of PNG files in given directory. */
  std::vector<std::string> framefiles;
  for (struct dirent *ent = readdir(indirp);
       ent != NULL; ent = readdir(indirp))
    {
      std::string filename(ent->d_name);
      if (has_png_extension(filename))
        framefiles.emplace_back(std::move(filename));
    }

  closedir(indirp);

  if (framefiles.empty())
    {
      std::cerr << "No PNG files found in directory '" << indir << "'.\n";
      return EXIT_FAILURE;
    }

  /* Load tile image */
  image_t *tile = image_new_from_pngfile(tilefile);
  if (!tile)
    {
      std::cerr << "Failed to load tile image: " << tilefile << "\n";
      return EXIT_FAILURE;
    }

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
      for (int i = 0; i < n_repeat; i++)
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
