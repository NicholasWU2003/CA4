/*
 * Skeleton code for use with Computer Architecture 2023 assignment 4,
 * LIACS, Leiden University.
 *
 * Helper classes to collect and summarize experiment results.
 */

#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <vector>
#include <algorithm>
#include <numeric>
#include <ostream>
#include <iomanip>

#include <mutex>
#include <chrono>

#include <cmath>
#include <sys/time.h>
#include <sys/resource.h>

#include <omp.h>


/* Stores results for a single measurement. This encompasses processing all
 * image (batches) in a single directory once. Load time and compute time are
 * measured and stored separately.
 */
struct Measurement
{
  public:
    Measurement(size_t nResults)
    {
      loadTimes.resize(nResults);
      computeTimes.resize(nResults);

#if defined (_OPENMP)
      maxThreads = omp_get_max_threads();
#else
      maxThreads = 1;
#endif
    }

    double getUserTime() const
    {
      return timeval_get_elapsed_s(endUsage.ru_utime, startUsage.ru_utime);
    }

    double getSystemTime() const
    {
      return timeval_get_elapsed_s(endUsage.ru_stime, startUsage.ru_stime);
    }

    static inline double timeval_get_elapsed_s(const struct timeval &endTime,
                                               const struct timeval &startTime)
      {
        struct timeval elapsedTime;
        timersub(&endTime, &startTime, &elapsedTime);

        return (double)elapsedTime.tv_sec +
            (double)elapsedTime.tv_usec / 1000000.0;
      }


    double realTime;
    struct rusage startUsage{};
    struct rusage endUsage{};

    int maxThreads;

    std::vector<double> loadTimes;
    std::vector<double> computeTimes;
};


/* Helper class to perform timing of the duration of the entire
 * experiment and to store the load and frame time of individual
 * frames in a Measurement.
 */
class ExperimentTimer
{
  public:
    using time_point = std::chrono::time_point<std::chrono::steady_clock>;

    ExperimentTimer(Measurement &measurement)
      : measurement(measurement)
    {
    }

    static time_point now()
    {
      return std::chrono::steady_clock::now();
    }

    void start()
    {
      getrusage(RUSAGE_SELF, &measurement.startUsage);
      startTime = now();
    }

    void end()
    {
      endTime = now();
      std::chrono::duration<double> diff = endTime - startTime;
      measurement.realTime = diff.count();

      getrusage(RUSAGE_SELF, &measurement.endUsage);
    }

    void setLoadTime(size_t i,
                     const time_point endTime,
                     const time_point startTime)
    {
      std::chrono::duration<double> diff = endTime - startTime;

      const std::lock_guard<std::mutex> lock(loadTimesMutex);
      measurement.loadTimes[i] = diff.count();
    }

    void setComputeTime(size_t i,
                        const time_point endTime,
                        const time_point startTime)
    {
      std::chrono::duration<double> diff = endTime - startTime;

      const std::lock_guard<std::mutex> lock(computeTimesMutex);
      measurement.computeTimes[i] = diff.count();
    }

    /* To store CUDA timings, in which case we use the CUDA timer
     * instead of std::chrono.
     */
    void setComputeTime(size_t i, const double duration)
    {
      const std::lock_guard<std::mutex> lock(computeTimesMutex);
      measurement.computeTimes[i] = duration;
    }

  private:
    Measurement &measurement;

    time_point startTime{};
    time_point endTime{};

    std::mutex loadTimesMutex;
    std::mutex computeTimesMutex;
};


/* Class to store results for an experiment consisting of multiple
 * measurements (multiple runs of processing all frames).
 */
class Experiment
{
  public:
    Experiment(std::vector<std::string> &framefiles,
               size_t batchSize, bool singleBatch)
      : framefiles(framefiles), batchSize(batchSize), singleBatch(singleBatch)
    { }

    Measurement &addMeasurement()
    {
      measurements.emplace_back(getNBatches());
      return back();
    }

    Measurement &back()
    {
      return measurements.back();
    }

    size_t getNFiles() const
    {
      return framefiles.size();
    }

    size_t getBatchSize() const
    {
      return batchSize;
    }

    /* Override that returns actual batch size for given batch number. */
    size_t getBatchSize(size_t batchNo) const
    {
      return std::min(getBatchSize(), getNFiles() - (batchNo * getBatchSize()));
    }

    size_t getNBatches() const
    {
      if (singleBatch)
        return 1;
      /* else */
      return std::ceil(getNFiles() / (float)getBatchSize());
    }

    const std::string &getFrameFile(size_t i, size_t j) const
    {
      return framefiles[i * getBatchSize() + j];
    }

    void outputCSVData(std::ostream &out)
    {
      /* Output measurements of load and compute time for the individual
       * image (batches).
       */
      out << "item";
      for (size_t i = 0; i < measurements.size(); ++i)
        out << ",load";
      for (size_t i = 0; i < measurements.size(); ++i)
        out << ",compute";
      out << "\n";

      std::vector<double> times;
      times.resize(measurements.size() * 2);
      for (size_t i = 0; i < getNBatches(); ++i)
        {
          if (batchSize == 1)
            out << framefiles[i];
          else
            out << "batch" << i;
          for (size_t j = 0; j < measurements.size(); ++j)
            {
              times[j] = measurements[j].loadTimes[i];
              times[j + measurements.size()] = measurements[j].computeTimes[i];
            }

          for (auto &t : times)
            out << "," << t;
          out << "\n";
        }

      out << "\n";

      /* Output overall statistics of each experiment */
      out << "real,user,system,n_cores,utilization\n";

      for (const auto &m : measurements)
        printCPUUtilization(m, out, true);
    }

    void printCPUUtilization(const Measurement &m, std::ostream &out,
                             bool csvOutput = false) const
    {
      double user = m.getUserTime();
      double system = m.getSystemTime();

      /* CPU utilization */
      double util = ((user + system) / m.realTime) * 100.0;

      if (csvOutput)
        out << std::setprecision(8)
            << m.realTime << "," << user
            << "," << system << "," << m.maxThreads
            << "," << util << "\n";
      else
        out << std::setprecision(8)
            << "total: "
            << "real " << m.realTime
            << "s, user " << user
            << "s, system " << system
            << "s  (" << util << " % CPU utilization, "
            << m.maxThreads << " cores allowed)\n";
    }

    void printStatistics(const Measurement &m, std::ostream &out) const
    {
      printStatistics(out, "load ", m.loadTimes);
      printStatistics(out, "compute", m.computeTimes);
    }

  private:
    std::vector<std::string> &framefiles;
    const size_t batchSize;
    const bool singleBatch;
    std::vector<Measurement> measurements;


    void printStatistics(std::ostream &out, const std::string &name,
                         const std::vector<double> &data) const
    {
      const auto minmax = std::minmax_element(std::begin(data), std::end(data));
      double min = *minmax.first;
      double max = *minmax.second;
      double sum = std::accumulate(std::begin(data), std::end(data), 0.0);

      /* stdev computation */
      double mean = sum / data.size();
      double sum_differences = 0.0;

      for (double value : data)
        sum_differences += (value - mean) * (value - mean);

      double stdev = std::sqrt(sum_differences / data.size());

      /* output */
      out << name << ": "
          << min << "s - " << max
          << "s, avg " << mean << "s +/- " << stdev << "\n";
    }
};


#endif /* __EXPERIMENT_H__ */
