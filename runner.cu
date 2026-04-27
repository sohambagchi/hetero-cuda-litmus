// runner.cu — Host-side runner for heterogeneous CPU-GPU litmus testing
#include <iostream>
#include <set>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cerrno>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "litmus_het.cuh"
#include "memory_backends.h"
#include "cpu_functions.h"

// =============================================================================
// Debug / error helpers
// =============================================================================

#ifdef HET_DEBUG
#define HET_DEBUG_ENABLED 1
#define HET_DEBUG_LOG(...) do { \
  fprintf(stderr, "[HET_DEBUG][runner] " __VA_ARGS__); \
  fprintf(stderr, "\n"); \
} while(0)
#else
#define HET_DEBUG_ENABLED 0
#define HET_DEBUG_LOG(...) do { } while(0)
#endif

#define RUNNER_FAIL(...) do { \
  fprintf(stderr, "[HET_ERROR][runner] " __VA_ARGS__); \
  fprintf(stderr, "\n"); \
  throw std::runtime_error("runner failure"); \
} while(0)

#define CUDA_CHECK(call) do { \
  cudaError_t _cuda_err = (call); \
  if (_cuda_err != cudaSuccess) { \
    fprintf(stderr, "[HET_ERROR][runner] CUDA call failed at %s:%d: %s returned %s\n", \
            __FILE__, __LINE__, #call, cudaGetErrorString(_cuda_err)); \
    throw std::runtime_error("cuda failure"); \
  } \
} while(0)

#define CUDA_KERNEL_CHECK(kernel_name) do { \
  cudaError_t _cuda_err = cudaGetLastError(); \
  if (_cuda_err != cudaSuccess) { \
    fprintf(stderr, "[HET_ERROR][runner] Kernel launch failed for %s at %s:%d: %s\n", \
            kernel_name, __FILE__, __LINE__, cudaGetErrorString(_cuda_err)); \
    throw std::runtime_error("kernel launch failure"); \
  } \
} while(0)

// =============================================================================
// Host-side parameter structs
// =============================================================================

typedef struct {
  int numMemLocations;
  int permuteLocation;
} TestParams;

typedef struct {
  // All 18 cuda-litmus params
  int testIterations;
  int testingWorkgroups;
  int maxWorkgroups;
  int workgroupSize;
  int shufflePct;
  int barrierPct;
  int stressLineSize;
  int stressTargetLines;
  int scratchMemorySize;
  int memStride;
  int memStressPct;
  int memStressIterations;
  int memStressPattern;
  int preStressPct;
  int preStressIterations;
  int preStressPattern;
  int stressAssignmentStrategy;
  int permuteThread;
  // Het-specific params
  int cpuStressThreads;         // 0 = auto (hardware_concurrency / 2)
  int cpuPreStressPct;          // 0-100
  int cpuPreStressIterations;   // 0-128
  int cpuPreStressPattern;      // 0-3
  int barrierSpinLimit;         // default 4096
} StressParams;

namespace {

const char* bool_to_string(bool value) {
  return value ? "true" : "false";
}

void validate_test_params(const TestParams& config, const char* filename) {
  if (config.numMemLocations <= 0) {
    RUNNER_FAIL("Invalid numMemLocations=%d parsed from %s", config.numMemLocations, filename);
  }
  if (config.permuteLocation <= 0) {
    RUNNER_FAIL("Invalid permuteLocation=%d parsed from %s", config.permuteLocation, filename);
  }
}

void validate_stress_params(const StressParams& config, const char* filename) {
  if (config.testIterations <= 0) {
    RUNNER_FAIL("Invalid testIterations=%d parsed from %s", config.testIterations, filename);
  }
  if (config.testingWorkgroups <= 0) {
    RUNNER_FAIL("Invalid testingWorkgroups=%d parsed from %s", config.testingWorkgroups, filename);
  }
  if (config.maxWorkgroups < config.testingWorkgroups) {
    RUNNER_FAIL("Invalid workgroup range in %s: maxWorkgroups=%d < testingWorkgroups=%d",
                filename, config.maxWorkgroups, config.testingWorkgroups);
  }
  if (config.workgroupSize <= 0) {
    RUNNER_FAIL("Invalid workgroupSize=%d parsed from %s", config.workgroupSize, filename);
  }
  if (config.stressLineSize <= 0 || config.stressTargetLines <= 0 || config.scratchMemorySize <= 0) {
    RUNNER_FAIL("Invalid stress memory config in %s: lineSize=%d targetLines=%d scratchMemorySize=%d",
                filename, config.stressLineSize, config.stressTargetLines, config.scratchMemorySize);
  }
  if (config.scratchMemorySize < config.stressLineSize * config.stressTargetLines) {
    RUNNER_FAIL("scratchMemorySize=%d too small for lineSize=%d targetLines=%d in %s",
                config.scratchMemorySize, config.stressLineSize, config.stressTargetLines, filename);
  }
  if (config.memStride <= 0) {
    RUNNER_FAIL("Invalid memStride=%d parsed from %s", config.memStride, filename);
  }
  if (config.cpuStressThreads < 0) {
    RUNNER_FAIL("Invalid cpuStressThreads=%d parsed from %s", config.cpuStressThreads, filename);
  }
  if (config.barrierSpinLimit <= 0) {
    RUNNER_FAIL("Invalid barrierSpinLimit=%d parsed from %s", config.barrierSpinLimit, filename);
  }
}

void log_test_params(const TestParams& config, const char* filename) {
  HET_DEBUG_LOG("Loaded test params from %s: numMemLocations=%d permuteLocation=%d",
                filename, config.numMemLocations, config.permuteLocation);
}

void log_stress_params(const StressParams& config, const char* filename) {
  HET_DEBUG_LOG(
    "Loaded stress params from %s: testIterations=%d testingWorkgroups=%d maxWorkgroups=%d workgroupSize=%d shufflePct=%d barrierPct=%d stressLineSize=%d stressTargetLines=%d scratchMemorySize=%d memStride=%d memStressPct=%d memStressIterations=%d memStressPattern=%d preStressPct=%d preStressIterations=%d preStressPattern=%d stressAssignmentStrategy=%d permuteThread=%d cpuStressThreads=%d cpuPreStressPct=%d cpuPreStressIterations=%d cpuPreStressPattern=%d barrierSpinLimit=%d",
    filename,
    config.testIterations,
    config.testingWorkgroups,
    config.maxWorkgroups,
    config.workgroupSize,
    config.shufflePct,
    config.barrierPct,
    config.stressLineSize,
    config.stressTargetLines,
    config.scratchMemorySize,
    config.memStride,
    config.memStressPct,
    config.memStressIterations,
    config.memStressPattern,
    config.preStressPct,
    config.preStressIterations,
    config.preStressPattern,
    config.stressAssignmentStrategy,
    config.permuteThread,
    config.cpuStressThreads,
    config.cpuPreStressPct,
    config.cpuPreStressIterations,
    config.cpuPreStressPattern,
    config.barrierSpinLimit);
}

void log_kernel_params(const KernelParams& config) {
  HET_DEBUG_LOG(
    "Kernel params: barrier=%s mem_stress=%s mem_stress_iterations=%d mem_stress_pattern=%d pre_stress=%s pre_stress_iterations=%d pre_stress_pattern=%d permute_thread=%d permute_location=%d testing_workgroups=%d mem_stride=%d mem_offset=%d total_instances=%d barrier_spin_limit=%d",
    bool_to_string(config.barrier),
    bool_to_string(config.mem_stress),
    config.mem_stress_iterations,
    config.mem_stress_pattern,
    bool_to_string(config.pre_stress),
    config.pre_stress_iterations,
    config.pre_stress_pattern,
    config.permute_thread,
    config.permute_location,
    config.testing_workgroups,
    config.mem_stride,
    config.mem_offset,
    config.total_instances,
    config.barrier_spin_limit);
}

}  // namespace

// =============================================================================
// Parameter file parsers
// =============================================================================

int parseTestParamsFile(const char* filename, TestParams* config) {
  config->numMemLocations = 0;
  config->permuteLocation = 0;

  HET_DEBUG_LOG("Opening test params file: %s", filename);
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "[HET_ERROR][runner] Failed to open test params file '%s': %s\n",
            filename, strerror(errno));
    return -1;
  }
  char line[256];
  int line_number = 0;
  while (fgets(line, sizeof(line), file)) {
    line_number++;
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
      if (strcmp(key, "numMemLocations") == 0) config->numMemLocations = value;
      else if (strcmp(key, "permuteLocation") == 0) config->permuteLocation = value;
      else HET_DEBUG_LOG("Ignoring unknown test param key '%s' on line %d in %s", key, line_number, filename);
    } else {
      std::string raw_line(line);
      raw_line.erase(std::remove(raw_line.begin(), raw_line.end(), '\n'), raw_line.end());
      if (!raw_line.empty()) {
        HET_DEBUG_LOG("Skipping unparsable test param line %d in %s: %s", line_number, filename, raw_line.c_str());
      }
    }
  }
  fclose(file);
  validate_test_params(*config, filename);
  log_test_params(*config, filename);
  return 0;
}

int parseStressParamsFile(const char* filename, StressParams* config) {
  memset(config, 0, sizeof(*config));
  // Set defaults for het-specific params
  config->cpuStressThreads = 0;
  config->cpuPreStressPct = 50;
  config->cpuPreStressIterations = 64;
  config->cpuPreStressPattern = 0;
  config->barrierSpinLimit = 4096;

  HET_DEBUG_LOG("Opening stress params file: %s", filename);
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "[HET_ERROR][runner] Failed to open stress params file '%s': %s\n",
            filename, strerror(errno));
    return -1;
  }
  char line[256];
  int line_number = 0;
  while (fgets(line, sizeof(line), file)) {
    line_number++;
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
      if (strcmp(key, "testIterations") == 0) config->testIterations = value;
      else if (strcmp(key, "testingWorkgroups") == 0) config->testingWorkgroups = value;
      else if (strcmp(key, "maxWorkgroups") == 0) config->maxWorkgroups = value;
      else if (strcmp(key, "workgroupSize") == 0) config->workgroupSize = value;
      else if (strcmp(key, "shufflePct") == 0) config->shufflePct = value;
      else if (strcmp(key, "barrierPct") == 0) config->barrierPct = value;
      else if (strcmp(key, "stressLineSize") == 0) config->stressLineSize = value;
      else if (strcmp(key, "stressTargetLines") == 0) config->stressTargetLines = value;
      else if (strcmp(key, "scratchMemorySize") == 0) config->scratchMemorySize = value;
      else if (strcmp(key, "memStride") == 0) config->memStride = value;
      else if (strcmp(key, "memStressPct") == 0) config->memStressPct = value;
      else if (strcmp(key, "memStressIterations") == 0) config->memStressIterations = value;
      else if (strcmp(key, "memStressPattern") == 0) config->memStressPattern = value;
      else if (strcmp(key, "preStressPct") == 0) config->preStressPct = value;
      else if (strcmp(key, "preStressIterations") == 0) config->preStressIterations = value;
      else if (strcmp(key, "preStressPattern") == 0) config->preStressPattern = value;
      else if (strcmp(key, "stressAssignmentStrategy") == 0) config->stressAssignmentStrategy = value;
      else if (strcmp(key, "permuteThread") == 0) config->permuteThread = value;
      // Het-specific params
      else if (strcmp(key, "cpuStressThreads") == 0) config->cpuStressThreads = value;
      else if (strcmp(key, "cpuPreStressPct") == 0) config->cpuPreStressPct = value;
      else if (strcmp(key, "cpuPreStressIterations") == 0) config->cpuPreStressIterations = value;
      else if (strcmp(key, "cpuPreStressPattern") == 0) config->cpuPreStressPattern = value;
      else if (strcmp(key, "barrierSpinLimit") == 0) config->barrierSpinLimit = value;
      else HET_DEBUG_LOG("Ignoring unknown stress param key '%s' on line %d in %s", key, line_number, filename);
    } else {
      std::string raw_line(line);
      raw_line.erase(std::remove(raw_line.begin(), raw_line.end(), '\n'), raw_line.end());
      if (!raw_line.empty()) {
        HET_DEBUG_LOG("Skipping unparsable stress param line %d in %s: %s", line_number, filename, raw_line.c_str());
      }
    }
  }
  fclose(file);
  validate_stress_params(*config, filename);
  log_stress_params(*config, filename);
  return 0;
}

// =============================================================================
// Helper functions (from cuda-litmus, unchanged)
// =============================================================================

int setBetween(int min, int max) {
  if (min == max) return min;
  int size = rand() % (max - min + 1);
  return min + size;
}

bool percentageCheck(int percentage) {
  return rand() % 100 < percentage;
}

void setShuffledWorkgroups(uint* h_shuffledWorkgroups, int numWorkgroups, int shufflePct) {
  HET_DEBUG_LOG("Preparing shuffled workgroups: numWorkgroups=%d shufflePct=%d", numWorkgroups, shufflePct);
  for (int i = 0; i < numWorkgroups; i++) {
    h_shuffledWorkgroups[i] = i;
  }
  if (percentageCheck(shufflePct)) {
    for (int i = numWorkgroups - 1; i > 0; i--) {
      int swap = rand() % (i + 1);
      int temp = h_shuffledWorkgroups[i];
      h_shuffledWorkgroups[i] = h_shuffledWorkgroups[swap];
      h_shuffledWorkgroups[swap] = temp;
    }
  }
}

void setScratchLocations(uint* h_locations, int numWorkgroups, StressParams params) {
  std::set<int> usedRegions;
  int numRegions = params.scratchMemorySize / params.stressLineSize;
  if (numRegions <= 0) {
    RUNNER_FAIL("Computed invalid numRegions=%d from scratchMemorySize=%d and stressLineSize=%d",
                numRegions, params.scratchMemorySize, params.stressLineSize);
  }
  if (params.stressTargetLines > numRegions) {
    RUNNER_FAIL("stressTargetLines=%d exceeds available scratch regions=%d", params.stressTargetLines, numRegions);
  }

  HET_DEBUG_LOG("Preparing scratch locations: numWorkgroups=%d numRegions=%d stressTargetLines=%d strategy=%d",
                numWorkgroups, numRegions, params.stressTargetLines, params.stressAssignmentStrategy);
  for (int i = 0; i < params.stressTargetLines; i++) {
    int region = rand() % numRegions;
    while (usedRegions.count(region))
      region = rand() % numRegions;
    usedRegions.insert(region);
    int locInRegion = rand() % params.stressLineSize;
    switch (params.stressAssignmentStrategy) {
    case 0:
      for (int j = i; j < numWorkgroups; j += params.stressTargetLines) {
        h_locations[j] = (region * params.stressLineSize) + locInRegion;
      }
      break;
    case 1: {
      int workgroupsPerLocation = numWorkgroups / params.stressTargetLines;
      for (int j = 0; j < workgroupsPerLocation; j++) {
        h_locations[i * workgroupsPerLocation + j] = (region * params.stressLineSize) + locInRegion;
      }
      if (i == params.stressTargetLines - 1 && numWorkgroups % params.stressTargetLines != 0) {
        for (int j = 0; j < numWorkgroups % params.stressTargetLines; j++) {
          h_locations[numWorkgroups - j - 1] = (region * params.stressLineSize) + locInRegion;
        }
      }
      break;
    }
    }
  }
}

void setDynamicKernelParams(KernelParams* h_kernelParams, StressParams params) {
  h_kernelParams->barrier = percentageCheck(params.barrierPct);
  h_kernelParams->mem_stress = percentageCheck(params.memStressPct);
  h_kernelParams->pre_stress = percentageCheck(params.preStressPct);
  log_kernel_params(*h_kernelParams);
}

void setStaticKernelParams(KernelParams* h_kernelParams, StressParams stressParams, TestParams testParams) {
  h_kernelParams->mem_stress_iterations = stressParams.memStressIterations;
  h_kernelParams->mem_stress_pattern = stressParams.memStressPattern;
  h_kernelParams->pre_stress_iterations = stressParams.preStressIterations;
  h_kernelParams->pre_stress_pattern = stressParams.preStressPattern;
  h_kernelParams->permute_thread = stressParams.permuteThread;
  h_kernelParams->permute_location = testParams.permuteLocation;
  h_kernelParams->testing_workgroups = stressParams.testingWorkgroups;
  h_kernelParams->mem_stride = stressParams.memStride;
  h_kernelParams->mem_offset = stressParams.memStride;
  // Het-specific
  h_kernelParams->total_instances = stressParams.workgroupSize * stressParams.testingWorkgroups;
  h_kernelParams->barrier_spin_limit = stressParams.barrierSpinLimit;
}

int total_behaviors(TestResults* results) {
  return results->res0 + results->res1 + results->res2 + results->res3 +
    results->res4 + results->res5 + results->res6 + results->res7 +
    results->res8 + results->res9 + results->res10 + results->res11 +
    results->res12 + results->res13 + results->res14 + results->res15 +
    results->weak + results->other;
}

// =============================================================================
// Main run function
// =============================================================================

void run(StressParams stressParams, TestParams testParams, bool print_results) {
  int testingThreads = stressParams.workgroupSize * stressParams.testingWorkgroups;
  int numCpuWorkers = std::thread::hardware_concurrency();
  if (numCpuWorkers <= 0) numCpuWorkers = 4;  // fallback
  int instancesPerCpuThread = (testingThreads + numCpuWorkers - 1) / numCpuWorkers;

  HET_DEBUG_LOG("Run start: testingThreads=%d numCpuWorkers=%d instancesPerCpuThread=%d print_results=%s debug_build=%s",
                testingThreads,
                numCpuWorkers,
                instancesPerCpuThread,
                bool_to_string(print_results),
                bool_to_string(HET_DEBUG_ENABLED != 0));

  // -------------------------------------------------------------------------
  // 1. Allocate shared memory (CPU-GPU accessible via het_malloc)
  // -------------------------------------------------------------------------
  int testLocSize = testingThreads * testParams.numMemLocations * stressParams.memStride * sizeof(uint);

  het_atomic_uint *h_testLocations, *d_testLocations;
  het_malloc((void**)&h_testLocations, (void**)&d_testLocations, testLocSize);

  int readResultsSize = sizeof(ReadResults) * testingThreads;
  ReadResults *h_readResults, *d_readResults;
  het_malloc((void**)&h_readResults, (void**)&d_readResults, readResultsSize);

  int hetBarriersSize = sizeof(het_barrier_t) * testingThreads;
  het_barrier_t *h_hetBarriers, *d_hetBarriers;
  het_malloc((void**)&h_hetBarriers, (void**)&d_hetBarriers, hetBarriersSize);

  TestResults *h_testResults, *d_testResults;
  het_malloc((void**)&h_testResults, (void**)&d_testResults, sizeof(TestResults));

  TestInstance *h_testInstances, *d_testInstances;
  het_malloc((void**)&h_testInstances, (void**)&d_testInstances, sizeof(TestInstance) * testingThreads);

  int weakSize = sizeof(bool) * testingThreads;
  bool *h_weak, *d_weak;
  het_malloc((void**)&h_weak, (void**)&d_weak, weakSize);

  // -------------------------------------------------------------------------
  // 2. Allocate GPU-only memory (scratchpad, shuffled workgroups, etc.)
  // -------------------------------------------------------------------------
  int shuffledWorkgroupsSize = stressParams.maxWorkgroups * sizeof(uint);
  uint* h_shuffledWorkgroups = (uint*)malloc(shuffledWorkgroupsSize);
  if (h_shuffledWorkgroups == nullptr) {
    RUNNER_FAIL("malloc failed for h_shuffledWorkgroups (%d bytes)", shuffledWorkgroupsSize);
  }
  uint* d_shuffledWorkgroups;
  CUDA_CHECK(cudaMalloc(&d_shuffledWorkgroups, shuffledWorkgroupsSize));

  cuda::atomic<uint, cuda::thread_scope_device>* d_gpuBarrier;
  CUDA_CHECK(cudaMalloc(&d_gpuBarrier, sizeof(cuda::atomic<uint, cuda::thread_scope_device>)));

  int scratchpadSize = stressParams.scratchMemorySize * sizeof(uint);
  uint* d_scratchpad;
  CUDA_CHECK(cudaMalloc(&d_scratchpad, scratchpadSize));

  int scratchLocationsSize = stressParams.maxWorkgroups * sizeof(uint);
  uint* h_scratchLocations = (uint*)malloc(scratchLocationsSize);
  if (h_scratchLocations == nullptr) {
    RUNNER_FAIL("malloc failed for h_scratchLocations (%d bytes)", scratchLocationsSize);
  }
  uint* d_scratchLocations;
  CUDA_CHECK(cudaMalloc(&d_scratchLocations, scratchLocationsSize));

  KernelParams* h_kernelParams = (KernelParams*)malloc(sizeof(KernelParams));
  if (h_kernelParams == nullptr) {
    RUNNER_FAIL("malloc failed for h_kernelParams (%zu bytes)", sizeof(KernelParams));
  }
  KernelParams* d_kernelParams;
  CUDA_CHECK(cudaMalloc(&d_kernelParams, sizeof(KernelParams)));
  setStaticKernelParams(h_kernelParams, stressParams, testParams);
  HET_DEBUG_LOG("Static kernel params prepared");
  log_kernel_params(*h_kernelParams);

  // -------------------------------------------------------------------------
  // 3. CPU stress infrastructure
  // -------------------------------------------------------------------------
  int numStressThreads = stressParams.cpuStressThreads > 0
    ? stressParams.cpuStressThreads : std::max(1, numCpuWorkers / 2);
  int stressArraySize = 1024;
  volatile uint* stressArray = (volatile uint*)malloc(stressArraySize * sizeof(uint));
  if (stressArray == nullptr) {
    RUNNER_FAIL("malloc failed for CPU stress array (%zu bytes)", stressArraySize * sizeof(uint));
  }
  memset((void*)stressArray, 0, stressArraySize * sizeof(uint));
  stressArray[0] = 1;
  stressArray[stressArraySize - 1] = 1;
  volatile bool stopStress = false;

  HET_DEBUG_LOG("CPU stress setup: numStressThreads=%d stressArraySize=%d", numStressThreads, stressArraySize);

  // Per-CPU-thread scratchpads for inline pre-stress
  std::vector<volatile uint*> cpuScratchpads(numCpuWorkers);
  for (int i = 0; i < numCpuWorkers; i++) {
    cpuScratchpads[i] = (volatile uint*)malloc(64 * sizeof(uint));
    if (cpuScratchpads[i] == nullptr) {
      RUNNER_FAIL("malloc failed for CPU scratchpad %d (%zu bytes)", i, 64 * sizeof(uint));
    }
    memset((void*)cpuScratchpads[i], 0, 64 * sizeof(uint));
  }

  // Start background CPU stress threads
  std::vector<std::thread> bgStressThreads;
  for (int i = 0; i < numStressThreads; i++) {
    bgStressThreads.emplace_back(cpu_memory_stress_thread, stressArray, stressArraySize, &stopStress);
  }

  // -------------------------------------------------------------------------
  // 4. Main iteration loop
  // -------------------------------------------------------------------------
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int weakBehaviors = 0;
  int totalBehaviors = 0;

  for (int iter = 0; iter < stressParams.testIterations; iter++) {
    int numWorkgroups = setBetween(stressParams.testingWorkgroups, stressParams.maxWorkgroups);
    HET_DEBUG_LOG("Iteration %d/%d start: numWorkgroups=%d", iter + 1, stressParams.testIterations, numWorkgroups);

    // Clear shared memory
    het_memset(h_testLocations, 0, testLocSize);
    het_memset(h_readResults, 0, readResultsSize);
    het_memset(h_hetBarriers, 0, hetBarriersSize);
    het_memset(h_testResults, 0, sizeof(TestResults));
    het_memset(h_testInstances, 0, sizeof(TestInstance) * testingThreads);
    het_memset(h_weak, 0, weakSize);

    // Clear GPU-only memory
    CUDA_CHECK(cudaMemset(d_gpuBarrier, 0, sizeof(cuda::atomic<uint, cuda::thread_scope_device>)));
    CUDA_CHECK(cudaMemset(d_scratchpad, 0, scratchpadSize));

    // Randomize per-iteration params
    setShuffledWorkgroups(h_shuffledWorkgroups, numWorkgroups, stressParams.shufflePct);
    CUDA_CHECK(cudaMemcpy(d_shuffledWorkgroups, h_shuffledWorkgroups, shuffledWorkgroupsSize, cudaMemcpyHostToDevice));
    setScratchLocations(h_scratchLocations, numWorkgroups, stressParams);
    CUDA_CHECK(cudaMemcpy(d_scratchLocations, h_scratchLocations, scratchLocationsSize, cudaMemcpyHostToDevice));
    setDynamicKernelParams(h_kernelParams, stressParams);
    CUDA_CHECK(cudaMemcpy(d_kernelParams, h_kernelParams, sizeof(KernelParams), cudaMemcpyHostToDevice));

    // ----- 4a. Launch GPU kernel (asynchronous) -----
    HET_DEBUG_LOG("Launching litmus_test: blocks=%d threads_per_block=%d", numWorkgroups, stressParams.workgroupSize);
    litmus_test<<<numWorkgroups, stressParams.workgroupSize>>>(
      d_testLocations, d_readResults, d_shuffledWorkgroups,
      d_gpuBarrier, d_scratchpad, d_scratchLocations,
      d_kernelParams, d_testInstances, d_hetBarriers);
    CUDA_KERNEL_CHECK("litmus_test");

    // ----- 4b. Launch CPU test threads -----
    bool doCpuPreStress = percentageCheck(stressParams.cpuPreStressPct);
    HET_DEBUG_LOG("Spawning CPU test threads: doCpuPreStress=%s", bool_to_string(doCpuPreStress));
    std::vector<std::thread> cpuTestThreads;
    for (int t = 0; t < numCpuWorkers; t++) {
      int startInst = t * instancesPerCpuThread;
      int endInst = std::min(startInst + instancesPerCpuThread, testingThreads);
      if (startInst < testingThreads) {
        HET_DEBUG_LOG("CPU worker %d handles instances [%d, %d)", t, startInst, endInst);
        cpuTestThreads.emplace_back(cpu_test_thread,
          t, startInst, endInst,
          h_testLocations, h_readResults, h_kernelParams,
          h_hetBarriers, cpuScratchpads[t],
          doCpuPreStress, stressParams.cpuPreStressIterations,
          stressParams.cpuPreStressPattern);
      }
    }

    // ----- 4c. Join CPU threads + sync GPU -----
    HET_DEBUG_LOG("Waiting for %zu CPU test threads", cpuTestThreads.size());
    for (auto& t : cpuTestThreads) t.join();
    HET_DEBUG_LOG("CPU threads joined; synchronizing after litmus_test");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----- 4d. Run check_results kernel -----
    HET_DEBUG_LOG("Launching check_results: blocks=%d threads_per_block=%d", stressParams.testingWorkgroups, stressParams.workgroupSize);
    check_results<<<stressParams.testingWorkgroups, stressParams.workgroupSize>>>(
      d_testLocations, d_readResults, d_testResults, d_kernelParams, d_weak);
    CUDA_KERNEL_CHECK("check_results");
    HET_DEBUG_LOG("Synchronizing after check_results");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----- 4e. Collect results -----
    if (print_results) {
      std::cout << "Iteration " << iter << "\n";
      for (int i = 0; i < testingThreads; i++) {
        if (h_weak[i]) {
          std::cout << "Weak result " << i << "\n";
          std::cout << "  t0: " << h_testInstances[i].t0;
          std::cout << "  t1: " << h_testInstances[i].t1;
          std::cout << "  t2: " << h_testInstances[i].t2;
          std::cout << "  t3: " << h_testInstances[i].t3 << "\n";
          std::cout << "  x: " << h_testInstances[i].x;
          std::cout << "  y: " << h_testInstances[i].y;
          std::cout << "  z: " << h_testInstances[i].z << "\n";
        }
      }
    }
    int iterationWeakBehaviors = host_check_results(h_testResults, print_results);
    int iterationTotalBehaviors = total_behaviors(h_testResults);
    weakBehaviors += iterationWeakBehaviors;
    totalBehaviors += iterationTotalBehaviors;
    HET_DEBUG_LOG("Iteration %d/%d complete: iteration_total=%d iteration_weak=%d cumulative_total=%d cumulative_weak=%d",
                  iter + 1,
                  stressParams.testIterations,
                  iterationTotalBehaviors,
                  iterationWeakBehaviors,
                  totalBehaviors,
                  weakBehaviors);
  }

  // -------------------------------------------------------------------------
  // 5. Report results
  // -------------------------------------------------------------------------
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
  std::cout << std::fixed << std::setprecision(0)
            << "Weak behavior rate: " << float(weakBehaviors) / duration.count()
            << " per second\n";
  std::cout << "Total behaviors: " << totalBehaviors << "\n";
  std::cout << "Number of weak behaviors: " << weakBehaviors << "\n";
  HET_DEBUG_LOG("Run complete: duration=%.6f totalBehaviors=%d weakBehaviors=%d",
                duration.count(), totalBehaviors, weakBehaviors);

  // -------------------------------------------------------------------------
  // 6. Cleanup
  // -------------------------------------------------------------------------
  stopStress = true;
  HET_DEBUG_LOG("Stopping %zu background CPU stress threads", bgStressThreads.size());
  for (auto& t : bgStressThreads) t.join();

  het_free(h_testLocations);
  het_free(h_readResults);
  het_free(h_hetBarriers);
  het_free(h_testResults);
  het_free(h_testInstances);
  het_free(h_weak);

  cudaFree(d_shuffledWorkgroups);
  free(h_shuffledWorkgroups);
  cudaFree(d_gpuBarrier);
  cudaFree(d_scratchpad);
  cudaFree(d_scratchLocations);
  free(h_scratchLocations);
  cudaFree(d_kernelParams);
  free(h_kernelParams);

  for (int i = 0; i < numCpuWorkers; i++) {
    free((void*)cpuScratchpads[i]);
  }
  free((void*)stressArray);
  HET_DEBUG_LOG("Cleanup complete");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
  char* stress_params_file = nullptr;
  char* test_params_file = nullptr;
  bool print_results = false;

  int c;
  while ((c = getopt(argc, argv, "xs:t:")) != -1)
    switch (c) {
    case 's':
      stress_params_file = optarg;
      break;
    case 't':
      test_params_file = optarg;
      break;
    case 'x':
      print_results = true;
      break;
    case '?':
      if (optopt == 's' || optopt == 't')
        std::cerr << "Option -" << (char)optopt << " requires an argument\n";
      else
        std::cerr << "Unknown option -" << (char)optopt << "\n";
      return 1;
    default:
      abort();
    }

  if (stress_params_file == nullptr) {
    std::cerr << "Stress param file (-s) must be set\n";
    return 1;
  }
  if (test_params_file == nullptr) {
    std::cerr << "Test param file (-t) must be set\n";
    return 1;
  }

  HET_DEBUG_LOG("Program start: argc=%d debug_build=%s", argc, bool_to_string(HET_DEBUG_ENABLED != 0));
  HET_DEBUG_LOG("CLI options: stress_params_file=%s test_params_file=%s print_results=%s",
                stress_params_file,
                test_params_file,
                bool_to_string(print_results));

  try {
    StressParams stressParams;
    if (parseStressParamsFile(stress_params_file, &stressParams) != 0) {
      return 1;
    }

    TestParams testParams;
    if (parseTestParamsFile(test_params_file, &testParams) != 0) {
      return 1;
    }

    run(stressParams, testParams, print_results);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[HET_ERROR][runner] Fatal error: " << ex.what() << "\n";
    return 1;
  }
}
