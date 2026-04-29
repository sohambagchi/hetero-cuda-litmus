// ismm_runner.cu — targeted ISMM runner for SB and IRIW
#include <chrono>
#include <cstring>
#include <ctime>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include "cpu_functions.h"
#include "functions.cu"
#include "litmus_het.cuh"
#include "memory_backends.h"

#define CUDA_CHECK(call) do { \
  cudaError_t _cuda_err = (call); \
  if (_cuda_err != cudaSuccess) { \
    std::cerr << "CUDA call failed: " << #call << " -> " \
              << cudaGetErrorString(_cuda_err) << "\n"; \
    throw std::runtime_error("cuda failure"); \
  } \
} while (0)

struct TestParams {
  int numMemLocations;
  int permuteLocation;
};

struct StressParams {
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
  int cpuStressThreads;
  int cpuPreStressPct;
  int cpuPreStressIterations;
  int cpuPreStressPattern;
  int barrierSpinLimit;
};

enum TestKind {
  TEST_SB = 0,
  TEST_IRIW = 1,
};

enum DomainKind {
  DOMAIN_CPU = 0,
  DOMAIN_GPU = 1,
};

enum GpuScopeKind {
  GPU_SCOPE_SYSTEM = 0,
  GPU_SCOPE_DEVICE = 1,
};

enum CpuStoreKind {
  CPU_STORE_NONE = 0,
  CPU_STORE_RELAXED = 1,
  CPU_STORE_RELEASE = 2,
};

enum CpuLoadKind {
  CPU_LOAD_NONE = 0,
  CPU_LOAD_RELAXED = 1,
  CPU_LOAD_ACQUIRE = 2,
  CPU_LOAD_RCSC = 3,
  CPU_LOAD_RCPC = 4,
};

enum GpuStoreKind {
  GPU_STORE_NONE = 0,
  GPU_STORE_RELAXED = 1,
  GPU_STORE_RELEASE = 2,
};

enum GpuLoadKind {
  GPU_LOAD_NONE = 0,
  GPU_LOAD_RELAXED = 1,
  GPU_LOAD_ACQUIRE = 2,
};

struct RoleConfig {
  int domain;
  int gpuScope;
  int cpuStore;
  int cpuLoad1;
  int cpuLoad2;
  int gpuStore;
  int gpuLoad1;
  int gpuLoad2;
};

struct ExperimentConfig {
  const char* name;
  const char* expectation;
  int testKind;
  int roleCount;
  RoleConfig roles[4];
};

struct RunConfig {
  int testKind;
  int roleCount;
  int totalInstances;
  int memStride;
  int memOffset;
  int permuteLocation;
  int barrierSpinLimit;
  int gpuPreStress;
  int gpuPreStressIterations;
  int gpuPreStressPattern;
  RoleConfig roles[4];
};

struct IterationCounts {
  int weak = 0;
  int total = 0;
  int other = 0;
  int na = 0;
};

static constexpr RoleConfig CPU_RELAXED_SB = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_RELAXED, CPU_LOAD_RELAXED, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RELEASE_RCSC_SB = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_RELEASE, CPU_LOAD_RCSC, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RELEASE_RCPC_SB = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_RELEASE, CPU_LOAD_RCPC, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELAXED_SYSTEM_SB = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_RELAXED, GPU_LOAD_RELAXED, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELEASE_ACQUIRE_SYSTEM_SB = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_RELEASE, GPU_LOAD_ACQUIRE, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELEASE_ACQUIRE_DEVICE_SB = {
  DOMAIN_GPU, GPU_SCOPE_DEVICE, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_RELEASE, GPU_LOAD_ACQUIRE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RELAXED_IRIW_WRITER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_RELAXED, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RELEASE_IRIW_WRITER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_RELEASE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RELAXED_IRIW_READER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_RELAXED, CPU_LOAD_RELAXED,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RCSC_IRIW_READER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_RCSC, CPU_LOAD_RELAXED,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_RCPC_IRIW_READER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_RCPC, CPU_LOAD_RELAXED,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig CPU_ACQUIRE_IRIW_READER = {
  DOMAIN_CPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_ACQUIRE, CPU_LOAD_RELAXED,
  GPU_STORE_NONE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELAXED_SYSTEM_IRIW_WRITER = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_RELAXED, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELEASE_SYSTEM_IRIW_WRITER = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_RELEASE, GPU_LOAD_NONE, GPU_LOAD_NONE,
};

static constexpr RoleConfig GPU_RELAXED_SYSTEM_IRIW_READER = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_RELAXED, GPU_LOAD_RELAXED,
};

static constexpr RoleConfig GPU_ACQUIRE_SYSTEM_IRIW_READER = {
  DOMAIN_GPU, GPU_SCOPE_SYSTEM, CPU_STORE_NONE, CPU_LOAD_NONE, CPU_LOAD_NONE,
  GPU_STORE_NONE, GPU_LOAD_ACQUIRE, GPU_LOAD_RELAXED,
};

static constexpr ExperimentConfig kExperiments[] = {
  {
    "sb-arm-baseline",
    "allowed",
    TEST_SB,
    2,
    {CPU_RELAXED_SB, CPU_RELAXED_SB, {}, {}},
  },
  {
    "sb-ptx-baseline",
    "allowed",
    TEST_SB,
    2,
    {GPU_RELAXED_SYSTEM_SB, GPU_RELAXED_SYSTEM_SB, {}, {}},
  },
  {
    "sb-arm-rel-acq-rcsc",
    "disallowed",
    TEST_SB,
    2,
    {CPU_RELEASE_RCSC_SB, CPU_RELEASE_RCSC_SB, {}, {}},
  },
  {
    "sb-arm-rel-acq-rcpc",
    "allowed",
    TEST_SB,
    2,
    {CPU_RELEASE_RCPC_SB, CPU_RELEASE_RCPC_SB, {}, {}},
  },
  {
    "sb-ptx-rel-acq-system",
    "allowed",
    TEST_SB,
    2,
    {GPU_RELEASE_ACQUIRE_SYSTEM_SB, GPU_RELEASE_ACQUIRE_SYSTEM_SB, {}, {}},
  },
  {
    "sb-ptx-rel-acq-device",
    "requested",
    TEST_SB,
    2,
    {GPU_RELEASE_ACQUIRE_DEVICE_SB, GPU_RELEASE_ACQUIRE_DEVICE_SB, {}, {}},
  },
  {
    "sb-mixed-arm-ptx-t0-arm",
    "allowed",
    TEST_SB,
    2,
    {CPU_RELEASE_RCSC_SB, GPU_RELEASE_ACQUIRE_SYSTEM_SB, {}, {}},
  },
  {
    "sb-mixed-arm-ptx-t0-ptx",
    "allowed",
    TEST_SB,
    2,
    {GPU_RELEASE_ACQUIRE_SYSTEM_SB, CPU_RELEASE_RCSC_SB, {}, {}},
  },
  {
    "iriw-arm-baseline",
    "allowed",
    TEST_IRIW,
    4,
    {
      CPU_RELAXED_IRIW_WRITER,
      CPU_RELAXED_IRIW_READER,
      CPU_RELAXED_IRIW_WRITER,
      CPU_RELAXED_IRIW_READER,
    },
  },
  {
    "iriw-ptx-baseline",
    "allowed",
    TEST_IRIW,
    4,
    {
      GPU_RELAXED_SYSTEM_IRIW_WRITER,
      GPU_RELAXED_SYSTEM_IRIW_READER,
      GPU_RELAXED_SYSTEM_IRIW_WRITER,
      GPU_RELAXED_SYSTEM_IRIW_READER,
    },
  },
  {
    "iriw-arm-rel-acq-rcsc",
    "disallowed",
    TEST_IRIW,
    4,
    {
      CPU_RELEASE_IRIW_WRITER,
      CPU_RCSC_IRIW_READER,
      CPU_RELEASE_IRIW_WRITER,
      CPU_RCSC_IRIW_READER,
    },
  },
  {
    "iriw-arm-rel-acq-rcpc",
    "disallowed",
    TEST_IRIW,
    4,
    {
      CPU_RELEASE_IRIW_WRITER,
      CPU_RCPC_IRIW_READER,
      CPU_RELEASE_IRIW_WRITER,
      CPU_RCPC_IRIW_READER,
    },
  },
  {
    "iriw-ptx-rel-acq",
    "allowed",
    TEST_IRIW,
    4,
    {
      GPU_RELEASE_SYSTEM_IRIW_WRITER,
      GPU_ACQUIRE_SYSTEM_IRIW_READER,
      GPU_RELEASE_SYSTEM_IRIW_WRITER,
      GPU_ACQUIRE_SYSTEM_IRIW_READER,
    },
  },
  {
    "iriw-mixed-2rel-ptx-2acq-arm",
    "disallowed",
    TEST_IRIW,
    4,
    {
      GPU_RELEASE_SYSTEM_IRIW_WRITER,
      CPU_ACQUIRE_IRIW_READER,
      GPU_RELEASE_SYSTEM_IRIW_WRITER,
      CPU_ACQUIRE_IRIW_READER,
    },
  },
  {
    "iriw-mixed-2acq-ptx-2rel-arm",
    "allowed",
    TEST_IRIW,
    4,
    {
      CPU_RELEASE_IRIW_WRITER,
      GPU_ACQUIRE_SYSTEM_IRIW_READER,
      CPU_RELEASE_IRIW_WRITER,
      GPU_ACQUIRE_SYSTEM_IRIW_READER,
    },
  },
};

constexpr int kExperimentCount = sizeof(kExperiments) / sizeof(kExperiments[0]);

int random_between(int min, int max) {
  if (min >= max) {
    return min;
  }
  return min + (std::rand() % (max - min + 1));
}

bool percentage_check(int percentage) {
  if (percentage <= 0) {
    return false;
  }
  if (percentage >= 100) {
    return true;
  }
  return (std::rand() % 100) < percentage;
}

int parse_key_value_file(const char* filename, TestParams* params) {
  params->numMemLocations = 0;
  params->permuteLocation = 0;
  FILE* file = fopen(filename, "r");
  if (file == nullptr) {
    std::perror(filename);
    return -1;
  }
  char line[256];
  while (fgets(line, sizeof(line), file) != nullptr) {
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) != 2) {
      continue;
    }
    if (strcmp(key, "numMemLocations") == 0) {
      params->numMemLocations = value;
    } else if (strcmp(key, "permuteLocation") == 0) {
      params->permuteLocation = value;
    }
  }
  fclose(file);
  return 0;
}

int parse_key_value_file(const char* filename, StressParams* params) {
  memset(params, 0, sizeof(*params));
  params->cpuStressThreads = 0;
  params->cpuPreStressPct = 50;
  params->cpuPreStressIterations = 64;
  params->cpuPreStressPattern = 0;
  params->barrierSpinLimit = 4096;

  FILE* file = fopen(filename, "r");
  if (file == nullptr) {
    std::perror(filename);
    return -1;
  }
  char line[256];
  while (fgets(line, sizeof(line), file) != nullptr) {
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) != 2) {
      continue;
    }
    if (strcmp(key, "testIterations") == 0) params->testIterations = value;
    else if (strcmp(key, "testingWorkgroups") == 0) params->testingWorkgroups = value;
    else if (strcmp(key, "maxWorkgroups") == 0) params->maxWorkgroups = value;
    else if (strcmp(key, "workgroupSize") == 0) params->workgroupSize = value;
    else if (strcmp(key, "shufflePct") == 0) params->shufflePct = value;
    else if (strcmp(key, "barrierPct") == 0) params->barrierPct = value;
    else if (strcmp(key, "stressLineSize") == 0) params->stressLineSize = value;
    else if (strcmp(key, "stressTargetLines") == 0) params->stressTargetLines = value;
    else if (strcmp(key, "scratchMemorySize") == 0) params->scratchMemorySize = value;
    else if (strcmp(key, "memStride") == 0) params->memStride = value;
    else if (strcmp(key, "memStressPct") == 0) params->memStressPct = value;
    else if (strcmp(key, "memStressIterations") == 0) params->memStressIterations = value;
    else if (strcmp(key, "memStressPattern") == 0) params->memStressPattern = value;
    else if (strcmp(key, "preStressPct") == 0) params->preStressPct = value;
    else if (strcmp(key, "preStressIterations") == 0) params->preStressIterations = value;
    else if (strcmp(key, "preStressPattern") == 0) params->preStressPattern = value;
    else if (strcmp(key, "stressAssignmentStrategy") == 0) params->stressAssignmentStrategy = value;
    else if (strcmp(key, "permuteThread") == 0) params->permuteThread = value;
    else if (strcmp(key, "cpuStressThreads") == 0) params->cpuStressThreads = value;
    else if (strcmp(key, "cpuPreStressPct") == 0) params->cpuPreStressPct = value;
    else if (strcmp(key, "cpuPreStressIterations") == 0) params->cpuPreStressIterations = value;
    else if (strcmp(key, "cpuPreStressPattern") == 0) params->cpuPreStressPattern = value;
    else if (strcmp(key, "barrierSpinLimit") == 0) params->barrierSpinLimit = value;
  }
  fclose(file);
  return 0;
}

const ExperimentConfig* find_experiment(const char* name) {
  for (int i = 0; i < kExperimentCount; i++) {
    if (strcmp(kExperiments[i].name, name) == 0) {
      return &kExperiments[i];
    }
  }
  return nullptr;
}

void print_experiments() {
  for (int i = 0; i < kExperimentCount; i++) {
    std::cout << kExperiments[i].name << " (expected " << kExperiments[i].expectation << ")\n";
  }
}

__host__ __device__ inline uint x_addr_for(int instance, int memStride) {
  return instance * memStride * 2;
}

__host__ __device__ inline uint y_addr_for_host_device(int instance, int totalInstances,
                                                       int memStride, int memOffset,
                                                       int permuteLocation) {
#ifdef __CUDA_ARCH__
  return permute_id(instance, permuteLocation, totalInstances) * memStride * 2 + memOffset;
#else
  return cpu_permute_id(instance, permuteLocation, totalInstances) * memStride * 2 + memOffset;
#endif
}

inline void cpu_store_value(het_atomic_uint* location, int storeKind, uint value) {
  switch (storeKind) {
    case CPU_STORE_RELAXED:
      location->ref<cuda::thread_scope_system>().store(value, cuda::memory_order_relaxed);
      break;
    case CPU_STORE_RELEASE:
#if defined(__aarch64__)
      asm volatile("stlr %w1, [%0]" :: "r"(&location->raw()), "r"(value) : "memory");
#else
      location->ref<cuda::thread_scope_system>().store(value, cuda::memory_order_release);
#endif
      break;
    default:
      break;
  }
}

inline uint cpu_load_value(const het_atomic_uint* location, int loadKind) {
  switch (loadKind) {
    case CPU_LOAD_RELAXED:
      return location->ref<cuda::thread_scope_system>().load(cuda::memory_order_relaxed);
    case CPU_LOAD_ACQUIRE:
      return location->ref<cuda::thread_scope_system>().load(cuda::memory_order_acquire);
    case CPU_LOAD_RCSC: {
#if defined(__aarch64__)
      uint value;
      asm volatile("ldar %w0, [%1]" : "=r"(value) : "r"(&location->raw()) : "memory");
      return value;
#else
      return location->ref<cuda::thread_scope_system>().load(cuda::memory_order_acquire);
#endif
    }
    case CPU_LOAD_RCPC: {
#if defined(__aarch64__)
      uint value;
      asm volatile("ldapr %w0, [%1]" : "=r"(value) : "r"(&location->raw()) : "memory");
      return value;
#else
      return location->ref<cuda::thread_scope_system>().load(cuda::memory_order_acquire);
#endif
    }
    default:
      return 0;
  }
}

template <cuda::thread_scope Scope>
__device__ inline void gpu_store_scoped(het_atomic_uint* location, uint value, int storeKind) {
  cuda::memory_order order = cuda::memory_order_relaxed;
  if (storeKind == GPU_STORE_RELEASE) {
    order = cuda::memory_order_release;
  }
  if (storeKind != GPU_STORE_NONE) {
    location->ref<Scope>().store(value, order);
  }
}

template <cuda::thread_scope Scope>
__device__ inline uint gpu_load_scoped(const het_atomic_uint* location, int loadKind) {
  cuda::memory_order order = cuda::memory_order_relaxed;
  if (loadKind == GPU_LOAD_ACQUIRE) {
    order = cuda::memory_order_acquire;
  }
  return location->ref<Scope>().load(order);
}

__device__ inline void gpu_store_value(het_atomic_uint* location, const RoleConfig& role, uint value) {
  if (role.gpuScope == GPU_SCOPE_DEVICE) {
    gpu_store_scoped<cuda::thread_scope_device>(location, value, role.gpuStore);
  } else {
    gpu_store_scoped<cuda::thread_scope_system>(location, value, role.gpuStore);
  }
}

__device__ inline uint gpu_load_value(const het_atomic_uint* location, const RoleConfig& role, int whichLoad) {
  int loadKind = (whichLoad == 0) ? role.gpuLoad1 : role.gpuLoad2;
  if (role.gpuScope == GPU_SCOPE_DEVICE) {
    return gpu_load_scoped<cuda::thread_scope_device>(location, loadKind);
  }
  return gpu_load_scoped<cuda::thread_scope_system>(location, loadKind);
}

__device__ void execute_gpu_role(int roleIndex, int instance,
                                 het_atomic_uint* testLocations,
                                 ReadResults* readResults,
                                 het_barrier_t* barriers,
                                 uint* scratchpad,
                                 uint* scratchLocations,
                                 const RunConfig config) {
  const RoleConfig role = config.roles[roleIndex];
  if (role.domain != DOMAIN_GPU) {
    return;
  }

  if (config.gpuPreStress) {
    do_stress(scratchpad, scratchLocations, config.gpuPreStressIterations, config.gpuPreStressPattern);
  }

  het_spin(&barriers[instance], config.roleCount, config.barrierSpinLimit);

  uint xAddr = x_addr_for(instance, config.memStride);
  uint yAddr = y_addr_for_host_device(instance, config.totalInstances, config.memStride,
                                      config.memOffset, config.permuteLocation);

  if (config.testKind == TEST_SB) {
    if (roleIndex == 0) {
      gpu_store_value(&testLocations[xAddr], role, 1);
      readResults[instance].r0 = gpu_load_value(&testLocations[yAddr], role, 0);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    } else {
      gpu_store_value(&testLocations[yAddr], role, 1);
      readResults[instance].r1 = gpu_load_value(&testLocations[xAddr], role, 0);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    }
    return;
  }

  if (roleIndex == 0) {
    gpu_store_value(&testLocations[xAddr], role, 1);
  } else if (roleIndex == 1) {
    readResults[instance].r0 = gpu_load_value(&testLocations[xAddr], role, 0);
    readResults[instance].r1 = gpu_load_value(&testLocations[yAddr], role, 1);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
  } else if (roleIndex == 2) {
    gpu_store_value(&testLocations[yAddr], role, 1);
  } else {
    readResults[instance].r2 = gpu_load_value(&testLocations[yAddr], role, 0);
    readResults[instance].r3 = gpu_load_value(&testLocations[xAddr], role, 1);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
  }
}

__global__ void gpu_role_kernel(het_atomic_uint* testLocations,
                                ReadResults* readResults,
                                het_barrier_t* barriers,
                                uint* scratchpad,
                                uint* scratchLocations,
                                RunConfig config) {
  int taskIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int taskCount = config.totalInstances * config.roleCount;
  if (taskIndex >= taskCount) {
    return;
  }

  int instance = taskIndex / config.roleCount;
  int roleIndex = taskIndex % config.roleCount;
  execute_gpu_role(roleIndex, instance, testLocations, readResults, barriers,
                   scratchpad, scratchLocations, config);
}

struct CpuWorkerContext {
  const RunConfig* config;
  het_atomic_uint* testLocations;
  ReadResults* readResults;
  het_barrier_t* barriers;
  volatile uint* scratchpad;
  bool doPreStress;
  int preStressIterations;
  int preStressPattern;
  int roleIndex;
  int workerIndex;
  int workerCount;
};

void execute_cpu_role(const CpuWorkerContext& ctx, int instance) {
  const RoleConfig& role = ctx.config->roles[ctx.roleIndex];
  uint xAddr = x_addr_for(instance, ctx.config->memStride);
  uint yAddr = y_addr_for_host_device(instance, ctx.config->totalInstances, ctx.config->memStride,
                                      ctx.config->memOffset, ctx.config->permuteLocation);

  if (ctx.doPreStress) {
    cpu_do_stress(ctx.scratchpad, ctx.preStressIterations, ctx.preStressPattern);
  }

  cpu_het_spin(&ctx.barriers[instance], ctx.config->roleCount, ctx.config->barrierSpinLimit);

  if (ctx.config->testKind == TEST_SB) {
    if (ctx.roleIndex == 0) {
      cpu_store_value(&ctx.testLocations[xAddr], role.cpuStore, 1);
      ctx.readResults[instance].r0 = cpu_load_value(&ctx.testLocations[yAddr], role.cpuLoad1);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    } else {
      cpu_store_value(&ctx.testLocations[yAddr], role.cpuStore, 1);
      ctx.readResults[instance].r1 = cpu_load_value(&ctx.testLocations[xAddr], role.cpuLoad1);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    }
    return;
  }

  if (ctx.roleIndex == 0) {
    cpu_store_value(&ctx.testLocations[xAddr], role.cpuStore, 1);
  } else if (ctx.roleIndex == 1) {
    ctx.readResults[instance].r0 = cpu_load_value(&ctx.testLocations[xAddr], role.cpuLoad1);
    ctx.readResults[instance].r1 = cpu_load_value(&ctx.testLocations[yAddr], role.cpuLoad2);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
  } else if (ctx.roleIndex == 2) {
    cpu_store_value(&ctx.testLocations[yAddr], role.cpuStore, 1);
  } else {
    ctx.readResults[instance].r2 = cpu_load_value(&ctx.testLocations[yAddr], role.cpuLoad1);
    ctx.readResults[instance].r3 = cpu_load_value(&ctx.testLocations[xAddr], role.cpuLoad2);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
  }
}

void cpu_role_worker(CpuWorkerContext ctx) {
  for (int instance = ctx.workerIndex; instance < ctx.config->totalInstances; instance += ctx.workerCount) {
    execute_cpu_role(ctx, instance);
  }
}

void set_scratch_locations(uint* scratchLocations, int numBlocks, const StressParams& params) {
  if (numBlocks <= 0) {
    return;
  }
  int numRegions = params.scratchMemorySize / params.stressLineSize;
  if (numRegions <= 0) {
    numRegions = 1;
  }
  for (int i = 0; i < numBlocks; i++) {
    int region = random_between(0, numRegions - 1);
    int locInRegion = random_between(0, params.stressLineSize - 1);
    scratchLocations[i] = region * params.stressLineSize + locInRegion;
  }
}

IterationCounts classify_iteration(const ExperimentConfig& experiment,
                                   const RunConfig& config,
                                   het_atomic_uint* testLocations,
                                   ReadResults* readResults) {
  IterationCounts counts;
  for (int instance = 0; instance < config.totalInstances; instance++) {
    uint xAddr = x_addr_for(instance, config.memStride);
    uint yAddr = y_addr_for_host_device(instance, config.totalInstances, config.memStride,
                                        config.memOffset, config.permuteLocation);
    uint x = testLocations[xAddr].ref<cuda::thread_scope_system>().load(cuda::memory_order_relaxed);
    uint y = testLocations[yAddr].ref<cuda::thread_scope_system>().load(cuda::memory_order_relaxed);
    uint r0 = readResults[instance].r0;
    uint r1 = readResults[instance].r1;
    uint r2 = readResults[instance].r2;
    uint r3 = readResults[instance].r3;

    if (experiment.testKind == TEST_SB) {
      if (x == 0 && y == 0) {
        counts.na++;
      } else if (r0 == 0 && r1 == 0) {
        counts.weak++;
      } else {
        counts.other++;
      }
      continue;
    }

    if (x == 0) {
      counts.na++;
    } else if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) {
      counts.weak++;
    } else {
      counts.other++;
    }
  }

  counts.total = counts.weak + counts.other;
  return counts;
}

void usage() {
  std::cout << "Usage: ./ismm-runner -e <experiment> -s <stress-params> -t <test-params>\n"
            << "       ./ismm-runner --list-experiments\n";
}

int main(int argc, char** argv) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  const char* experimentName = nullptr;
  const char* stressFile = nullptr;
  const char* testFile = nullptr;
  bool listExperiments = false;

  static option longOptions[] = {
    {"experiment", required_argument, nullptr, 'e'},
    {"stress", required_argument, nullptr, 's'},
    {"test", required_argument, nullptr, 't'},
    {"list-experiments", no_argument, nullptr, 'l'},
    {nullptr, 0, nullptr, 0},
  };

  while (true) {
    int optionIndex = 0;
    int c = getopt_long(argc, argv, "e:s:t:l", longOptions, &optionIndex);
    if (c == -1) {
      break;
    }
    switch (c) {
      case 'e':
        experimentName = optarg;
        break;
      case 's':
        stressFile = optarg;
        break;
      case 't':
        testFile = optarg;
        break;
      case 'l':
        listExperiments = true;
        break;
      default:
        usage();
        return 1;
    }
  }

  if (listExperiments) {
    print_experiments();
    return 0;
  }

  if (experimentName == nullptr || stressFile == nullptr || testFile == nullptr) {
    usage();
    return 1;
  }

  const ExperimentConfig* experiment = find_experiment(experimentName);
  if (experiment == nullptr) {
    std::cerr << "Unknown experiment: " << experimentName << "\n";
    print_experiments();
    return 1;
  }

  StressParams stressParams;
  TestParams testParams;
  if (parse_key_value_file(stressFile, &stressParams) != 0 ||
      parse_key_value_file(testFile, &testParams) != 0) {
    return 1;
  }

  RunConfig config = {};
  config.testKind = experiment->testKind;
  config.roleCount = experiment->roleCount;
  config.totalInstances = stressParams.workgroupSize * stressParams.testingWorkgroups;
  config.memStride = stressParams.memStride;
  config.memOffset = stressParams.memStride;
  config.permuteLocation = testParams.permuteLocation;
  config.barrierSpinLimit = stressParams.barrierSpinLimit;
  for (int i = 0; i < 4; i++) {
    config.roles[i] = experiment->roles[i];
  }

  int cpuRoleCount = 0;
  int gpuRoleCount = 0;
  for (int i = 0; i < config.roleCount; i++) {
    if (config.roles[i].domain == DOMAIN_CPU) {
      cpuRoleCount++;
    } else if (config.roles[i].domain == DOMAIN_GPU) {
      gpuRoleCount++;
    }
  }

  int totalInstances = config.totalInstances;
  size_t testLocationsSize = static_cast<size_t>(totalInstances) * testParams.numMemLocations
                           * stressParams.memStride * sizeof(het_atomic_uint);
  size_t readResultsSize = static_cast<size_t>(totalInstances) * sizeof(ReadResults);
  size_t barriersSize = static_cast<size_t>(totalInstances) * sizeof(het_barrier_t);

  het_atomic_uint *hTestLocations, *dTestLocations;
  ReadResults *hReadResults, *dReadResults;
  het_barrier_t *hBarriers, *dBarriers;
  het_malloc(reinterpret_cast<void**>(&hTestLocations), reinterpret_cast<void**>(&dTestLocations), testLocationsSize);
  het_malloc(reinterpret_cast<void**>(&hReadResults), reinterpret_cast<void**>(&dReadResults), readResultsSize);
  het_malloc(reinterpret_cast<void**>(&hBarriers), reinterpret_cast<void**>(&dBarriers), barriersSize);

  uint* dScratchpad = nullptr;
  uint* dScratchLocations = nullptr;
  uint* hScratchLocations = nullptr;
  int maxTaskCount = totalInstances * config.roleCount;
  int maxBlocks = (maxTaskCount + stressParams.workgroupSize - 1) / stressParams.workgroupSize;
  if (maxBlocks <= 0) {
    maxBlocks = 1;
  }
  if (gpuRoleCount > 0) {
    CUDA_CHECK(cudaMalloc(&dScratchpad, static_cast<size_t>(stressParams.scratchMemorySize) * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&dScratchLocations, static_cast<size_t>(maxBlocks) * sizeof(uint)));
    hScratchLocations = static_cast<uint*>(malloc(static_cast<size_t>(maxBlocks) * sizeof(uint)));
  }

  int numCpuWorkers = stressParams.cpuStressThreads > 0
    ? stressParams.cpuStressThreads
    : static_cast<int>(std::thread::hardware_concurrency());
  if (numCpuWorkers <= 0) {
    numCpuWorkers = 4;
  }

  int roleWorkerCount = cpuRoleCount > 0 ? std::max(1, numCpuWorkers / cpuRoleCount) : 0;
  int totalCpuThreads = roleWorkerCount * cpuRoleCount;
  std::vector<volatile uint*> cpuScratchpads(totalCpuThreads, nullptr);
  for (int i = 0; i < totalCpuThreads; i++) {
    cpuScratchpads[i] = static_cast<volatile uint*>(malloc(64 * sizeof(uint)));
    memset(const_cast<uint*>(cpuScratchpads[i]), 0, 64 * sizeof(uint));
  }

  int backgroundStressThreads = std::max(1, numCpuWorkers / 2);
  int stressArraySize = 1024;
  volatile uint* stressArray = static_cast<volatile uint*>(malloc(stressArraySize * sizeof(uint)));
  memset(const_cast<uint*>(stressArray), 0, stressArraySize * sizeof(uint));
  stressArray[0] = 1;
  stressArray[stressArraySize - 1] = 1;
  volatile bool stopStress = false;
  std::vector<std::thread> stressThreads;
  for (int i = 0; i < backgroundStressThreads; i++) {
    stressThreads.emplace_back(cpu_memory_stress_thread, stressArray, stressArraySize, &stopStress);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int totalWeak = 0;
  int totalBehaviors = 0;

  try {
    for (int iteration = 0; iteration < stressParams.testIterations; iteration++) {
      het_memset(hTestLocations, 0, testLocationsSize);
      het_memset(hReadResults, 0, readResultsSize);
      het_memset(hBarriers, 0, barriersSize);

      bool doGpuPreStress = percentage_check(stressParams.preStressPct);
      bool doCpuPreStress = percentage_check(stressParams.cpuPreStressPct);
      config.gpuPreStress = doGpuPreStress ? 1 : 0;
      config.gpuPreStressIterations = stressParams.preStressIterations;
      config.gpuPreStressPattern = stressParams.preStressPattern;

      int taskCount = totalInstances * config.roleCount;
      int blocks = (taskCount + stressParams.workgroupSize - 1) / stressParams.workgroupSize;
      if (blocks <= 0) {
        blocks = 1;
      }

      if (gpuRoleCount > 0) {
        CUDA_CHECK(cudaMemset(dScratchpad, 0, static_cast<size_t>(stressParams.scratchMemorySize) * sizeof(uint)));
        set_scratch_locations(hScratchLocations, blocks, stressParams);
        CUDA_CHECK(cudaMemcpy(dScratchLocations, hScratchLocations,
                              static_cast<size_t>(blocks) * sizeof(uint),
                              cudaMemcpyHostToDevice));
        gpu_role_kernel<<<blocks, stressParams.workgroupSize>>>(
          dTestLocations, dReadResults, dBarriers, dScratchpad, dScratchLocations, config);
        CUDA_CHECK(cudaGetLastError());
      }

      std::vector<std::thread> cpuThreads;
      int scratchIndex = 0;
      for (int roleIndex = 0; roleIndex < config.roleCount; roleIndex++) {
        if (config.roles[roleIndex].domain != DOMAIN_CPU) {
          continue;
        }
        for (int workerIndex = 0; workerIndex < roleWorkerCount; workerIndex++) {
          CpuWorkerContext ctx = {
            &config,
            hTestLocations,
            hReadResults,
            hBarriers,
            cpuScratchpads[scratchIndex],
            doCpuPreStress,
            stressParams.cpuPreStressIterations,
            stressParams.cpuPreStressPattern,
            roleIndex,
            workerIndex,
            roleWorkerCount,
          };
          cpuThreads.emplace_back(cpu_role_worker, ctx);
          scratchIndex++;
        }
      }

      for (auto& thread : cpuThreads) {
        thread.join();
      }
      if (gpuRoleCount > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      IterationCounts counts = classify_iteration(*experiment, config, hTestLocations, hReadResults);
      totalWeak += counts.weak;
      totalBehaviors += counts.total;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "Experiment: " << experiment->name << "\n";
    std::cout << "Expectation: " << experiment->expectation << "\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
    std::cout << std::fixed << std::setprecision(0)
              << "Weak behavior rate: " << (duration.count() > 0.0f ? (float(totalWeak) / duration.count()) : 0.0f)
              << " per second\n";
    std::cout << "Total behaviors: " << totalBehaviors << "\n";
    std::cout << "Number of weak behaviors: " << totalWeak << "\n";
  } catch (...) {
    stopStress = true;
    for (auto& thread : stressThreads) {
      thread.join();
    }
    throw;
  }

  stopStress = true;
  for (auto& thread : stressThreads) {
    thread.join();
  }

  for (volatile uint* scratchpad : cpuScratchpads) {
    free(const_cast<uint*>(scratchpad));
  }
  free(const_cast<uint*>(stressArray));

  if (hScratchLocations != nullptr) {
    free(hScratchLocations);
  }
  if (dScratchpad != nullptr) {
    cudaFree(dScratchpad);
  }
  if (dScratchLocations != nullptr) {
    cudaFree(dScratchLocations);
  }

  het_free(hTestLocations);
  het_free(hReadResults);
  het_free(hBarriers);
  return 0;
}
