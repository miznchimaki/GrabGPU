// Based on miznchimaki/GrabGPU gg.cu, blob cc2c225f8bfb62fd94953548310b17acf5015d3c.
// CLI update: named options, help, and literal arguments for real shell scripts.
// Build: nvcc -std=c++11 -arch=sm_80 gg.cu -o gg  (A100 example)

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <sys/stat.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define sleep_ms(t) std::this_thread::sleep_for(std::chrono::milliseconds((long)(t)))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

const double bytes_per_gb = 1024.0 * 1024.0 * 1024.0;
// ============================================================================
// Command line. Parsing and --help do not initialize CUDA.
// This is a GPU workload/reservation utility, not a model-training program.
// ============================================================================

struct Options {
  size_t occupy_size = 0;
  float total_time = 0.0f;
  float utilization = 0.0f;  // Set by the required --arg4; no CLI default.
  std::vector<int> gpu_ids;
  bool all_gpus = false;
  std::string script_path;
  std::vector<std::string> script_args;
};

static void print_help(const char* program) {
  std::cout
      << "GrabGPU - Tensor Core workload / GPU reservation utility\n\n"
      << "Usage:\n  " << program
      << " [--script PATH] [--NAME VALUE ...]\n"
      << "      --arg1 GIB --arg2 HOURS --arg3 GPU_IDS --arg4 FRACTION\n\n"
      << "The last four pairs are required, in exactly the order shown:\n"
      << "  --arg1 GIB       GPU memory to allocate per device, in GiB (> 0).\n"
      << "  --arg2 HOURS     Built-in workload duration in hours (>= 0).\n"
      << "                   0 runs one duty-cycle iteration after calibration.\n"
      << "  --arg3 GPU_IDS   CUDA-visible IDs, e.g. 0 or 0,1; -1 = all visible GPUs.\n"
      << "  --arg4 FRACTION Target duty fraction in (0, 1]; required, no default.\n\n"
      << "Options before those last four pairs:\n"
      << "  --script PATH   Run a real shell script using /bin/sh after allocation\n"
      << "                  succeeds and the reserved buffers are released.\n"
      << "  --NAME VALUE    Forward this pair verbatim to --script as arguments.\n"
      << "                  Any number of these pairs is allowed. They require\n"
      << "                  --script and are not consumed by the built-in kernel.\n"
      << "                  Training defaults belong to the actual script.\n"
      << "  --help, -h, help\n"
      << "                  Show this help without initializing CUDA.\n\n"
      << "All value-taking options use --name value (not --name=value).\n"
      << "Quote values containing spaces. Values cannot start with '--'.\n"
      << "The names arg1, arg2, arg3, arg4, script and help are reserved.\n"
      << "The old --utilization option has been replaced by required --arg4.\n"
      << "GPU IDs are relative to CUDA_VISIBLE_DEVICES when it is set.\n"
      << "Script mode inherits CUDA_VISIBLE_DEVICES unchanged. --arg3 only\n"
      << "selects devices for the reservation; the script selects its own GPUs.\n"
      << "In script mode, HOURS and utilization do not limit the script.\n"
      << "Releasing the buffers before the script starts is not an atomic handoff.\n"
      << "Only arguments actually passed at launch appear in the command line;\n"
      << "internal defaults are not inserted into process arguments.\n\n"
      << "Examples:\n  " << program
      << " --arg1 16 --arg2 24 --arg3 0,1 --arg4 0.5\n  " << program
      << " --arg1 16 --arg2 24 --arg3 0 --arg4 0.6\n  " << program
      << " --script run.sh --epochs 10 --batch-size 64 --lr 1e-4"
      << " --arg1 16 --arg2 24 --arg3 0,1 --arg4 0.5\n\n"
      << "run.sh must read or forward its arguments (for example, with \"$@\").\n"
      << "No model, dataset, training loop or training defaults are built in.\n";
}

static double parse_number(const std::string& value, const char* name) {
  if (value.empty() ||
      std::isspace(static_cast<unsigned char>(value.front()))) {
    throw std::invalid_argument(std::string(name) + " requires a number");
  }
  errno = 0;
  char* end = nullptr;
  const double result = std::strtod(value.c_str(), &end);
  if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
      !std::isfinite(result)) {
    throw std::invalid_argument(std::string(name) +
                                " requires a finite number: " + value);
  }
  return result;
}

static std::vector<int> parse_gpu_ids(const std::string& value,
                                      bool& all_gpus) {
  all_gpus = (value == "-1");
  if (all_gpus) return std::vector<int>();
  if (value.empty() || value.back() == ',') {
    throw std::invalid_argument("--arg3 requires GPU IDs such as 0 or 0,1");
  }
  std::vector<int> ids;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      throw std::invalid_argument("--arg3 contains an empty GPU ID");
    }
    int id = 0;
    for (char ch : token) {
      if (ch < '0' || ch > '9') {
        throw std::invalid_argument("--arg3 has invalid GPU ID: " + token);
      }
      const int digit = ch - '0';
      if (id > (std::numeric_limits<int>::max() - digit) / 10) {
        throw std::invalid_argument("--arg3 GPU ID is too large: " + token);
      }
      id = id * 10 + digit;
    }
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
      throw std::invalid_argument("--arg3 has duplicate GPU ID: " + token);
    }
    ids.push_back(id);
  }
  if (ids.empty()) {
    throw std::invalid_argument("--arg3 must select at least one GPU");
  }
  return ids;
}

static bool is_long_option(const std::string& name) {
  if (name.size() < 3 || name.compare(0, 2, "--") != 0) return false;
  if (!std::isalnum(static_cast<unsigned char>(name[2]))) return false;
  for (size_t i = 2; i < name.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(name[i]);
    if (!std::isalnum(ch) && ch != '-' && ch != '_') return false;
  }
  return true;
}

static Options process_args(int argc, char** argv) {
  if (argc < 9 || std::string(argv[argc - 8]) != "--arg1" ||
      std::string(argv[argc - 6]) != "--arg2" ||
      std::string(argv[argc - 4]) != "--arg3" ||
      std::string(argv[argc - 2]) != "--arg4") {
    throw std::invalid_argument(
        "end the command with --arg1 GIB --arg2 HOURS --arg3 GPU_IDS --arg4 FRACTION");
  }

  Options options;
  const int tail = argc - 8;
  bool have_script = false;
  for (int i = 1; i < tail; i += 2) {
    const std::string name(argv[i]);
    if (!is_long_option(name) || name == "--help") {
      throw std::invalid_argument("expected an option in --name value form: " +
                                  name);
    }
    if (name == "--arg1" || name == "--arg2" ||
        name == "--arg3" || name == "--arg4") {
      throw std::invalid_argument(name +
                                  " must appear exactly once at the end");
    }
    if (i + 1 >= tail || std::string(argv[i + 1]).compare(0, 2, "--") == 0) {
      throw std::invalid_argument("missing value for " + name);
    }
    const std::string value(argv[i + 1]);
    if (name == "--script") {
      if (have_script || value.empty()) {
        throw std::invalid_argument("--script requires one nonempty path");
      }
      options.script_path = value;
      have_script = true;
    } else if (name == "--utilization") {
      throw std::invalid_argument("--utilization has been replaced by required --arg4");
    } else {
      options.script_args.push_back(name);
      options.script_args.push_back(value);
    }
  }

  if (!options.script_args.empty() && !have_script) {
    throw std::invalid_argument(options.script_args.front() +
        " is a script argument and requires --script PATH");
  }

  const double gib = parse_number(argv[tail + 1], "--arg1");
  const long double bytes =
      static_cast<long double>(gib) * 1024.0L * 1024.0L * 1024.0L;
  if (gib <= 0.0 || bytes < 1.0L ||
      bytes >= static_cast<long double>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument("--arg1 must be a positive, representable GiB size");
  }
  options.occupy_size = static_cast<size_t>(bytes);

  const double hours = parse_number(argv[tail + 3], "--arg2");
  if (hours < 0.0 || hours > std::numeric_limits<float>::max() ||
      (hours > 0.0 && hours < std::numeric_limits<float>::min())) {
    throw std::invalid_argument("--arg2 must be nonnegative and representable");
  }
  options.total_time = static_cast<float>(hours);
  options.gpu_ids = parse_gpu_ids(argv[tail + 5], options.all_gpus);

  const double fraction = parse_number(argv[tail + 7], "--arg4");
  if (fraction <= 0.0 || fraction > 1.0 ||
      fraction < static_cast<double>(std::numeric_limits<float>::min())) {
    throw std::invalid_argument("--arg4 must be in (0, 1]");
  }
  options.utilization = static_cast<float>(fraction);
  return options;
}

static bool help_requested(int argc, char** argv) {
  if (argc == 1) return true;
  if (argc == 2 && std::string(argv[1]) == "help") return true;
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      return true;
    }
  }
  return false;
}

// End of command-line-only code.


// ============================================================================
// Tensor-core occupier kernel
//
// Goal: make the DCGM/GPM metrics `sm_active`, `sm_occupancy` and
// `tensor_active` (HMMA) all read high — close to the coarse GPU-utilization
// number — during the on-phase. The original kernel ran a scalar FP32 add loop,
// which raised the coarse "gpu util %" but left tensor_active at ~0 and
// sm_occupancy mediocre.
//
// Each warp runs a tight chain of 16x16x16 half->float HMMA instructions. The
// accumulator is both input and output of every mma_sync (D = A*B + C), so the
// iterations form a true dependency chain the compiler cannot fold away, and
// the final store of the accumulator makes the whole loop side-effecting (no
// dead-code elimination). A and B are constant register-resident fragments, so
// the inner loop is pure tensor-core compute with zero memory traffic — which
// keeps the HMMA pipe saturated and makes run-time linear in `iterations`.
//
// Measured on H20 (sm_90) during a sustained on-phase:
//   sm_active ~99%, tensor_active ~95%, hmma ~95%, sm_occupancy ~97%
//   (sm_occupancy only reaches ~97% when the grid is OVERSUBSCRIBED — see
//    launch config below; a single resident wave measured only ~56%.)
// ============================================================================
__global__ void tensor_occupier_kernel(char* buffer, size_t buffer_size,
                                        int iterations) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(a_frag, __float2half(1.0f));
  wmma::fill_fragment(b_frag, __float2half(1.0f));
  wmma::fill_fragment(c_frag, 0.0f);

  // Sustained HMMA loop: c = a*b + c. The read-modify-write on c_frag chains the
  // iterations together so ptxas can neither hoist nor eliminate them.
  for (int iter = 0; iter < iterations; ++iter) {
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Sink: write one accumulator lane into the (already-allocated) buffer so the
  // loop has an observable side effect and survives optimization.
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer_size) buffer[idx] = (char)c_frag.x[0];
}

// ============================================================================
// Host-side launch helpers
// ============================================================================

// Launch the tensor kernel on every requested GPU asynchronously so they all
// run concurrently, then the caller synchronizes them together.
static void launch_tensor_kernel(char** array, size_t buffer_size,
                                 int iterations, int grid_x, int block_x,
                                 const std::vector<int>& gpu_ids) {
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    tensor_occupier_kernel<<<grid_x, block_x>>>(array[id], buffer_size,
                                                iterations);
    CUDA_CHECK(cudaGetLastError());
  }
}

// One-time calibration: pick the iteration count that makes a single launch run
// for ~`target_on_time_ms` (default 50 ms). A steady on-phase of tens of ms is
// long enough that the DCGM/GPM sampling window (~100 ms; nvidia-smi averages
// over 1 s) captures a stable high value instead of a brief spike. This also
// removes the manual loop-count tuning the old README asked users to do.
//
// Per-launch throughput on a shared GPU is noisy — measured on H20 it swings
// ~2x launch-to-launch (68..146 iter/ms) even at a fixed iteration count — so a
// single timed sample is unreliable. We take several samples per GPU and use the
// MEDIAN (robust to outliers), then the slowest GPU (fewest iters/ms) so every
// GPU runs for at least the target on-time. The exact on-time need not be hit:
// anything in the tens-of-ms range works, and the duty-cycle sleep in the main
// loop recomputes the off-time from the *measured* on-time every iteration, so
// the metric time-average stays correct regardless of drift. Hence NO per-sample
// re-calibration in the hot loop (that would chase the 2x jitter and oscillate).
static int calibrate_iterations(char** array, size_t buffer_size,
                                const std::vector<int>& gpu_ids, int grid_x,
                                int block_x, float target_on_time_ms) {
  // Warm-up: establish context and ramp clocks so timed runs are representative.
  launch_tensor_kernel(array, buffer_size, /*iterations=*/200, grid_x, block_x,
                       gpu_ids);
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  const int calib_iter = 4000;
  const int n_samples = 7;

  double min_iter_per_ms = 1e30;
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<double> samples;
    samples.reserve(n_samples);
    for (int s = 0; s < n_samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      tensor_occupier_kernel<<<grid_x, block_x>>>(array[id], buffer_size,
                                                  calib_iter);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (ms > 0.0f) samples.push_back((double)calib_iter / (double)ms);
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    std::sort(samples.begin(), samples.end());
    double median_iter_per_ms = samples[samples.size() / 2];
    if (median_iter_per_ms < min_iter_per_ms)
      min_iter_per_ms = median_iter_per_ms;
    printf("  GPU-%d calibration: median %.1f iter/ms over %d samples\n", id,
           median_iter_per_ms, (int)samples.size());
  }

  int target_iter = (int)(min_iter_per_ms * (double)target_on_time_ms);
  if (target_iter < 100) target_iter = 100;  // safety floor
  printf("Calibration: %.1f iter/ms -> %d iters for ~%.0f ms on-time\n",
         min_iter_per_ms, target_iter, (double)target_on_time_ms);
  return target_iter;
}

// ============================================================================
// Default (built-in) occupancy script — tensor-core aware.
// ============================================================================
void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids, float utilization) {
  printf("Running tensor occupier with target utilization: %.2f%% "
         ">>>>>>>>>>>>>>>>>>>>\n",
         utilization * 100);

  // --- Launch config ---------------------------------------------------------
  // block = 256 threads (8 warps). On H20 this permits 8 resident blocks/SM =
  // 2048 threads/SM = 100% theoretical occupancy.
  //
  // The number of blocks that actually fit per SM is queried at runtime via the
  // occupancy API (robust across GPUs/driver versions). We then OVERSUBSCRIBE
  // the grid by `oversub` waves: launching only one resident wave measured just
  // ~56% achieved sm_occupancy, because warps drain at the kernel's tail with no
  // backlog to refill the slots. Oversubscribing keeps every warp slot
  // continuously full, pushing achieved sm_occupancy to ~97%.
  const int block_x = 256;
  const int oversub = 32;

  int sm_count = 0;
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaSetDevice(gpu_ids[0]));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    gpu_ids[0]));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, tensor_occupier_kernel, block_x, 0));
  if (blocks_per_sm < 1) blocks_per_sm = 1;

  const int grid_x = sm_count * blocks_per_sm * oversub;

  printf("Launch: %d blocks x %d threads | %d SMs, %d resident blocks/SM, "
         "%dx oversubscribed\n",
         grid_x, block_x, sm_count, blocks_per_sm, oversub);

  // --- Calibrate iteration count for a steady ~50 ms on-phase ----------------
  const float target_on_time_ms = 50.0f;
  int target_iter = calibrate_iterations(array, occupy_size, gpu_ids, grid_x,
                                         block_x, target_on_time_ms);

  // --- Main duty-cycle loop ---------------------------------------------------
  auto start_total = std::chrono::steady_clock::now();
  auto last_log_time = start_total;

  while (true) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // Launch on ALL GPUs first (async), then sync ALL -> concurrent execution.
    launch_tensor_kernel(array, occupy_size, target_iter, grid_x, block_x,
                         gpu_ids);
    for (int id : gpu_ids) {
      CUDA_CHECK(cudaSetDevice(id));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double on_time_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Duty-cycle sleep: the OFF phase time-averages sm_active / sm_occupancy /
    // tensor_active down to ~= the target utilization. The off-time is derived
    // from the *measured* on-time each iteration, so even though per-launch
    // timing is noisy on a shared GPU the duty ratio (and therefore the metric
    // average) stays correct — no iteration re-tuning needed.
    // e.g. util=0.6 -> ~50 ms on + ~33 ms off, and all three metrics read ~60%.
    if (utilization < 1.0f && on_time_ms > 0) {
      double off_time_ms = on_time_ms * (1.0 / utilization - 1.0);
      if (off_time_ms > 0.5) {
        // Sleep in bounded chunks: the original cast to long can overflow
        // for tiny (but valid) utilization fractions. Check the deadline too.
        while (off_time_ms > 0.0) {
          const double elapsed = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start_total).count();
          const double remaining_ms =
              (static_cast<double>(total_time) * 3600.0 - elapsed) * 1000.0;
          if (remaining_ms <= 0.0) break;
          const double chunk_ms =
              std::min(1000.0, std::min(off_time_ms, remaining_ms));
          std::this_thread::sleep_for(
              std::chrono::duration<double, std::milli>(chunk_ms));
          off_time_ms -= chunk_ms;
        }
      }
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed_hours =
        std::chrono::duration<double, std::ratio<3600>>(now - start_total).count();
    if (elapsed_hours > total_time) break;

    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time)
            .count() > 10) {
      printf("Occupied time: %.2f hours (Last Kernel Duration: %.3f ms, "
             "Iterations: %d)\n",
             elapsed_hours, on_time_ms, target_iter);
      last_log_time = now;
    }
  }
}

void allocate_mem(char** array, size_t array_size, size_t occupy_size,
                  const std::vector<int>& gpu_ids) {
  std::vector<bool> allocated(array_size, false);
  int cnt = 0;
  while (true) {
    printf("Try allocate GPU memory %d times >>>>>>>>>>>>>>>>>>>>\n", ++cnt);
    bool all_allocated = true;
    for (int id : gpu_ids) {
      if (!allocated[id]) {
        CUDA_CHECK(cudaSetDevice(id));
        cudaError_t status = cudaMalloc(&array[id], occupy_size);
        size_t total_size, avail_size;
        CUDA_CHECK(cudaMemGetInfo(&avail_size, &total_size));
        if (status != cudaSuccess) {
          if (status != cudaErrorMemoryAllocation) {
            CUDA_CHECK(status);
          }
          // Clear the handled allocation error before retrying.
          (void)cudaGetLastError();
          printf(
              "GPU-%d: Failed to allocate %.2f GB GPU memory (%.2f GB "
              "available)\n",
              id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
          all_allocated = false;
        } else {
          allocated[id] = true;
          printf(
              "GPU-%d: Successfully allocate %.2f GB GPU memory (%.2f GB "
              "available)\n",
              id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
        }
      }
    }
    if (all_allocated) break;
    sleep_ms(5000);
  }
  printf("Successfully allocate memory on all GPUs!\n");
}

// Validate the real script before waiting for GPU memory.
static void validate_script_path(const std::string& script_path) {
  if (script_path.empty()) return;
  struct stat info;
  if (stat(script_path.c_str(), &info) != 0) {
    throw std::invalid_argument("cannot open script: " + script_path +
                                ": " + std::strerror(errno));
  }
  if (!S_ISREG(info.st_mode) || access(script_path.c_str(), R_OK) != 0) {
    throw std::invalid_argument("script must be a readable regular file: " +
                                script_path);
  }
}

static int initialize_devices(Options& options) {
  int gpu_num = 0;
  CUDA_CHECK(cudaGetDeviceCount(&gpu_num));
  if (gpu_num <= 0) throw std::runtime_error("no CUDA-visible GPU is available");
  if (options.all_gpus) {
    for (int id = 0; id < gpu_num; ++id) options.gpu_ids.push_back(id);
  }
  for (int id : options.gpu_ids) {
    if (id < 0 || id >= gpu_num) {
      throw std::invalid_argument("GPU ID " + std::to_string(id) +
          " is outside the CUDA-visible range 0.." + std::to_string(gpu_num - 1));
    }
    CUDA_CHECK(cudaSetDevice(id));
    size_t total_size = 0, avail_size = 0;
    CUDA_CHECK(cudaMemGetInfo(&avail_size, &total_size));
    if (options.occupy_size > total_size) {
      throw std::invalid_argument("--arg1 exceeds total memory on GPU " +
                                  std::to_string(id));
    }
    if (options.script_path.empty()) {
      cudaDeviceProp properties;
      CUDA_CHECK(cudaGetDeviceProperties(&properties, id));
      if (properties.major < 7) {
        throw std::invalid_argument("the built-in Tensor Core kernel requires "
                                    "compute capability >= 7.0");
      }
    }
  }
  return gpu_num;
}

static void release_mem(char** array, const std::vector<int>& gpu_ids) {
  for (int id : gpu_ids) {
    if (array[id] != nullptr) {
      CUDA_CHECK(cudaSetDevice(id));
      CUDA_CHECK(cudaFree(array[id]));
      array[id] = nullptr;
    }
  }
}

static void run_custom_script(const Options& options) {
  // Preserve the original /bin/sh behavior. execv passes literal arguments;
  // script paths/values containing spaces or shell metacharacters are safe.
  std::string path = options.script_path;
  if (!path.empty() && path.front() == '-') path = "./" + path;
  std::vector<std::string> arguments;
  arguments.push_back("/bin/sh");
  arguments.push_back(path);
  arguments.insert(arguments.end(), options.script_args.begin(),
                   options.script_args.end());
  std::vector<char*> argv;
  for (std::string& value : arguments) {
    argv.push_back(const_cast<char*>(value.c_str()));
  }
  argv.push_back(nullptr);
  std::cout << "Starting actual script: " << options.script_path
            << " (GPU reservations released)\n";
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  execv("/bin/sh", argv.data());
  throw std::runtime_error(std::string("cannot execute /bin/sh: ") +
                           std::strerror(errno));
}

int main(int argc, char** argv) {
  if (help_requested(argc, argv)) {
    print_help(argv[0]);
    return 0;
  }

  Options options;
  try {
    options = process_args(argc, argv);
    validate_script_path(options.script_path);
    const int gpu_num = initialize_devices(options);
    std::vector<char*> array(static_cast<size_t>(gpu_num), nullptr);

    std::cout << "GrabGPU: "
              << (options.script_path.empty() ? "Tensor Core workload" :
                                                  "reservation before real script")
              << "\nGPU memory per device (GiB): "
              << options.occupy_size / bytes_per_gb
              << "\nBuilt-in duration (h): " << options.total_time
              << "\nBuilt-in target utilization: "
              << options.utilization * 100.0f << "%\nGPU IDs: ";
    for (size_t i = 0; i < options.gpu_ids.size(); ++i) {
      std::cout << (i ? "," : "") << options.gpu_ids[i];
    }
    std::cout << std::endl;

    allocate_mem(array.data(), array.size(), options.occupy_size, options.gpu_ids);
    if (options.script_path.empty()) {
      run_default_script(array.data(), options.occupy_size, options.total_time,
                         options.gpu_ids, options.utilization);
      release_mem(array.data(), options.gpu_ids);
    } else {
      release_mem(array.data(), options.gpu_ids);
      // Tear down these CUDA contexts before replacing this process.
      for (int id : options.gpu_ids) {
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaDeviceReset());
      }
      run_custom_script(options);
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what()
              << "\nRun '" << argv[0] << " --help' for usage." << std::endl;
    return 1;
  }

  return 0;
}
