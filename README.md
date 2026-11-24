# cuHadron — A GPU Data Race Benchmark for CUDA

cuHadron is a benchmark suite for data races in CUDA programs. It leverages modern CUDA features and targets lesser-known race scenarios to help evaluate and compare GPU race detectors. The suite currently includes 27 carefully crafted test cases covering asynchronous copy operations (cp.async, cp.async.bulk), inter-kernel races across streams, CPU–GPU and inter-GPU races using Unified Virtual Memory, distributed shared memory in thread block clusters, and inter-subwarp divergent execution. The benchmark spans multiple execution scopes and memory spaces, and includes false-positive tests to check for false positives (e.g. byte-level independence within a word).

## Test Categories

| Category | Description |
|----------|-------------|
| `memcpy/` | cp.async (global → shared memory) |
| `bulkcpy/` | cp.async.bulk (compute capability 9.0+) |
| `interkernel/` | Inter-kernel races across streams (global memory) |
| `hostdevice/` | CPU–GPU (host–device) races using unified memory |
| `asyncmemcpy/` | Races between cudaMemcpyAsync and kernels |
| `intersubwarp/` | Inter-subwarp races within same warp (shared memory) |
| `dsmem/` | Distributed shared memory (compute capability 9.0+, thread block clusters) |
| `multigpu/` | Multi-GPU races (global memory) |
| `false_positives/` | False positive checks (byte-level vs word-level) |
| `interwave/` | Inter-wave races (e.g. blocks in different waves, global memory) |
| `scopedsync/` | Scoped synchronization between thread blocks in global memory: flags, counters and locks missing a release, an acquire or a fence (PTX memory model) |

## Dependencies

**Software**

- **CUDA Toolkit** 12.x or 13.x: `nvcc`, `nvidia-smi`. **CUDA_HOME** defaults to `/usr/local/cuda` in the Makefile; set it only if your toolkit is installed elsewhere (see **Configuration**).
- **Build**: GNU Make.
- **Test runner**: Bash (`run_tests.sh`).

**Hardware**

- **GPU**: Most tests need **sm_80** (8.0+); **bulkcpy** and **dsmem** need **sm_90** (9.0+). See **test_requirements.txt**. Build with **ARCH** (e.g. `sm_80`, `sm_90`); binaries are named `*.<arch>.out`.
- **Multi-GPU**: The `multigpu/` tests require multiple GPUs; they may be skipped on single-GPU systems.

## Quick Start

```bash
# export CUDA_HOME=/path/to/cuda   # only if not in /usr/local/cuda
make
./run_tests.sh all               # Run all tests (built-in race checker)
./run_tests.sh --baseline all    # Run without race checker
./run_tests.sh memcpy            # Run one category
./run_tests.sh memcpy/shared_writewrite_race   # Run one test
```

Binaries are named `<test>.<arch>.out` (e.g. `.sm80.out`, `.sm90.out`). Only tests whose category supports your ARCH are built (see test_requirements.txt).

## Running Tests with Your Own Race Detection Tool

Builds produce plain CUDA binaries (no built-in race instrumentation). To evaluate your own detector:

1. **Build**: `make` (or `make ARCH=sm_80`, `make ARCH=sm_100` etc.) — produces `*.<arch>.out` only for categories that support that ARCH (see test_requirements.txt).

2. **Run those binaries under your tool** (e.g. preload library, wrapper script, or instrumented CUDA runtime). Binary names match your ARCH (e.g. `*.sm90.out`, `*.sm100.out`).

3. **Optional**: `./run_tests.sh --baseline all` builds and runs the tests; you can capture binary paths or run from category dirs.

**Using the runner with your tool:** Set **`RUN_CMD`** so the script runs each test under your tool. The runner classifies results by **exit code only**: your tool must **exit 0** when no race is detected and **exit 1** when a race is detected.

**Example — NVIDIA Compute Sanitizer (racecheck):** Use `--error-exitcode=1` so the tool exits 1 when it reports a race; otherwise the runner will misclassify results.

```bash
RUN_CMD="compute-sanitizer --tool=racecheck --error-exitcode=1" ./run_tests.sh all
```

To run a single category or test: `RUN_CMD="compute-sanitizer --tool=racecheck --error-exitcode=1" ./run_tests.sh memcpy` or `./run_tests.sh memcpy/shared_writewrite_race`.

## Test Summary

| Test File | Memory | Race Type | Cause |
|-----------|--------|-----------|-------|
| `memcpy/shared_writewrite_race.cu` | Shared | Write-Write | Multiple threads write same shared locations (cp.async) |
| `memcpy/global_readwrite_race.cu` | Global | Read-Write | One warp reads via cp.async while another writes |
| `memcpy/shared_readwrite_race.cu` | Shared | Read-Write | Read before cp.async completes |
| `bulkcpy/shared_readwrite_race.cu` | Shared | Read-Write | Bulk copy reads while other threads write |
| `bulkcpy/global_writewrite_race.cu` | Global | Write-Write | Multiple blocks write same global location |
| `bulkcpy/global_readwrite_race.cu` | Global | Read-Write | Read before cp.async.bulk completes |
| `interkernel/global_writewrite_race.cu` | Global | Write-Write | Two kernels on different streams write same addresses |
| `interkernel/global_readwrite_race.cu` | Global | Read-Write | Consumer reads before producer completes |
| `asyncmemcpy/memcpy_htod_kernel_race.cu` | Global | Read-Write | Kernel reads before cudaMemcpyAsync HtoD completes |
| `asyncmemcpy/kernel_memcpy_dtoh_race.cu` | Global | Read-Write | cudaMemcpyAsync DtoH reads before kernel completes |
| `hostdevice/global_writewrite_race.cu` | Unified | Write-Write | CPU and GPU write same unified addresses |
| `hostdevice/global_readwrite_race.cu` | Unified | Read-Write | CPU reads before GPU kernel completes |
| `hostdevice/global_writeread_race.cu` | Unified | Read-Write | GPU reads while CPU updates |
| `intersubwarp/shared_readwrite_race.cu` | Shared | Read-Write | Divergent threads in warp read/write same address |
| `intersubwarp/shared_writewrite_race.cu` | Shared | Write-Write | Divergent threads in warp write same address |
| `dsmem/shared_readwrite_race.cu` | DSMEM | Read-Write | CTAs access distributed shared memory without sync |
| `dsmem/shared_writewrite_race.cu` | DSMEM | Write-Write | CTAs write DSMEM without sync |
| `multigpu/global_writewrite_race.cu` | Global | Write-Write | Multiple GPUs write same addresses |
| `multigpu/global_readwrite_race.cu` | Global | Read-Write | GPU reads before other GPU completes |
| `false_positives/shared_bytewise_*`, `global_bytewise_*` | Shared/Global | — | False positive checks (see below) |
| `interwave/global_writewrite_race.cu` | Global | Write-Write | Inter-wave: blocks in different waves write same address |

## Fixed (Race-Free) Versions

Most tests support a "fixed" build where synchronization is added so the race is removed. Use this to check that your detector does not report false positives on correct code.

- **Build**: `make FIXED=1`
- **Run**: `./run_tests.sh --fixed all` or `./run_tests.sh --fixed <category>`

Tests use `#ifdef FIXED` to compile the race-free variant (e.g. non-overlapping memory, events, or cluster sync). Fixed runs should show no races detected.

## False Positive Checks

The **`false_positives/`** tests access **different bytes in the same word**. There is no data race; a correct detector should **not** report a race. Use them to check that your tool does not produce false positives. The test runner can report these separately so you can verify correct behavior.

## Build Options

Builds produce plain CUDA binaries for use with your own race detection tool. Per-category minimums (sm_80 usual, sm_90 for bulkcpy/dsmem) are in **test_requirements.txt**; only tests that support your **ARCH** are built.

```bash
make                    # Build for default ARCH=sm_90
make ARCH=sm_80         # Build tests that support sm_80 (bulkcpy, dsmem excluded)
make ARCH=sm_100        # Build for e.g. Blackwell
make FIXED=1            # Build race-free versions (for false-positive checking)
make clean              # Remove binaries
```

Set **ARCH** (e.g. `ARCH=sm_80`, `ARCH=sm_100`) to build and run for a specific architecture; the test runner uses it when invoking `make` and when selecting binaries (`*.sm80.out`, `*.sm90.out`, etc.).

## Limitations

- **Architecture**: Most tests require **sm_80** (8.0+); **bulkcpy** and **dsmem** require **sm_90** (9.0+). See test_requirements.txt.
- **Multi-GPU**: `multigpu/` tests require multiple GPUs; they may be skipped on single-GPU systems.
- **Undetectable by some tools**: Races where the CPU reads from unified memory, or where cudaMemcpyAsync (DtoH) reads device memory, are not visible to tools that do not instrument CPU or DMA. Those tests still contain valid races and are useful for coverage or future tools.

## Configuration

**What to set**

- **CUDA_HOME** — Default is `/usr/local/cuda` in the Makefile. Override if your CUDA toolkit is elsewhere: set `CUDA_HOME` in the Makefile (top of file) or export it before running `make` (e.g. `export CUDA_HOME=/opt/cuda`).
- **ARCH** — Optional. Target architecture (default `sm_90`). Use `sm_80`, `sm_90`, `sm_100` etc. Only categories that support that ARCH are built (test_requirements.txt). Binaries are named `*.<arch>.out`; the test runner uses the same ARCH when invoking `make` and when selecting binaries.

## Adding New Tests

**New test in an existing category**

- Add a `.cu` file in the category folder (e.g. `interkernel/global_readwrite_race.cu`). Follow the naming convention `memory_racetype_[direction_]race.cu` where needed.
- No Makefile or `run_tests.sh` change — wildcards pick up new files.
- Optionally add a row to the Test Summary table in this README.

**New category**

- Create a new directory and add `.cu` files.
- **test_requirements.txt**: Add a line `category min_arch min_gpus` (e.g. `newcat 80 1` or `newcat 90 1`). The Makefile reads this; no Makefile change needed.
- **run_tests.sh**: Add the category name to the list of directories used for "all" and to the `case` that accepts a category name.
- **README**: Add the category to the Test Categories table and to Files and structure; optionally add test rows to the Test Summary table.

## Files and Structure

```
├── test_requirements.txt   # Per-category min_arch, min_gpus (used by Makefile)
├── memcpy/                 # cp.async tests
├── bulkcpy/                # cp.async.bulk (sm_90+)
├── interkernel/            # Inter-kernel stream races
├── hostdevice/             # CPU–GPU unified memory races
├── asyncmemcpy/            # cudaMemcpyAsync vs kernels
├── intersubwarp/           # Inter-subwarp shared memory
├── dsmem/                  # Distributed shared memory (sm_90+)
├── multigpu/               # Multi-GPU races
├── false_positives/        # False positive checks
├── interwave/              # Inter-wave global memory races
├── Makefile
├── run_tests.sh
└── README.md
├── scopedsync/             # Release/acquire and fence races between thread blocks (global memory)
```
