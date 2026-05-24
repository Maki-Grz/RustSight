## Project Overview & Abstract

RustSight is an embedded, high-throughput computer vision architecture designed for localized, real-time object and face detection on resource-constrained Windows-ARM64 platforms. The system is engineered to maximize the computational efficiency of the Qualcomm Hexagon Neural Processing Unit (NPU) embedded within Snapdragon X-series systems-on-chip (SoCs). It implements a decoupled, process-isolated pipeline that balances the demands of network video stream ingestion, hardware-accelerated video decoding, multi-dimensional tensor manipulation, and quantized neural network inference.

In multi-stage edge vision pipelines, performance degradation typically occurs due to thread contention and memory allocation overhead during frame ingestion and pre-processing. RustSight mitigates these issues by isolating the video decoding engine from the primary execution thread and bypassing high-overhead CPU inference layers. The total latency per frame frame processing cycle $L_{\text{frame}}$ is modeled as:

$$L_{\text{frame}} = L_{\text{decode}} + L_{\text{transform}} + L_{\text{inference}} + L_{\text{post}}$$

where $L_{\text{decode}}$ represents process-isolated frame extraction, $L_{\text{transform}}$ represents spatial normalization and channel transpositions ($HWC \rightarrow NCHW$), $L_{\text{inference}}$ represents quantized NPU tensor execution, and $L_{\text{post}}$ represents non-maximum suppression (NMS) bounding-box filtering. By enforcing compile-time memory guarantees and utilizing fixed-point integer quantization layers ($\text{w8a8}$), RustSight delivers sub-millisecond coordination profiles critical for high-availability telemetry and autonomous edge monitoring.

---

## Core Architecture & Design Decisions

### Process-Isolated Asymmetric Video Ingestion Pipeline

The ingestion layer implements an asymmetric process-isolated architecture. Rather than compiling heavy multimedia decoding frameworks (e.g., native `libavcodec` bindings) directly into the application space, RustSight spawns an external, optimized `ffmpeg` sub-process. This process decodes incoming network streams (retrieved via a synchronized `yt-dlp` utility hook) or local video files into uncompressed, raw `rgb24` planar sequences, which are written directly to standard output (`pipe:1`).

```
+--------------------------------------------------------------------------+
|                          External OS Subprocesses                        |
|  [Network/File Stream] -> (yt-dlp Hook) -> (ffmpeg Raw rgb24 Decoder)   |
+--------------------------------------------------------------------------+
                                     |
                         Inter-Process Pipe (stdout)
                                     |
                                     v
+--------------------------------------------------------------------------+
|                          RustSight Core Engine                           |
|  [Linear Shared Buffer] -> [Asynchronous Frame Rate Governor]            |
|                                    |                                     |
|                                    v                                     |
|  [OpenCV Grid Resizing] -> [HWC to NCHW Memory Transposition Matrix]     |
+--------------------------------------------------------------------------+
                                     |
                        ONNX Runtime Input Binding
                                     |
                                     v
+--------------------------------------------------------------------------+
|                   Qualcomm AI Engine Direct (QNN) Backend               |
|  (QnnHtp.dll Abstraction via Hexagon Tensor Processor / htp_arch_75)    |
+--------------------------------------------------------------------------+

```

This design decision offers two key advantages:

1. **Fault Isolation:** Memory management anomalies or format violations within network data streams remain encapsulated inside the isolated child process. If a stream failure occurs, the operating system cleans up the process resources without compromising the stability of the parent Rust runtime.
2. **True Multiprocessing:** Video decoding execution runs on separate OS threads, maximizing the utilization of the multi-core Qualcomm Oryon CPU matrix while leaving primary application cores free for downstream inference operations.

### Quantized Hardware Execution Provider Configurations

Neural inference executes via the ONNX Runtime (ORT) core interface, utilizing the Qualcomm AI Engine Direct (`QNN`) execution provider target. The system targets a highly optimized, fully quantized neural network model topology (`gear_guard_net.onnx`) configured with 8-bit weight and activation precision levels ($\text{w8a8}$).

To guarantee low-latency processing, the configuration forces the QNN backend into an explicit hardware profile:

* **`backend_type` / `htp`:** Restricts model execution exclusively to the Hexagon NPU, bypassing standard CPU execution fallbacks.
* **`htp_performance_mode` / `burst`:** Overrides standard power-saving scheduling profiles, locking the processor clocks into high-frequency execution ceilings to achieve maximum throughput.
* **`htp_arch` / `75`:** Informs the compiler compilation layout to target Hexagon v73/v75 microarchitectures directly, matching the physical silicon layout of Snapdragon X series processors.
* **`GraphOptimizationLevel::Disable`:** Graph optimization passes are explicitly turned off. This prevents generic optimization mechanisms from modifying the node layouts of pre-quantized mathematical layers, which can cause accuracy loss or compilation failures within the QNN driver layers.

---

## Algorithmic Design & Data Flow

The internal data flow coordinates raw data streaming, spatial transformation matrices, and fixed-point de-quantization transformations.

### 1. Spatial Transformation and Spatial Inversion Mechanics

Raw bytes are read sequentially from the standard output pipe into a pre-allocated allocation array corresponding to an uncompressed frame size of $1280 \times 720 \times 3$ bytes. The frame is wrapped into an unmanaged OpenCV matrix structure (`Mat`) and rescaled via linear interpolation into the dimensions expected by the underlying model network structure ($192 \times 320$).

Because native hardware image layouts are packed sequentially by row (Height-Width-Channel, $HWC$), but the tensor execution graph requires an indexed planar sequence (Minibatch-Channel-Height-Width, $NCHW$), the system performs an explicit layout transposition pass via `hwc_to_nchw`:

$$\mathcal{A}[0, c, y, x] = \mathcal{P}[(y \cdot W + x) \cdot 3 + c]$$

where $\mathcal{A}$ corresponds to the output 4D tensor matrix, $\mathcal{P}$ represents the source pixel array, $W$ is the target model width ($192$), and $c \in \{0, 1, 2\}$ maps the respective red, green, and blue color channels.

### 2. Analytical Bounding-Box De-quantization

The outputs extracted from the NPU execution layer (`boxes`, `scores`, `class_idx`) are returned as unsigned 8-bit integer vectors (`u8`). To reconstruct physically valid spatial target domains, the application applies an algorithmic de-quantization transformation:

$$X_{\text{float}} = (X_{\text{quant}} - Z_p) \cdot S$$

where $Z_p$ represents the specific layer zero-point offset scalar ($27$ for bounding boxes), and $S$ corresponds to the specific layer scaling transformation multiplier ($1.5097413$ for bounding boxes, $0.0038347607$ for network score layers).

### 3. Non-Maximum Suppression (NMS) Filtering

To resolve spatial bounding redundancy, candidate boxes with confidence weights above $0.40$ undergo a Non-Maximum Suppression routine. The candidate coordinates are sorted by raw confidence vectors in descending order. Overlapping candidate matrices are evaluated and eliminated if their Intersection over Union (IoU) ratio exceeds a strict verification threshold ($0.45$):

$$\text{IoU}(B_i, B_j) = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i \cup B_j)} = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i) + \text{Area}(B_j) - \text{Area}(B_i \cap B_j)}$$

---

## Technical Specifications & Performance Metrics

### Algorithmic Invariants and Constraints

The execution parameters balance memory footprint bounds and processing safety limits:

| Parameter Component | Technical Metric Definition | Operational Functional Boundary |
| --- | --- | --- |
| Input Display Resolution | $1280 \times 720 \text{ Pixels}$ | Native Display Rescaling Interpolation Base |
| Model Tensor Dimensions | $192 \times 320 \times 3 \text{ Bytes}$ | Core $NCHW$ Network Target Space |
| Tensor Precision Quantization | $\text{uint8 (w8a8 Mapping)}$ | Native Hexagon Fixed-Point Target Compatibility |
| Algorithmic Anchor Pruning | $\text{Score Threshold} > 0.40$ | Initial Sparsification Vector Selection Ceiling |
| Suppression Overlap Bounding | $\text{IoU Threshold} = 0.45$ | Multi-Box Redundancy Pruning Metric |

### Dynamic Runtime Profiling

The application measures frame processing intervals using high-resolution hardware timers (`std::time::Instant`). If processing cycles drop below the target framerate threshold ($5.0\text{ Hz}$), the diagnostic overlay flags a `CPU?` fallback state. This indicates that network layers are executing via slower CPU software layers due to missing driver components or driver linkage failure. Under normal conditions, processing cycles execute at high framerates, flagging a native `NPU` execution state.

---

## Deployment & Computational Requirements

### Operating System and System Driver Layout

* **Host Operating System:** Windows 11 ARM64 (`aarch64-pc-windows-msvc`).
* **Kernel Linkage Drivers:** Qualcomm RPC and NPU system infrastructures must be accessible via the Windows DriverStore repository:
* `C:\Windows\System32\DriverStore\FileRepository\qcadsprpc8380.inf_*` (Hexagon Compute Core)
* `C:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8380.inf_*` (NPU HTP Subsystem)


* **Required Native Binaries:** The following runtime drivers must be copied directly into the folder containing the compiled executable:
* `QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnHtpV73Stub.dll`, `QnnSystem.dll`



### Build System Matrix and Package Configuration

The system compilation setup requires native ARM64 libraries managed via the `vcpkg` package manager. The environment is configured using the following PowerShell automation schema:

```powershell
# Instantiate vcpkg tracking infrastructure
$vcpkgRoot = "$HOME\vcpkg"
if (!(Test-Path $vcpkgRoot)) {
    git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
    & "$vcpkgRoot\bootstrap-vcpkg.bat"
}
$env:VCPKG_ROOT = $vcpkgRoot

# Enforce native arm64-windows compilation structures for OpenCV 4
$installPath = "$vcpkgRoot\installed\arm64-windows"
& "$vcpkgRoot\vcpkg" install opencv4[world,opencl]:arm64-windows --recurse

# Expose build parameters to the Cargo toolchain
$env:OPENCV_LINK_PATHS = "$installPath\lib"
$env:OPENCV_INCLUDE_PATHS = "$installPath\include\opencv4"
$env:OPENCV_LINK_LIBS = "opencv_world4"
$env:VCPKGRS_DYNAMIC = "1"
$env:VCPKGRS_TRIPLET = "arm64-windows"
$env:PATH = "$installPath\bin;" + $env:PATH
$env:ORT_STRATEGY = "manual"
$env:OPENCV_DISABLE_PROBES = "cmake,pkg_config"

```

### Execution Directives

To execute the high-performance inference engine, pass the input source URI and optional tracking identifier parameters via the command line interface:

```bash
# Compilation parameter targeting production release profiles
cargo build --release

# Execute target network telemetry acquisition loop
.\target\release\rustsight.exe "https://rtsp.stream.internal/ch1" "CAM-NORTH-01"

# Execute local raw video monitoring file source
.\target\release\rustsight.exe "C:\storage\records\test_sequence.mp4" "BENCH-02"

```
