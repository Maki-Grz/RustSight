# RustSight

RustSight is a high-performance face detection tool implemented in Rust, utilizing the SCRFD model with ONNX Runtime and OpenCV. It is designed for efficient inference on modern hardware.

> [!CAUTION]
> This project is specifically compatible with Windows 11 ARM64 and was developed on this architecture. Support for other platforms is not guaranteed.

## Features

- SCRFD face detection model integration.
- Optimized inference using ONNX Runtime (ort) with DirectML and QNN execution providers.
- Real-time video processing via OpenCV.
- Letterboxing and Non-Maximum Suppression (NMS) for accurate detection.
- Multi-threaded processing for improved performance.

## Prerequisites

- Windows 11 ARM64 (recommended).
- Rust 1.75 or later.
- OpenCV 4.x installed and configured in your environment.
- ONNX Runtime libraries.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/RustSight.git
   cd RustSight
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

Run the application with the desired configuration:

```bash
cargo run --release -- --model model/scrfd_500m.onnx --source 0
```

### Arguments

- `--model`: Path to the ONNX model file.
- `--source`: Video source (camera index or file path).
- `--provider`: Execution provider (cpu, directml, or qnn).

## License

This project is prepared for open source. Please refer to the LICENSE file for details.
