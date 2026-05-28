# Unhash

A high-performance, GPU-accelerated hash reversing and collision tool using Apple's Metal framework. This project implements a sophisticated "meet-in-the-middle" attack to find character sequences that match specific hash targets.

## Project Structure

- **Metal Kernels**:
  - `Dict.metal`: Kernels for building and searching the dictionary-based hash map.
  - `Main.metal`: Kernels for the fixed-length (16-character) prefix/suffix search.
- **Swift Drivers**:
  - `dict.swift`: The main driver for dictionary-based attacks. Loads words from `dict.txt` and sweeps combinations on the GPU.
  - `main.swift`: Driver for the exhaustive 16-character alphanumeric search.
- **Data**:
  - `dict.txt`: Your input dictionary (required for `dict.swift`).
  - `solutions.txt`: Output file where matched solutions are logged with timestamps.

## Prerequisites

- **OS**: macOS
- **Hardware**: Mac with Metal-capable GPU (Apple Silicon or Intel with AMD/Intel GPU)
- **Software**: Xcode Command Line Tools (`swiftc` and `metal` tools)

## Usage

### 1. Dictionary-Based Search
This mode uses words from `dict.txt` to find phrases that match the target hash.

1. Create a `dict.txt` file and populate it with words (one per line).
2. Compile and run:
   ```bash
   swiftc -O dict.swift -o dict && ./dict
   ```

### 2. Fixed-Length Exhaustive Search (16 Characters)
This mode performs a structured search across a 32-character alphabet for a 16-character string.

1. Compile and run:
   ```bash
   swiftc -O main.swift -o main && ./main
   ```
2. (Optional) Resume search from a specific suffix ID:
   ```bash
   ./main [suffixID]
   ```

## Performance Note
Both tools are heavily optimized for GPU execution:
- Uses **Linear Probing** in the GPU hash map to ensure 100% collision survival.
- Implements **L1 Cache Optimization** (threadgroup memory) for high-occupancy searching.
- Uses **Boundary Checking** to handle quantization artifacts in hash space.

## Credits
Based on the logic and constants found in:
[Scratch Project 164028530](https://scratch.mit.edu/projects/164028530)
