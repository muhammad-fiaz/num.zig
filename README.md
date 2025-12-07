<div align="center">


<a href="https://muhammad-fiaz.github.io/num.zig/"><img src="https://img.shields.io/badge/docs-muhammad--fiaz.github.io-blue" alt="Documentation"></a>
<a href="https://ziglang.org/"><img src="https://img.shields.io/badge/Zig-0.15.1-orange.svg?logo=zig" alt="Zig Version"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/stars/muhammad-fiaz/num.zig" alt="GitHub stars"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/issues"><img src="https://img.shields.io/github/issues/muhammad-fiaz/num.zig" alt="GitHub issues"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/pulls"><img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/num.zig" alt="GitHub pull requests"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/last-commit/muhammad-fiaz/num.zig" alt="GitHub last commit"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/license/muhammad-fiaz/num.zig" alt="License"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/actions/workflows/ci.yml"><img src="https://github.com/muhammad-fiaz/num.zig/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/platforms-linux%20%7C%20windows%20%7C%20macos-blue" alt="Supported Platforms">
<a href="https://github.com/muhammad-fiaz/num.zig/actions/workflows/github-code-scanning/codeql"><img src="https://github.com/muhammad-fiaz/num.zig/actions/workflows/github-code-scanning/codeql/badge.svg" alt="CodeQL"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/actions/workflows/release.yml"><img src="https://github.com/muhammad-fiaz/num.zig/actions/workflows/release.yml/badge.svg" alt="Release"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/releases/latest"><img src="https://img.shields.io/github/v/release/muhammad-fiaz/num.zig?label=Latest%20Release&style=flat-square" alt="Latest Release"></a>
<a href="https://pay.muhammadfiaz.com"><img src="https://img.shields.io/badge/Sponsor-pay.muhammadfiaz.com-ff69b4?style=flat&logo=heart" alt="Sponsor"></a>
<a href="https://github.com/sponsors/muhammad-fiaz"><img src="https://img.shields.io/badge/Sponsor-💖-pink?style=social&logo=github" alt="GitHub Sponsors"></a>
<a href="https://hits.sh/muhammad-fiaz/num.zig/"><img src="https://hits.sh/muhammad-fiaz/num.zig.svg?label=Visitors&extraCount=0&color=green" alt="Repo Visitors"></a>

<p><em>A fast, high-performance, memory-safe numerical computing and machine learning library for Zig.</em></p>

<b>📚 <a href="https://muhammad-fiaz.github.io/num.zig/">Documentation</a> |
<a href="https://muhammad-fiaz.github.io/num.zig/api/overview">API Reference</a> |
<a href="https://muhammad-fiaz.github.io/num.zig/guide/quick-start">Quick Start</a> |
<a href="CONTRIBUTING.md">Contributing</a></b>

</div>

A production-grade, high-performance numerical computing library for Zig, designed with a clean, intuitive, and developer-friendly API similar to NumPy.

---

<details>
<summary><strong>✨ Features of Num.Zig</strong> (click to expand)</summary>

| Feature | Description |
|---------|-------------|
| ✨ **NDArray** | N-dimensional array implementation with efficient memory management |
| 🎯 **Broadcasting** | NumPy-style broadcasting for arithmetic operations |
| 🚀 **Linear Algebra** | Matrix multiplication, dot products, QR/Cholesky/Eig decompositions, solvers |
| 📁 **Statistics** | Reductions (sum, mean, min, max, std, var, median) |
| 🔍 **Indexing** | Advanced slicing, boolean masking, take, where, nonzero |
| 📡 **Signal Processing** | Convolution, correlation, filtering modes (full, valid, same) |
| 📈 **Polynomials** | Evaluation, arithmetic, roots, derivatives, integrals |
| 🔢 **Calculus** | Finite differences, gradients |
| 🛠️ **Element-wise** | Clip, Round, Floor, Ceil, Abs, Sign, Min/Max, Trig, Log, Exp |
| 🔄 **Random** | Random number generation with various distributions |
| ⚡ **FFT** | N-dimensional Fast Fourier Transform |
| ℂ **Complex Numbers** | Complex number support and operations |
| 💾 **IO** | Binary save/load, Memory Mapping (mmap) |
| 🎨 **Machine Learning** | Dense Layers, Activation Functions, Loss Functions, Optimizers |
| 📊 **Memory Safe** | Built with Zig's safety features and explicit allocator control |
| 📝 **Cross-Platform** | Supports Windows, Linux, macOS, and bare metal |
| 🔗 **Zero Dependencies** | Pure Zig implementation with no external dependencies |
| ⚡ **Performance** | Optimized algorithms including tiled matrix multiplication |
</details>

----

<details>
<summary><strong>📌 Prerequisites & Supported Platforms</strong> (click to expand)</summary>

<br>

## Prerequisites

Before installing Num.Zig, ensure you have the following:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Zig** | 0.15.0+ | Download from [ziglang.org](https://ziglang.org/download/) |
| **Operating System** | Windows 10+, Linux, macOS | Cross-platform support |

> **Tip:** Verify your Zig installation by running `zig version` in your terminal.

---

## Supported Platforms

Num.Zig supports a wide range of platforms and architectures:

| Platform | Architectures | Status |
|----------|---------------|--------|
| **Windows** | x86_64, x86 | ✅ Full support |
| **Linux** | x86_64, x86, aarch64 | ✅ Full support |
| **macOS** | x86_64, aarch64 (Apple Silicon) | ✅ Full support |
| **Bare Metal / Freestanding** | x86_64, aarch64, arm, riscv64 | ✅ Full support |

</details>

---

## Installation

### Method 1: Starter Project (Recommended)

Download the starter project to get up and running quickly:

[**⬇️ Download Starter Project**](https://github.com/muhammad-fiaz/num.zig/releases/latest/download/project-starter-example.zip)

### Method 2: Zig Fetch

The easiest way to add Num.Zig to your existing project:

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/heads/main.tar.gz
```

This automatically adds the dependency with the correct hash to your `build.zig.zon`.

### Method 3: Manual Configuration

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .num = .{
        .url = "https://github.com/muhammad-fiaz/num.zig/archive/refs/heads/main.tar.gz",
        // .hash = "...", // Run zig build to get the hash
    },
},
```

Then in your `build.zig`:

```zig
const num = b.dependency("num", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("num", num.module("num"));
```

## Quick Start

```zig
const std = @import("std");
const num = @import("num");
const NDArray = num.NDArray;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2x3 matrix using arange (0, 1, 2, 3, 4, 5)
    var a = try NDArray(f32).arange(allocator, 0.0, 6.0, 1.0);
    defer a.deinit();
    try a.reshape(&.{ 2, 3 });

    // Create another matrix of ones
    var b = try NDArray(f32).ones(allocator, &.{ 2, 3 });
    defer b.deinit();

    // Add them together
    var c = try num.ops.add(f32, allocator, &a, &b);
    defer c.deinit();

    // Print result
    const val = try c.get(&.{ 0, 0 }); // 0.0 + 1.0 = 1.0
    std.debug.print("Result at [0,0]: {d}\n", .{val});
}
```

## Running Examples

This repository includes several runnable examples covering different aspects of the library:

- `zig build run-basics`: Basic array creation, I/O, and indexing.
- `zig build run-manipulation`: Reshaping, transposing, and flattening arrays.
- `zig build run-math`: Arithmetic operations, broadcasting, and statistics.
- `zig build run-linalg`: Linear algebra operations (matmul, solve).
- `zig build run-random`: Random number generation distributions.
- `zig build run-ml`: Machine learning components (Dense layer, ReLU, MSE).
- `zig build run-fft`: Fast Fourier Transform.
- `zig build run-indexing`: Advanced indexing (slicing, take).
- `zig build run-signal_poly`: Signal processing and polynomials.
- `zig build run-setops`: Set operations.

To run an example:
```bash
zig build run-basics
```

## Usage Examples

### Linear Algebra

```zig
const allocator = std.heap.page_allocator;

// Matrix Multiplication
var a = try NDArray(f32).init(allocator, &.{ 2, 3 });
// ... fill a ...
var b = try NDArray(f32).init(allocator, &.{ 3, 2 });
// ... fill b ...

var c = try num.linalg.matmul(f32, allocator, &a, &b);
defer c.deinit();
```

### Machine Learning (Dense Layer)

```zig
const allocator = std.heap.page_allocator;
var layer = try num.ml.layers.Dense.init(allocator, 10, 5); // Input 10, Output 5
defer layer.deinit();

var input = try NDArray(f32).zeros(allocator, &.{ 1, 10 });
defer input.deinit();

var output = try layer.forward(allocator, &input);
defer output.deinit();
```

## Performance

Num.Zig is designed for high performance. It uses tiled algorithms for matrix multiplication to optimize cache usage and minimize memory bandwidth bottlenecks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
