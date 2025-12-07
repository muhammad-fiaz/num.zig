# Quick Start

## Installation

1.  Fetch the package:
    ```bash
    zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/heads/main.tar.gz
    ```

2.  Add to `build.zig`:
    ```zig
    const num = b.dependency("num", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("num", num.module("num"));
    ```

## Basic Usage

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2x3 array of zeros
    var a = try num.NDArray(f32).zeros(allocator, &.{2, 3});
    defer a.deinit();

    // Set a value
    try a.set(&.{0, 0}, 1.5);

    // Print
    std.debug.print("Value at (0,0): {d}\n", .{try a.get(&.{0, 0})});
}
```

## Running Examples

The repository includes several examples demonstrating different features:

```bash
zig build run-basics
zig build run-manipulation
zig build run-math
zig build run-linalg
zig build run-random
zig build run-ml
zig build run-fft
```
