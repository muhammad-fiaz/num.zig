# Quick Start

Get started with `num.zig` in minutes! This guide will walk you through installation and your first numerical computing program.

## Installation

### Prerequisites

- Zig 0.13.0 or later
- Git (for cloning examples)

### Method 1: Zig Fetch (Recommended)

Add `num.zig` to your project dependencies:

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/heads/main.tar.gz
```

This will automatically update your `build.zig.zon` file.

### Method 2: Manual Configuration

1. Add to `build.zig.zon`:

```zig
.{
    .name = "my_project",
    .version = "0.1.0",
    .dependencies = .{
        .num = .{
            .url = "https://github.com/muhammad-fiaz/num.zig/archive/refs/heads/main.tar.gz",
            // Add hash after first build
        },
    },
}
```

2. Update `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "my_project",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add num.zig dependency
    const num = b.dependency("num", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("num", num.module("num"));

    b.installArtifact(exe);
}
```

3. Build your project:

```bash
zig build
```

## Your First Program

Create `src/main.zig`:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 1D array
    var arr = try num.NDArray(f64).arange(allocator, 0.0, 10.0, 1.0);
    defer arr.deinit(allocator);

    std.debug.print("Array: ", .{});
    try arr.print();
}
```

**Run:**
```bash
zig build run
```

**Output:**
```
Array: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```

## Basic Examples

### Example 1: Creating Arrays

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Zeros
    var zeros = try num.NDArray(f64).zeros(allocator, &.{3, 3});
    defer zeros.deinit(allocator);
    std.debug.print("Zeros:\n", .{});
    try zeros.print();

    // Ones
    var ones = try num.NDArray(f64).ones(allocator, &.{2, 4});
    defer ones.deinit(allocator);
    std.debug.print("\nOnes:\n", .{});
    try ones.print();

    // Custom value
    var sevens = try num.NDArray(f64).full(allocator, &.{2, 2}, 7.0);
    defer sevens.deinit(allocator);
    std.debug.print("\nSevens:\n", .{});
    try sevens.print();

    // Identity matrix
    var identity = try num.NDArray(f64).eye(allocator, 4);
    defer identity.deinit(allocator);
    std.debug.print("\nIdentity:\n", .{});
    try identity.print();
}
```

**Output:**
```
Zeros:
[[0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0]]

Ones:
[[1.0, 1.0, 1.0, 1.0],
 [1.0, 1.0, 1.0, 1.0]]

Sevens:
[[7.0, 7.0],
 [7.0, 7.0]]

Identity:
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]]
```

### Example 2: Array Arithmetic

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create arrays
    var a = try num.NDArray(f64).arange(allocator, 1.0, 6.0, 1.0);
    defer a.deinit(allocator);
    
    var b = try num.NDArray(f64).full(allocator, &.{5}, 2.0);
    defer b.deinit(allocator);

    std.debug.print("a = ", .{});
    try a.print();
    std.debug.print("b = ", .{});
    try b.print();

    // Addition
    var sum = try num.ops.add(allocator, f64, a, b);
    defer sum.deinit(allocator);
    std.debug.print("\na + b = ", .{});
    try sum.print();

    // Multiplication
    var product = try num.ops.mul(allocator, f64, a, b);
    defer product.deinit(allocator);
    std.debug.print("a * b = ", .{});
    try product.print();

    // Power
    var squared = try num.ops.pow(allocator, f64, a, 2.0);
    defer squared.deinit(allocator);
    std.debug.print("a² = ", .{});
    try squared.print();
}
```

**Output:**
```
a = [1.0, 2.0, 3.0, 4.0, 5.0]
b = [2.0, 2.0, 2.0, 2.0, 2.0]

a + b = [3.0, 4.0, 5.0, 6.0, 7.0]
a * b = [2.0, 4.0, 6.0, 8.0, 10.0]
a² = [1.0, 4.0, 9.0, 16.0, 25.0]
```

### Example 3: Matrix Operations

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2×3 matrix
    var matrix = try num.NDArray(f64).arange(allocator, 1.0, 7.0, 1.0);
    defer matrix.deinit(allocator);
    // Reshape to 2×3
    try matrix.reshape(&.{2, 3});

    std.debug.print("Matrix:\n", .{});
    try matrix.print();

    // Access elements
    const val = try matrix.get(&.{1, 2});
    std.debug.print("\nElement at [1,2]: {d}\n", .{val});

    // Set elements
    try matrix.set(&.{0, 0}, 10.0);
    std.debug.print("\nAfter setting [0,0] = 10:\n", .{});
    try matrix.print();

    // Matrix properties
    std.debug.print("\nShape: {any}\n", .{matrix.shape});
    std.debug.print("Rank: {}\n", .{matrix.rank()});
    std.debug.print("Size: {}\n", .{matrix.size()});
}
```

**Output:**
```
Matrix:
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]

Element at [1,2]: 6.0

After setting [0,0] = 10:
[[10.0, 2.0, 3.0],
 [ 4.0, 5.0, 6.0]]

Shape: [2, 3]
Rank: 2
Size: 6
```

### Example 4: Statistics

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var data = try num.NDArray(f64).arange(allocator, 1.0, 11.0, 1.0);
    defer data.deinit(allocator);

    std.debug.print("Data: ", .{});
    try data.print();

    const mean = try num.stats.mean(allocator, f64, data);
    const std_dev = try num.stats.stdDev(allocator, f64, data);
    const min_val = try num.stats.min(allocator, f64, data);
    const max_val = try num.stats.max(allocator, f64, data);

    std.debug.print("\nStatistics:\n", .{});
    std.debug.print("  Mean: {d:.2}\n", .{mean});
    std.debug.print("  Std Dev: {d:.2}\n", .{std_dev});
    std.debug.print("  Min: {d:.2}\n", .{min_val});
    std.debug.print("  Max: {d:.2}\n", .{max_val});
}
```

**Output:**
```
Data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

Statistics:
  Mean: 5.50
  Std Dev: 2.87
  Min: 1.00
  Max: 10.00
```

### Example 5: Linear Algebra

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create two matrices
    var A = try num.NDArray(f64).ones(allocator, &.{2, 3});
    defer A.deinit(allocator);
    for (0..6) |i| A.data[i] = @floatFromInt(i + 1);

    var B = try num.NDArray(f64).ones(allocator, &.{3, 2});
    defer B.deinit(allocator);
    for (0..6) |i| B.data[i] = @floatFromInt(i + 1);

    std.debug.print("Matrix A (2×3):\n", .{});
    try A.print();
    std.debug.print("\nMatrix B (3×2):\n", .{});
    try B.print();

    // Matrix multiplication
    var C = try num.linalg.matmul(f64, allocator, &A, &B);
    defer C.deinit(allocator);

    std.debug.print("\nA × B (2×2):\n", .{});
    try C.print();
}
```

**Output:**
```
Matrix A (2×3):
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]

Matrix B (3×2):
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]]

A × B (2×2):
[[22.0, 28.0],
 [49.0, 64.0]]
```

## Next Steps

Now that you've got the basics, explore:

- **[NDArray Guide](./ndarray.md)**: Deep dive into array creation and manipulation
- **[Operations](./operations.md)**: Learn about element-wise operations
- **[Linear Algebra](./linalg.md)**: Matrix operations and decompositions
- **[Statistics](./stats.md)**: Statistical functions and analysis
- **[Machine Learning](./ml.md)**: Neural networks and optimization

## Common Patterns

### Memory Management

Always use `defer` to clean up:

```zig
var arr = try num.NDArray(f64).zeros(allocator, &.{100});
defer arr.deinit(allocator); // Automatically freed at scope end
```

### Error Handling

Most functions return errors that should be handled:

```zig
const result = try num.linalg.inverse(f64, allocator, &matrix);
// or
const result = num.linalg.inverse(f64, allocator, &matrix) catch |err| {
    std.debug.print("Error: {}\n", .{err});
    return err;
};
```

### Type Flexibility

Choose the right type for your use case:

```zig
var float_arr = try num.NDArray(f64).zeros(allocator, &.{10});  // Scientific
var int_arr = try num.NDArray(i32).zeros(allocator, &.{10});    // Counting
var byte_arr = try num.NDArray(u8).zeros(allocator, &.{10});    // Images
```

## Tips & Tricks

1. **Use `try` for cleaner error handling**
2. **Always `defer` your `deinit()` calls**
3. **Prefer `f64` for most scientific computing**
4. **Use `const` for arrays that won't change**
5. **Check array shapes before operations**

## Need Help?

- 📖 [Full Documentation](../api/overview.md)
- 💬 [GitHub Discussions](https://github.com/muhammad-fiaz/num.zig/discussions)
- 🐛 [Report Issues](https://github.com/muhammad-fiaz/num.zig/issues)
- ⭐ [Star on GitHub](https://github.com/muhammad-fiaz/num.zig)
